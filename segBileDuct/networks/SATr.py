import sys
sys.path.append('../')

import torch
import torch.nn as nn
from networks.init_weight import *
from networks.init_weight import *
from networks.utils_layers import MLP, conv_block, up_conv, DSV, measure_inference_speed
import math
import torch.nn.functional as F



class SingleEmbedding(nn.Module):


    def __init__(self, in_ch, img_size, patch_size, embed_dim=384) -> None:
        super(SingleEmbedding, self).__init__()

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])

        R = 256 // img_size
        r = 256 // (16 * R)

        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=(1,1), padding=0)

        self.avgPool = nn.AvgPool2d(kernel_size=r, stride=r)

    def forward(self, x):

        x = self.proj(x)
        x = self.avgPool(x)        
        x = x.flatten(2).transpose(1,2)
        return x


class AllEmbedding(nn.Module):

    def __init__(self, in_ch, img_size, patch_size, embed_dim=384) -> None:
        super(AllEmbedding, self).__init__()

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])

        R = 256 // img_size
        r = 256 // (16 * R)

        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=(3, r, r), stride=r)


    def forward(self, x):
        
        x = self.proj(x).squeeze()

        '''
        small bug：
        当 batch size 等于 1 的时候，这里会同时压缩掉第一个维度
        '''        
        if (len(x.shape) == 3): 
            x = x.unsqueeze(0)
        x = x.flatten(2).transpose(1,2)

        return x

class MultiHeadAttention(nn.Module):
    
    def __init__(self, embed_dim, num_heads) -> None:
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_attention_heads = num_heads
        self.attention_head_size = int(self.embed_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attn_dropout = nn.Dropout(0.0)
        self.proj_dropout = nn.Dropout(0.0)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.query = nn.Linear(self.embed_dim, self.all_head_size)
        self.key = nn.Linear(self.embed_dim, self.all_head_size)
        self.value = nn.Linear(self.embed_dim, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, features):
        
        '''
        features: type: []; size: 3, which are q, k, v ,sperately
        '''
        mixed_query_layer = self.query(features[0])
        mixed_key_layer = self.key(features[1])
        mixed_value_layer = self.value(features[2])

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class SATr(nn.Module):

    def __init__(self, in_ch, img_size, patch_size, embed_dim, attention_heads) -> None:
        super(SATr, self).__init__()

        self.img_size = (img_size, img_size)
        self.single_slice_embedding = SingleEmbedding(in_ch, img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.all_slice_embedding = AllEmbedding(in_ch, img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.multi_head_attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=attention_heads)

        self.MLP = MLP(embedding_dim=embed_dim, mlp_dim=embed_dim * 4)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, features):
        '''
        features: (batch, channel, z, x, y)
        '''
        f_l = features[:,:,0]
        f_k = features[:,:,1]
        f_u = features[:,:,2]
        f_all = features

        E_l = self.single_slice_embedding(f_l)
        E_u = self.single_slice_embedding(f_u)
        E_k = self.single_slice_embedding(f_k)
        E_all = self.all_slice_embedding(f_all)

        f_q = f_k = E_l + E_u
        f_v = E_k + E_all

        attention_input = [f_q, f_k, f_v]

        _f_msa = self.multi_head_attention(attention_input)
        _f_msa = self.dropout(_f_msa)
        f_msa = _f_msa + f_v
        f_msa = self.layer_norm1(f_msa)

        f_mlp = self.MLP(f_msa)
        f_tr = f_mlp + f_msa # 理论上讲，这里输出的应该是 batch×256×384
        f_tr = self.layer_norm2(f_tr)


        # RESHAPE
        B, n_patch, hidden = f_tr.size()
        h, w = int(math.sqrt(n_patch)), int(math.sqrt(n_patch))
        f_reshape = f_tr.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        
        f_SATr = F.interpolate(f_reshape, self.img_size, mode='bilinear')
        
        return f_SATr
    

class HrbridBlock(nn.Module):

    def __init__(self, in_ch, out_ch, img_size, patch_size, embed_dim, attention_heads):
        super(HrbridBlock, self).__init__()

        self.conv3D = nn.Conv3d(in_ch, embed_dim, kernel_size=(3,1,1), stride=1, padding=0)

        self.sliceAttention = SATr(in_ch, img_size, patch_size, embed_dim, attention_heads)

        self.fuse_feature = nn.Conv3d(embed_dim, out_ch, (2,1,1), padding=0)

    def forward(self, features):
        
        '''
        features: type: []; size: 3, which are lower, key, upper slices, seperately.
        '''
        f_l = features[0].unsqueeze(2)
        f_k = features[1].unsqueeze(2)
        f_u = features[2].unsqueeze(2)

        f_cat = torch.cat([f_l, f_k, f_u], dim=2)

        f_fuse = self.conv3D(f_cat) 
        f_SATr = self.sliceAttention(f_cat)
        f_SATr = f_SATr.unsqueeze(2)

        assert len(f_fuse.shape) == len(f_SATr.shape)
        
        f_final = torch.cat([f_fuse, f_SATr], dim=2)

        output = self.fuse_feature(f_final).squeeze()
        
        if len(output.shape) == 3:
            output = output.unsqueeze(0)

        return output



class BackBone(nn.Module):

    def __init__(self, in_ch, out_ch, img_size, patch_size=16, embed_dim=384, attention_heads=12) -> None:
        super(BackBone, self).__init__()

        self.conv1 = nn.ModuleList([nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )] * 3)
        
        self.SATR = HrbridBlock(out_ch, out_ch, img_size, patch_size, embed_dim, attention_heads)

    def forward(self, x):

        features = [slice_conv(x[idx]) for idx, slice_conv in enumerate(self.conv1)]
        
        f_SATR = self.SATR(features)

        return features, f_SATR
    


class Unet_SATR(nn.Module):

    def __init__(self, img_ch, out_ch=1, img_size=256, dsv=False) -> None:
        super(Unet_SATR, self).__init__()

        self.weight = InitWeights(init_type='kaiming')

        self.dsv = dsv

        filters = [32, 64, 128, 256, 512, 1024]

        filters = [x // 4 for x in filters]

        self.backbone1 = BackBone(in_ch=img_ch, out_ch=filters[0], img_size=img_size)
        self.conv1 = conv_block(in_ch=filters[0], out_ch=filters[1])

        self.backbone2 = BackBone(in_ch=filters[0], out_ch=filters[1], img_size=img_size//2)
        self.conv2 = conv_block(in_ch=filters[1], out_ch=filters[2])

        self.backbone3 = BackBone(in_ch=filters[1], out_ch=filters[2], img_size=img_size//4)
        self.conv3 = conv_block(in_ch=filters[2], out_ch=filters[3])

        self.backbone4 = BackBone(in_ch=filters[2], out_ch=filters[3], img_size=img_size//8)
        self.conv4 = conv_block(in_ch=filters[3], out_ch=filters[4])

        self.conv5 = conv_block(in_ch=filters[4], out_ch=filters[5])

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        '''定义decoder中的卷积过程'''
        self.up_conv5 = up_conv(in_ch=filters[5], out_ch=filters[4])
        self.up_conv4 = up_conv(in_ch=filters[4], out_ch=filters[3])
        self.up_conv3 = up_conv(in_ch=filters[3], out_ch=filters[2])
        self.up_conv2 = up_conv(in_ch=filters[2], out_ch=filters[1])

        '''定义decoder中的卷积过程'''
        self.conv4_ = conv_block(in_ch=filters[5], out_ch=filters[4])
        self.conv3_ = conv_block(in_ch=filters[4], out_ch=filters[3])
        self.conv2_ = conv_block(in_ch=filters[3], out_ch=filters[2])
        self.conv1_ = conv_block(in_ch=filters[2], out_ch=filters[1])

        self.dsv4 = DSV(in_channel=filters[4], out_channel=out_ch, scale_factor=8)
        self.dsv3 = DSV(in_channel=filters[3], out_channel=out_ch, scale_factor=4)
        self.dsv2 = DSV(in_channel=filters[2], out_channel=out_ch, scale_factor=2)
        self.dsv1 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        '''最后的1X1卷积'''

        if self.dsv is True:
            self.final = nn.Conv2d(out_ch*4, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.final = nn.Conv2d(filters[1], out_ch, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

        self.apply(self.weight)


    def forward(self, x):
        
        slices_num = None
        
        if len(x.shape) == 4:
            slices_num = x.shape[1]
            x = x.unsqueeze(1)
            input = [x[:,:,i,...] for i in range(slices_num)]
        
        # encoder
        features1, f_satr1 = self.backbone1(input)
        res_conv1 = self.conv1(f_satr1)
        features1_pool = [self.Maxpool(features1[i]) for i in range(slices_num)]
        x2 = self.Maxpool(res_conv1)
        
        features2, f_satr2 = self.backbone2(features1_pool)

        res_conv2 = self.conv2(f_satr2 + x2)
        features2_pool = [self.Maxpool(features2[i]) for i in range(slices_num)]
        x3 = self.Maxpool(res_conv2)

        features3, f_satr3 = self.backbone3(features2_pool)
        res_conv3 = self.conv3(f_satr3 + x3)
        features3_pool = [self.Maxpool(features3[i]) for i in range(slices_num)]
        x4 = self.Maxpool(res_conv3) 

        features4, f_satr4 = self.backbone4(features3_pool)
        res_conv4 = self.conv4(f_satr4 + x4)
        features4_pool = [self.Maxpool(features4[i]) for i in range(slices_num)]
        x5 = self.Maxpool(res_conv4) 

        res_conv5 = self.conv5(x5) # 最后一次无池化 torch.Size([8, 1024, 16, 16])
        
        # deocer and contact

        de4 = self.up_conv5(res_conv5)
        de4 = torch.cat((res_conv4, de4), dim=1)
        de4 = self.conv4_(de4)

        de3 = self.up_conv4(de4)
        de3 = torch.cat((res_conv3, de3), dim=1)
        de3 = self.conv3_(de3)

        de2 = self.up_conv3(de3)
        de2 = torch.cat((res_conv2, de2), dim=1)
        de2 = self.conv2_(de2)

        de1 = self.up_conv2(de2)
        de1 = torch.cat((res_conv1, de1), dim=1)
        de1 = self.conv1_(de1)

        if self.dsv is True:
            dsv4 = self.dsv4(de4)
            output4 = self.sigmoid(dsv4)

            dsv3 = self.dsv3(de3)
            output3 = self.sigmoid(dsv3)

            dsv2 = self.dsv2(de2)
            output2 = self.sigmoid(dsv2)

            dsv1 = self.dsv1(de1)
            output1 = self.sigmoid(dsv1)
            return [output4, output3, output1, output1]
            
        else:
            final = self.final(de1) # 最后的 1X1 卷积操作
            output = self.sigmoid(final)
            return output



def get_Unet_SATR(img_size, dsv=False):

    model = Unet_SATR(1, 1, img_size=img_size, dsv=dsv)
    return model

from torchstat import stat

if __name__ == '__main__':


    x = torch.rand([1, 3, 256, 256])

    model = Unet_SATR(1, 1, 256)

    total_params = sum([param.nelement() for param in model.parameters()])
    
    # stat(model, (3,256,256))
    print("Number of parameter: %.2fM" % (total_params/1e6))

    model = model.cuda()
    model.eval()
    print('fps: ', measure_inference_speed(model, (x.cuda(),)))
