'''
20022.04.22

AATM versition 5

基于version 4 的改动

backbone提取的特征进入axial transformer block 堆叠两次，得到的输入作为conv1的输入，

不再有(8, 3, 256, 256)的输入 

backbone 新加一个参数scale

尝试用原始维度*scale的维度提取特征，transformer之后再用3D卷积降到原始维度，这样可以使得backbone提取的特征数更多


先把scale设置为1，和version4对比一下，然后在结果好的版本上调scale到4和scale=1对比

'''
from pickle import FALSE
import sys
from sklearn.preprocessing import scale
sys.path.append('../')

import torch
import torch.nn as nn
from networks.init_weight import *
from networks.init_weight import *
from networks.utils_layers import MLP, conv_block, up_conv, AxialPositionalEmbedding, AxialAttention, DSV
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
        # x = x.flatten(2).transpose(1,2)
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


class AxialTransformerBlock(nn.Module):

    def __init__(self, embed_dim, attention_heads) -> None:
        super().__init__()

        self.axial_attention = AxialAttention(
            dim=embed_dim,
            num_dimensions=3,
            heads=attention_heads,
            dim_heads=None,
            dim_index=1,
            sum_axial_out=True
        )

        self.MLP = MLP(embedding_dim=embed_dim, mlp_dim=embed_dim * 4)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        '''
        param: x
        shape: (batch, 384, 3, 16, 16)
        return: (batch, 384, 3, 16, 16)
        '''
        _x = self.axial_attention(x)
        _x = self.dropout(_x)
        x = x + _x

        x = x.flatten(2).transpose(1, 2) # (batch, 384, 3, 16, 16) -> (batch, 768, 384)
        x = self.layer_norm1(x)

        _x = self.MLP(x)
        x = x + _x
        x = self.layer_norm2(x)

        batch, n_patch, hidden = x.size()
        slice, h, w = 3, int(math.sqrt(n_patch // 3)), int(math.sqrt(n_patch // 3)), 
        f_reshape = x.permute(0, 2, 1).contiguous().view(batch, hidden, slice, h, w)

        return f_reshape

class AxialTransformer(nn.Module):
    def __init__(self, embed_dim, attention_heads, depth=4):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [AxialTransformerBlock(embed_dim, attention_heads) for _ in range(depth)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x
        
class AATM(nn.Module):

    def __init__(self, in_ch, out_ch, img_size, patch_size, embed_dim, attention_heads, block_nums=None) -> None:
        super(AATM, self).__init__()

        self.img_size = (img_size, img_size)

        self.single_slice_embedding = SingleEmbedding(in_ch, img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        # self.all_slice_embedding = AllEmbedding(in_ch, img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.pos_embed = AxialPositionalEmbedding(
            dim=embed_dim, 
            shape=(3, 16, 16), 
            emb_dim_index=1
        )

        index = int(math.log2(256 // img_size))
        depth = block_nums[index]

        self.axial_transformer = AxialTransformer(embed_dim, attention_heads, depth)

        self.conv3D = nn.Conv3d(embed_dim, out_ch, kernel_size=(3,1,1), stride=1, padding=0)
        
    def forward(self, features):

        '''
        features: (batch, channel, z, x, y)

        return: (batch, channel, x, y)
        '''
        f_l = features[:,:,0]
        f_k = features[:,:,1]
        f_u = features[:,:,2]
        # f_all = features

        E_l = self.single_slice_embedding(f_l).unsqueeze(2) # (batch, 384, 1, 16, 16)
        E_u = self.single_slice_embedding(f_u).unsqueeze(2) # (batch, 384, 1, 16, 16)
        E_k = self.single_slice_embedding(f_k).unsqueeze(2) # (batch, 384, 1, 16, 16)
        
        f_qkv = torch.cat([E_l, E_u, E_k], dim=2) # (batch, 384, 3, 16, 16)
        f_embed = self.pos_embed(f_qkv)
        
        attention_output = self.axial_transformer(f_embed)
        
        f_fuse = self.conv3D(attention_output).squeeze()

        if attention_output.shape[0] == 1:
            f_fuse = f_fuse.unsqueeze(0)    
        
        f_AATM = F.interpolate(f_fuse, self.img_size, mode='bilinear')
        
        return f_AATM
    

class AxialBlock(nn.Module):

    def __init__(self, in_ch, out_ch, img_size, patch_size, embed_dim, attention_heads, block_nums):
        super(AxialBlock, self).__init__()

        self.AATM = AATM(in_ch, out_ch, img_size, patch_size, embed_dim, attention_heads, block_nums)

    def forward(self, features):
        
        '''
        features: type: []; size: 3, which are lower, key, upper slices, seperately.
        '''
        f_l = features[0].unsqueeze(2)
        f_k = features[1].unsqueeze(2)
        f_u = features[2].unsqueeze(2)

        f_cat = torch.cat([f_l, f_k, f_u], dim=2)

        f_AATM = self.AATM(f_cat)

        return f_AATM

class BackBone(nn.Module):

    def __init__(self, in_ch, out_ch, img_size, patch_size=16, embed_dim=384, attention_heads=12, block_nums=[2,2,4,2], scale=4) -> None:
        super(BackBone, self).__init__()

        self.conv = nn.ModuleList([nn.Sequential(
                nn.Conv2d(in_ch, out_ch*scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch*scale),
                nn.ReLU(inplace=True)
            )] * 3)
        
        self.axial_block = AxialBlock(out_ch*scale, out_ch, img_size, patch_size, embed_dim, attention_heads, block_nums)

    def forward(self, x):
        '''
        param: x : [] * 3; []->shape: (batch, channel, h, w)
        '''
        features = [slice_conv(x[idx]) for idx, slice_conv in enumerate(self.conv)]

        f_AATM = self.axial_block(features)

        return features, f_AATM



class Unet_AATM(nn.Module):

    def __init__(self, img_ch, out_ch=1, img_size=256, dsv=False) -> None:
        super(Unet_AATM, self).__init__()

        self.weight = InitWeights(init_type='kaiming')

        self.dsv = dsv

        filters = [64, 128, 256, 512, 1024]

        filters = [x // 4 for x in filters] # [16, 32, 64, 128, 256]

        scale = 1

        self.backbone1 = BackBone(in_ch=img_ch, out_ch=filters[0]//2, img_size=img_size, scale=scale) # 1->16->8
        self.encoder1 = conv_block(in_ch=filters[0]//2, out_ch=filters[0]) # 8->16->16

        self.backbone2 = BackBone(in_ch=filters[0]//2 * scale, out_ch=filters[0], img_size=img_size//2, scale=scale) # 16->32->16
        self.encoder2 = conv_block(in_ch=filters[0], out_ch=filters[1]) # 16->32->32

        self.backbone3 = BackBone(in_ch=filters[0] * scale, out_ch=filters[1], img_size=img_size//4, scale=scale) # 32->64->32
        self.encoder3 = conv_block(in_ch=filters[1], out_ch=filters[2]) # 32->64->64

        self.backbone4 = BackBone(in_ch=filters[1] * scale, out_ch=filters[2], img_size=img_size//8, scale=scale) #64->128->64
        self.encoder4 = conv_block(in_ch=filters[2], out_ch=filters[3]) # 64->128->128

        self.encoder5 = conv_block(in_ch=filters[3], out_ch=filters[4]) # 128->256->256

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        '''定义decoder中的卷积过程'''
        self.up_conv5 = up_conv(in_ch=filters[4], out_ch=filters[3])
        self.up_conv4 = up_conv(in_ch=filters[3], out_ch=filters[2])
        self.up_conv3 = up_conv(in_ch=filters[2], out_ch=filters[1])
        self.up_conv2 = up_conv(in_ch=filters[1], out_ch=filters[0])

        '''定义decoder中的卷积过程'''
        self.decoder4 = conv_block(in_ch=filters[4], out_ch=filters[3])
        self.decoder3 = conv_block(in_ch=filters[3], out_ch=filters[2])
        self.decoder2 = conv_block(in_ch=filters[2], out_ch=filters[1])
        self.decoder1 = conv_block(in_ch=filters[1], out_ch=filters[0])

        self.dsv4 = DSV(in_channel=filters[3], out_channel=out_ch, scale_factor=8)
        self.dsv3 = DSV(in_channel=filters[2], out_channel=out_ch, scale_factor=4)
        self.dsv2 = DSV(in_channel=filters[1], out_channel=out_ch, scale_factor=2)
        self.dsv1 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        '''最后的1X1卷积'''
        if self.dsv is True:
            self.final = nn.Conv2d(out_ch*4, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.apply(self.weight)


    def forward(self, x):
        
        slices_num = None
        
        if len(x.shape) == 4:
            slices_num = x.shape[1]
            _x = x.unsqueeze(1)
            input = [_x[:,:,i,...] for i in range(slices_num)]
        
        # encoder
        features1, f_aatm1 = self.backbone1(input)
        res_encoder1 = self.encoder1(f_aatm1)
        features1_pool = [self.Maxpool(features1[i]) for i in range(slices_num)]
        x2 = self.Maxpool(res_encoder1)
        
        features2, f_aatm2 = self.backbone2(features1_pool)
        res_encoder2 = self.encoder2(x2 + f_aatm2)
        features2_pool = [self.Maxpool(features2[i]) for i in range(slices_num)]
        x3 = self.Maxpool(res_encoder2)

        features3, f_aatm3 = self.backbone3(features2_pool)

        res_encoder3 = self.encoder3(x3 + f_aatm3)
        features3_pool = [self.Maxpool(features3[i]) for i in range(slices_num)]
        x4 = self.Maxpool(res_encoder3) 

        features4, f_aatm4 = self.backbone4(features3_pool)
        res_encoder4 = self.encoder4(x4 + f_aatm4)
        # features4_pool = [self.Maxpool(features4[i]) for i in range(slices_num)]
        x5 = self.Maxpool(res_encoder4) 

        res_encoder5 = self.encoder5(x5) # 最后一次无池化 torch.Size([8, 1024, 16, 16])
        
        # deocer and contact

        de4 = self.up_conv5(res_encoder5)
        de4 = torch.cat((res_encoder4, de4), dim=1)
        de4 = self.decoder4(de4)

        de3 = self.up_conv4(de4)
        de3 = torch.cat((res_encoder3, de3), dim=1)
        de3 = self.decoder3(de3)

        de2 = self.up_conv3(de3)
        de2 = torch.cat((res_encoder2, de2), dim=1)
        de2 = self.decoder2(de2)

        de1 = self.up_conv2(de2)
        de1 = torch.cat((res_encoder1, de1), dim=1)
        de1 = self.decoder1(de1)

        if self.dsv is True:
            dsv4 = self.dsv4(de4)
            dsv3 = self.dsv3(de3)
            dsv2 = self.dsv2(de2)
            dsv1 = self.dsv1(de1)
            final = self.final(torch.cat([dsv4, dsv3, dsv2, dsv1], dim=1))
        else:
            final = self.final(de1) # 最后的 1X1 卷积操作

        return nn.Sigmoid()(final)


def get_Unet_AATM_V5(img_size, dsv=False):

    print('use deep supervision: {}'.format(dsv))
    model = Unet_AATM(1, 1, img_size=img_size, dsv=dsv)
    return model


if __name__ == '__main__':


    x = torch.rand([8, 3, 256, 256])

    model = Unet_AATM(1, 1, 256, False)

    output = model(x)

    print(output.shape)

    total_params = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total_params/1e6))
