'''
去除单独的conv分支，backbone只有三个切片巻积、一个轴注意力transformer和一个巻积块

轴注意力采用AATM—V1版的注意力块。
'''


import sys
sys.path.append('../')

import torch
import torch.nn as nn
from networks.init_weight import *
from networks.init_weight import *
from networks.utils_layers import MLP, conv_block, up_conv, AxialPositionalEmbedding, AxialAttention, AxialAttention_3D
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

    def __init__(self, in_ch, img_size, patch_size, embed_dim, attention_heads, block_nums=None) -> None:
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

        self.conv3D = nn.Conv3d(embed_dim, in_ch, kernel_size=(3,1,1), stride=1, padding=0)
        
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

    def __init__(self, in_ch, img_size, patch_size, embed_dim, attention_heads, block_nums):
        super(AxialBlock, self).__init__()

        self.AATM = AATM(in_ch, img_size, patch_size, embed_dim, attention_heads, block_nums)

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


class single_conv(nn.Module):
    
    def __init__(self, in_ch, out_ch) -> None:
        super(single_conv, self).__init__()

        if in_ch == 1:
            in_ch = 3
        
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
    def forward(self, x):
        x = self.conv(x)
        return x


class BackBone(nn.Module):

    def __init__(self, in_ch, out_ch, img_size, patch_size=16, embed_dim=384, attention_heads=12, block_nums=[2, 2, 2, 2]) -> None:
        super(BackBone, self).__init__()

        self.conv2 = nn.ModuleList([nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )] * 3)
        
        self.axial_block = AxialBlock(out_ch, img_size, patch_size, embed_dim, attention_heads, block_nums)

    def forward(self, x2):
        

        features_x2 = [slice_conv(x2[idx]) for idx, slice_conv in enumerate(self.conv2)]
        
        f_AATM = self.axial_block(features_x2)


        return features_x2, f_AATM
    

class Unet_AATM(nn.Module):

    def __init__(self, img_ch, out_ch=1, img_size=256) -> None:
        super(Unet_AATM, self).__init__()

        self.weight = InitWeights(init_type='kaiming')

        filters = [32, 64, 128, 256, 512, 1024]

        filters = [x // 4 for x in filters]

        self.backbone1 = BackBone(in_ch=img_ch, out_ch=filters[0], img_size=img_size)
        self.conv1 = single_conv(in_ch=filters[0], out_ch=filters[1])

        self.backbone2 = BackBone(in_ch=filters[0], out_ch=filters[1], img_size=img_size//2)
        self.conv2 = single_conv(in_ch=filters[1], out_ch=filters[2])

        self.backbone3 = BackBone(in_ch=filters[1], out_ch=filters[2], img_size=img_size//4)
        self.conv3 = single_conv(in_ch=filters[2], out_ch=filters[3])

        self.backbone4 = BackBone(in_ch=filters[2], out_ch=filters[3], img_size=img_size//8)
        self.conv4 = single_conv(in_ch=filters[3], out_ch=filters[4])

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

        '''最后的1X1卷积'''
        self.conv_1x1 = nn.Conv2d(filters[1], out_ch, kernel_size=1, stride=1, padding=0)

        self.apply(self.weight)


    def forward(self, x):
        
        slices_num = None
        
        if len(x.shape) == 4:
            slices_num = x.shape[1]
            _x = x.unsqueeze(1)
            input = [_x[:,:,i,...] for i in range(slices_num)]
        
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
        # features4_pool = [self.Maxpool(features4[i]) for i in range(slices_num)]
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

        de = self.conv_1x1(de1) # 最后的 1X1 卷积操作

        return nn.Sigmoid()(de)



def get_Unet_AATM_V3(img_size):

    model = Unet_AATM(1, 1, img_size=img_size)
    return model


if __name__ == '__main__':


    x = torch.rand([8, 3, 256, 256])

    model = Unet_AATM(1, 1, 256)

    output = model(x)

    print(output.shape)

    total_params = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total_params/1e6))
