
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils import spectral_norm
from operator import itemgetter
import torch.nn.functional as F
from torch.nn import init
import sys
sys.path.append('../../../')

from medseg.models.segmentation_models.unet_parts import *

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)
        return x
    

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):

        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):

        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial
    

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
        
    return permutations

class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True):
        # assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out
    

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index = 1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        self.num_axials = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):

        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x
    

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
        shape: (batch, 384, z, 16, 16)
        return: (batch, 384, z, 16, 16)
        '''
        b, embed_dim, slices, p_h, p_w = x.size()

        _x = self.axial_attention(x)
        _x = self.dropout(_x)
        x = x + _x

        x = x.flatten(2).transpose(1, 2) # (batch, 384, 3, 16, 16) -> (batch, 768, 384)
        x = self.layer_norm1(x)

        _x = self.MLP(x)
        x = x + _x
        x = self.layer_norm2(x)

        f_reshape = x.permute(0, 2, 1).contiguous().view(b, embed_dim, slices, p_h, p_w)

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

    def __init__(self, in_ch, img_size, sample_slices, patch_size, embed_dim, attention_heads, block_nums=None) -> None:
        super(AATM, self).__init__()

        self.single_slice_embedding = SingleEmbedding(in_ch, img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.pos_embed = AxialPositionalEmbedding(
            dim=embed_dim, 
            shape=(sample_slices, patch_size, patch_size), 
            emb_dim_index=1
        )

        index = int(math.log2(256 // img_size))
        depth = block_nums[index]

        self.axial_transformer = AxialTransformer(embed_dim, attention_heads, depth)
        
    def forward(self, features):

        '''
        features: (batch, channel, z, x, y)

        return: (batch, channel, z, x, y)
        '''

        slices_num = features.shape[2]

        embedding_features = [self.single_slice_embedding(features[:,:,i]).unsqueeze(2) for i in range(slices_num)]
        
        f_qkv = torch.cat(embedding_features, dim=2) # (batch, 384, 3, 16, 16)
        f_embed = self.pos_embed(f_qkv)
        
        attention_output = self.axial_transformer(f_embed)

        return attention_output
    

class AxialBlock(nn.Module):

    def __init__(self, in_ch, img_size, sample_slices, patch_size, embed_dim, attention_heads, block_nums):
        super(AxialBlock, self).__init__()

        self.img_size = (img_size, img_size)

        self.sample_slices = sample_slices

        self.AATM = AATM(in_ch, img_size, sample_slices, patch_size, embed_dim, attention_heads, block_nums)

        self.conv3D = nn.Conv3d(embed_dim, in_ch, kernel_size=(sample_slices, 1, 1), stride=1, padding=0)

        self.conv_1x1 = nn.Conv3d(embed_dim, 1, kernel_size=(1, 1, 1), stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()
        
        # from config import args # 这里写法不太好。不管了，先把实验跑了再说吧 2022-07-26
        
        self.slice_weight = nn.Parameter(torch.ones([sample_slices//2, 256, 256])/2, requires_grad=False)
        
        if sample_slices >= 5 :
            for i in range(sample_slices//2 - 1):
                self.slice_weight[i] = 0.0

    def forward(self, features):
        
        '''
        features: type: []; size: 3, which are lower, key, upper slices, seperately.
        '''

        f_expand = [f.unsqueeze(2) for f in features]

        f_cat = torch.cat(f_expand, dim=2)

        f_AATM = self.AATM(f_cat)


        '''
        第一个分支: 压缩切片特征到四维, 然后指导第一个编码器
        '''
        f_fuse = self.conv3D(f_AATM).squeeze()
        # 这里要保证维度问题 (batch = 1时, squeeze会压缩掉batch维度
        if f_AATM.shape[0] == 1:
            f_fuse = f_fuse.unsqueeze(0)    
        f_reshape = F.interpolate(f_fuse, self.img_size, mode='bilinear')


        '''
        第二个分支, 生成伪标签
        '''

        f_auxiliary = self.conv_1x1(f_AATM).squeeze()
        if f_AATM.shape[0] == 1:
            f_auxiliary = f_auxiliary.unsqueeze(0)  
        assert len(f_auxiliary.shape) == 4

        f_auxiliary = F.interpolate(f_auxiliary, (256, 256),  mode='bilinear') # 修改：每个分支都插值到256 256

        f_prop = self.sigmoid(f_auxiliary)

        # f_prop = self.get_pseudo_label(f_prop)
        f_prop = None

        return f_reshape, f_prop

    def get_pseudo_label(self, f_prop):

        false_gt = torch.zeros(*f_prop[:,0,...].shape).cuda()

        for i in range(self.sample_slices//2):
            false_gt += self.slice_weight[i] * (f_prop[:,i,...]+f_prop[:,self.sample_slices-i-1,...])

        false_gt[false_gt>1.0] = 1.0
        mid_slice = f_prop[:,self.sample_slices//2,...]

        assert false_gt.shape == mid_slice.shape

        return [mid_slice, false_gt]
    
class BackBone(nn.Module):
    
    def __init__(self, in_ch, out_ch, img_size, sample_slices=3, patch_size=16, embed_dim=384, attention_heads=12, block_nums=[2,2,4,2]) -> None:
        super(BackBone, self).__init__()

        self.conv = nn.ModuleList([nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                
            )] * sample_slices)
        
        self.axial_block = AxialBlock(out_ch, img_size, sample_slices, patch_size, embed_dim, attention_heads, block_nums)

    def forward(self, x):
        '''
        param: x : [] * 3; []->shape: (batch, channel, h, w)
        '''
        features = [slice_conv(x[idx]) for idx, slice_conv in enumerate(self.conv)]
        
        f_AATM, f_prop = self.axial_block(features)

        return features, f_AATM, f_prop
    

class Bottleneck(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(Bottleneck, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):

        out = self.conv(x)
        return out
    


class Encoder(nn.Module):
    
    def __init__(self, in_ch, out_ch) -> None:
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x, axial_feature):
        x = self.conv1(x)
        x = x + axial_feature
        out = self.conv2(x)
        return out



class SATNet(nn.Module):

    def __init__(self, img_ch=1, out_ch=1, img_size=256, dsv=False, sample_slices=3, scale=4):
        super(SATNet, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        filters = [x // scale for x in filters] # [16, 32, 64, 128, 256]

        self.encoder1 = Encoder(in_ch=img_ch*sample_slices, out_ch=filters[0]) # 3->16->16
        self.encoder2 = Encoder(in_ch=filters[0], out_ch=filters[1]) # 16->32->32
        self.encoder3 = Encoder(in_ch=filters[1], out_ch=filters[2]) # 32->64->64
        self.encoder4 = Encoder(in_ch=filters[2], out_ch=filters[3]) # 128->128->128
        self.encoder5 = Bottleneck(in_ch=filters[3], out_ch=filters[4]) # 128->256->256

        self.backbone1 = BackBone(in_ch=img_ch, out_ch=filters[0], img_size=img_size, sample_slices=sample_slices) # 1->16->16
        self.backbone2 = BackBone(in_ch=filters[0], out_ch=filters[1], img_size=img_size//2, sample_slices=sample_slices) # 16->32->32
        self.backbone3 = BackBone(in_ch=filters[1], out_ch=filters[2], img_size=img_size//4, sample_slices=sample_slices)
        self.backbone4 = BackBone(in_ch=filters[2], out_ch=filters[3], img_size=img_size//8, sample_slices=sample_slices)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        slices_num = None
        
        if len(x.shape) == 4:
            slices_num = x.shape[1]
            _x = x.unsqueeze(1)
            input = [_x[:,:,i,...] for i in range(slices_num)]

        features = []
        # encoder
        features1, f_aatm1, slice_prop_1 = self.backbone1(input)
        res_encoder1 = self.encoder1(x, f_aatm1)
        features1_pool = [self.Maxpool(features1[i]) for i in range(slices_num)]
        x2 = self.Maxpool(res_encoder1)

        features.append(res_encoder1)
        
        features2, f_aatm2, slice_prop_2 = self.backbone2(features1_pool)
        res_encoder2 = self.encoder2(x2, f_aatm2)
        features2_pool = [self.Maxpool(features2[i]) for i in range(slices_num)]
        x3 = self.Maxpool(res_encoder2)

        features.append(res_encoder2)

        features3, f_aatm3, slice_prop_3 = self.backbone3(features2_pool)
        res_encoder3 = self.encoder3(x3, f_aatm3)
        features3_pool = [self.Maxpool(features3[i]) for i in range(slices_num)]
        x4 = self.Maxpool(res_encoder3) 
        
        features.append(res_encoder3)

        features4, f_aatm4, slice_prop_4 = self.backbone4(features3_pool)
        res_encoder4 = self.encoder4(x4, f_aatm4)
        # features4_pool = [self.Maxpool(features4[i]) for i in range(slices_num)]
        x5 = self.Maxpool(res_encoder4) 

        features.append(res_encoder4)

        res_encoder5 = self.encoder5(x5)

        return res_encoder5, features


class TAGEncoder(nn.Module):
    

    def __init__(self, input_channel=1, z_level_1_channel=None, z_level_2_channel=None, feature_reduce=1, encoder_dropout=None, norm=nn.InstanceNorm2d, if_SN=False, res_conn=False):
        super(TAGEncoder, self).__init__()

        self.res_connection = res_conn
        self.my_encoder = SATNet(img_ch=input_channel, out_ch=1, img_size=256, dsv=False, sample_slices=3, scale=feature_reduce * 2)
        

        self.code_decoupler = nn.Sequential(
            spectral_norm(nn.Conv2d(z_level_1_channel,
                                    z_level_2_channel, 3, padding=1, bias=True)),
            norm(z_level_2_channel),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(z_level_2_channel,
                                    z_level_2_channel, 3, padding=1, bias=True)),
            norm(z_level_2_channel),
            nn.ReLU(),
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def filter_code(self, z):
        z_s = self.code_decoupler(z)
        return z_s

    def forward(self, x):

        z_i, features = self.my_encoder(x)
        z_s = self.filter_code(z_i)

        if self.res_connection is True:
            return z_i, z_s, features
        else:
            return z_i, z_s



class TAGDecoder(nn.Module):
    def __init__(self, input_channel, output_channel=2,  feature_scale=2, decoder_dropout=None, norm=nn.BatchNorm2d, if_SN=False, last_layer_act=None):
        super(TAGDecoder, self).__init__()

        self.up1 = up(512 // feature_scale, 256 // feature_scale, 256 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up2 = up(256 // feature_scale, 128 // feature_scale, 128 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up3 = up(128 // feature_scale, 64 // feature_scale, 64 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up4 = up(64 // feature_scale, 32 // feature_scale, 32 // feature_scale,
                      norm=norm, dropout=decoder_dropout, if_SN=if_SN)

        self.outc = outconv(32 // feature_scale, output_channel)
        self.last_act = last_layer_act

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x, features):
        
        x = self.up1(x, features[-1])
        x = self.up2(x, features[-2])
        x = self.up3(x, features[-3])
        x = self.up4(x, features[-4])

        x = self.outc(x)

        if not self.last_act is None:
            x = self.last_act(x)

        return x

if __name__ == '__main__':

    encoder = TAGEncoder(input_channel=1, z_level_1_channel=256, z_level_2_channel=256, feature_reduce=2, res_conn=True);
    
    decoder = TAGDecoder(input_channel=512//2,  feature_scale=2, num_classes=2)
    
    input = torch.rand(1, 3, 256, 256)
    z_i, z_s, features = encoder(input)

    output = decoder(z_i, features)
    print(z_i.shape, z_s.shape, output.shape)

