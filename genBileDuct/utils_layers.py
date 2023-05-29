import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** 1 / 2

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


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


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=True, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape

        project = self.projection(img_patches)
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size)

        patches = torch.cat([token, project], dim=1)
        patches += self.embedding[:tokens + 1, :]

        x = self.dropout(patches)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        return x




class conv_block(nn.Module):

    
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):

    '''逆卷积'''
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
 
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)
        return x




class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi



from math import perm
from requests import head
import torch
from torch import device, nn
from operator import itemgetter
# from reversible import ReversibleSequence

# helper functions

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape

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

# helper classes

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x

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

# axial pos emb

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

# attention

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

# axial attention class

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



class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

class AxialAttention_3D(nn.Module):
    def __init__(self, in_channel, out_channel, attention_heads=8, kernel_size=56, axial='width'):
        assert (in_channel % attention_heads == 0) and (out_channel % attention_heads == 0)
        super(AxialAttention_3D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.attention_heads = attention_heads
        self.dim_heads = out_channel // attention_heads
        self.kernel_size = kernel_size
        self.axial = axial

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_channel, out_channel * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_channel * 2)
        self.bn_similarity = nn.BatchNorm2d(attention_heads * 3)
        self.bn_output = nn.BatchNorm1d(out_channel * 2)

        # Priority on encoding

        ## Initial values 

        self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False) 
        self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)


        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.dim_heads * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))

        self.reset_parameters()
        # self.print_para()

    def forward(self, x):

        if self.axial == 'width':
            x = x.permute(0, 2, 3, 1, 4)
        elif self.axial == 'height':
            x = x.permute(0, 2, 4, 1, 3)  # N, W, C, H
        else:
            x = x.permute(0, 3, 4, 1, 2)

        N, D, W, C, H = x.shape
        x = x.contiguous().view(N * D * W, C, H)

        # print('reshape: ', x.shape)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * D * W, self.attention_heads, self.dim_heads * 2, H), [self.dim_heads // 2, self.dim_heads // 2, self.dim_heads], dim=2)
        
        # print('q shape: ', q.shape)
        # print('k shape: ', k.shape)
        # print('v shape: ', v.shape)

        # print('relative shape: ', self.relative.shape)

        # Calculate position embedding

        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.dim_heads * 2, self.kernel_size, self.kernel_size)
        # print('all embeddings shape', all_embeddings.shape)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.dim_heads // 2, self.dim_heads // 2, self.dim_heads], dim=0)
        
        # print('q embedding shape: ', q_embedding.shape)
        # print('k embedding shape: ', k_embedding.shape)
        # print('v embedding shape: ', v_embedding.shape)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        # print('qr shape: ', qr.shape)
        # print('kr shape: ', kr.shape)
        # print('qk shape: ', qk.shape)

        # multiply by factors
        qr = torch.mul(qr, self.f_qr) # G_Q
        kr = torch.mul(kr, self.f_kr) # G_K
        

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * D * W, 3, self.attention_heads, H, H).sum(dim=1)
        
        #print('stacked_similarity shape', stacked_similarity.shape)

        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        # print('similarity shape: ', similarity.shape)

        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # print('sv shape', sv.shape)
        # print('sve shape', sve.shape)

        # multiply by factors

        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        # print('sv shape', sv.shape)
        # print('sve shape', sve.shape)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * D * W, self.out_channel * 2, H)
        # print('stacked output shape: ', stacked_output.shape)

        output = self.bn_output(stacked_output).view(N, D, W, self.out_channel, 2, H).sum(dim=-2)

        # print('output shape: ', output.shape)

        if self.axial == 'width':
            output = output.permute(0, 3, 1, 2, 4)
        elif self.axial == 'height':
            output = output.permute(0, 3, 1, 4, 2)
        else:
            output = output.permute(0, 3, 4, 1, 2)

        return output
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_channel))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.dim_heads))




class DSV(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor):
        super(DSV, self).__init__()

        self.dsv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0), 
            nn.Upsample(scale_factor=scale_factor, mode ='bilinear')
        )
    
    def forward(self, x):
        return self.dsv(x)




class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out





# from networks.ResNet import ResNet
# from networks.ResNet import resnet50

# class TAM(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  n_segment,
#                  kernel_size=3,
#                  stride=1,
#                  padding=1):
#         super(TAM, self).__init__()
#         self.in_channels = in_channels
#         self.n_segment = n_segment
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         print('TAM with kernel_size {}.'.format(kernel_size))

#         self.G = nn.Sequential(
#             nn.Linear(n_segment, n_segment * 2, bias=False),
#             nn.BatchNorm1d(n_segment * 2), nn.ReLU(inplace=True),
#             nn.Linear(n_segment * 2, kernel_size, bias=False), nn.Softmax(-1))

#         self.L = nn.Sequential(
#             nn.Conv1d(in_channels,
#                       in_channels // 4,
#                       kernel_size,
#                       stride=1,
#                       padding=kernel_size // 2,
#                       bias=False), nn.BatchNorm1d(in_channels // 4),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
#             nn.Sigmoid())

#     def forward(self, x):
#         # x.size = N*C*T*(H*W)
#         nt, c, h, w = x.size()
#         t = self.n_segment
#         n_batch = nt // t
#         new_x = x.view(n_batch, t, c, h, w).permute(0, 2, 1, 3,
#                                                      4).contiguous()
#         out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1))
#         out = out.view(-1, t)
#         conv_kernel = self.G(out.view(-1, t)).view(n_batch * c, 1, -1, 1)
#         local_activation = self.L(out.view(n_batch, c,
#                                            t)).view(n_batch, c, t, 1, 1)
#         new_x = new_x * local_activation
#         out = F.conv2d(new_x.view(1, n_batch * c, t, h * w),
#                        conv_kernel,
#                        bias=None,
#                        stride=(self.stride, 1),
#                        padding=(self.padding, 0),
#                        groups=n_batch * c)
#         out = out.view(n_batch, c, t, h, w)
#         out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)

#         return out


# class TemporalBottleneck(nn.Module):
#     def __init__(self,
#                  net,
#                  n_segment=8,
#                  t_kernel_size=3,
#                  t_stride=1,
#                  t_padding=1):
#         super(TemporalBottleneck, self).__init__()
#         self.net = net
#         # assert isinstance(net, torchvision.models.Bottleneck)
#         self.n_segment = n_segment
#         self.tam = TAM(in_channels=net.conv1.out_channels,
#                        n_segment=n_segment,
#                        kernel_size=t_kernel_size,
#                        stride=t_stride,
#                        padding=t_padding)

#     def forward(self, x):
#         identity = x

#         out = self.net.conv1(x)
#         out = self.net.bn1(out)
#         out = self.net.relu(out)
#         out = self.tam(out)

#         out = self.net.conv2(out)
#         out = self.net.bn2(out)
#         out = self.net.relu(out)

#         out = self.net.conv3(out)
#         out = self.net.bn3(out)

#         if self.net.downsample is not None:
#             identity = self.net.downsample(x)

#         out += identity
#         out = self.net.relu(out)

#         return out


# def make_temporal_modeling(net,
#                            n_segment=8,
#                            t_kernel_size=3,
#                            t_stride=1,
#                            t_padding=1):
#     if isinstance(net, ResNet):

#         print('continue...')
#         n_round = 1

#         def make_block_temporal(stage,
#                                 this_segment,
#                                 t_kernel_size=3,
#                                 t_stride=1,
#                                 t_padding=1):
#             blocks = list(stage.children())
#             print('=> Processing this stage with {} blocks residual'.format(
#                 len(blocks)))
#             for i, b in enumerate(blocks):
#                 # if i >= len(blocks)//2:
#                 if i % n_round == 0:
#                     blocks[i] = TemporalBottleneck(b, this_segment,
#                                                    t_kernel_size, t_stride,
#                                                    t_padding)
#             return nn.Sequential(*blocks)

#         net.layer1 = make_block_temporal(net.layer1, n_segment, t_kernel_size,
#                                          t_stride, t_padding)
#         net.layer2 = make_block_temporal(net.layer2, n_segment, t_kernel_size,
#                                          t_stride, t_padding)
#         net.layer3 = make_block_temporal(net.layer3, n_segment, t_kernel_size,
#                                          t_stride, t_padding)
#         net.layer4 = make_block_temporal(net.layer4, n_segment, t_kernel_size,
#                                          t_stride, t_padding)


# def get_TA_block():

#     backbone = resnet50()
    
#     make_temporal_modeling(backbone, 3)
    
#     return backbone


# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  

# 整个 ASPP 架构
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[1, 2, 5]):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        
        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)