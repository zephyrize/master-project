

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter
from networks.init_weight import *

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

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

        # print(x.shape)
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



class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True, stride=1):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

        self.stride = stride
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)

        if self.stride > 1:
            out = self.pooling(out)
        
        return out

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index = 1):
        super().__init__()
        parameters = []
        self.shape = shape
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


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, img_size=None):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        
        # self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        # self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        
        '''
        主要的改动在这个地方，原来的AxialAttention设计的是针对单个轴。
        替换的AxialAttention直接丢进去整个volume进行融合
        '''
        # 核心代码
        self.pos_embedding = AxialPositionalEmbedding(dim=width, shape=tuple(img_size))
        self.axial_attention = AxialAttention(dim=width, num_dimensions=2, heads=groups, dim_index=1,sum_axial_out=False, stride=stride)
        
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        # print('Axial block input shape:', x.shape)
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        # print('before axial attention shape: ', out.shape)
        # out = self.hight_block(out)
        # print('After hight attention shape: ', out.shape)
        # out = self.width_block(out)
        # print('After width attention shape: ', out.shape)
        
        out = self.pos_embedding(out)
        out = self.axial_attention(out)
        
        # print('after axial attention shape: ', out.shape)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        # print('Axial block output shape:', out.shape)
        # print('------------------------------------------')
        return out


class ResAxialAttentionUNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size = (256, 256), imgchan = 1):
        super(ResAxialAttentionUNet, self).__init__()

        self.weight = InitWeights(init_type='kaiming')

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(1) # 8 * s
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], img_size= np.array(img_size))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, img_size=np.array(img_size),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, img_size=np.array(img_size)//2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, img_size=np.array(img_size)//4,
                                       dilate=replace_stride_with_dilation[2])
        
        # Decoder
        self.decoder1 = nn.Conv2d(int(1024*2*s), int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv2d(int(1024*2*s), int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512*s) ,  int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)

        self.adjust   = nn.Conv2d(int(128*s) , num_classes, kernel_size=1, stride=1, padding=0)
        # self.soft     = nn.Softmax(dim=1) # 如果使用softmax 把上行代码的 1 改为num_classes
        self.activation = nn.Sigmoid()
        
        self.apply(self.weight)

    def _make_layer(self, block, planes, blocks, img_size = np.array([256, 256]), stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, img_size=img_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            img_size = img_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, img_size=img_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        
        # AxialAttention Encoder
        # pdb.set_trace()
        #print('original input x shape: ', x.shape)
        # x = self.conv1(x)
        # #print('first conv x shape: ', x.shape)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # #print('second conv x shape: ', x.shape)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        # #print('third conv x shape: ', x.shape)
        # x = self.bn3(x)
        # x = self.relu(x)
        # # print('After 3 conv layers shape: ', x.shape)
        
        # print('input x shape', x.shape)
        x1 = self.layer1(x)
        # print('After layer1 x1 shape: ', x1.shape)
        x2 = self.layer2(x1)
        # print('After layer2 x2 shape: ', x2.shape)
        x3 = self.layer3(x2)
        # print('After layer3 x3 shape: ', x3.shape)
        x4 = self.layer4(x3)
        # print('After layer4 x4 shape: ', x4.shape)  #   torch.Size([1, 256, 6, 8, 8])

        # print('decoder1 shape: ', self.decoder1(x4).shape)
        
        x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
        # print('x shape: ', x.shape)
        # print('x4 shape: ', x4.shape)
        x = torch.add(x, x4)

        x = F.relu(F.interpolate(self.decoder2(x4) , scale_factor=(2,2), mode ='bilinear'))
        # print(x.shape)
        x = torch.add(x, x3)
        x = F.relu(F.interpolate(self.decoder3(x) , scale_factor=(2,2), mode ='bilinear'))
        # print('decoder 3 shape: ', x.shape)# 
        x = torch.add(x, x2)
        x = F.relu(F.interpolate(self.decoder4(x) , scale_factor=(2,2), mode ='bilinear'))
        # print('decoder 4 shape: ', x.shape)
        x = torch.add(x, x1)
        x = F.relu(self.decoder5(x))
        # print('decoder 5 shape: ', x.shape)
        x = self.adjust(F.relu(x))
        # pdb.set_trace()
        return self.activation(x)

    def forward(self, x):
        return self._forward_impl(x)



def get_axial_attention_model(args):

    model = ResAxialAttentionUNet(AxialBlock, [1, 2, 4, 1], s= 0.5, img_size=(256, 256), imgchan=1)

    return model
    
if __name__ == '__main__':
    

    import sys
    sys.path.append('../')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
    model = ResAxialAttentionUNet(AxialBlock, [1, 2, 4, 1], s= 0.5, img_size=(256, 256), imgchan=1).to(device)

    x = torch.randn([1, 1, 256, 256]).to(device)

    out = model(x)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    print("Out shape: ", out.shape)

