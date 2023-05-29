import sys
sys.path.append('../')

import torch
import torch.nn as nn
from networks.init_weight import *
from networks.utils_layers import *

from utils.helper import get_device
device = get_device()
class Encoder1(nn.Module):
    
    def __init__(self, in_ch, out_ch) -> None:
        super(Encoder1, self).__init__()
        
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

class Encoder2(nn.Module):
    
    def __init__(self, filters) -> None:
        super(Encoder2, self).__init__()

        backbone = get_TA_block().to(device)

        self.conv = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        self.layers = []
        self.layers.append(backbone.layer1)
        self.layers.append(backbone.layer2)
        self.layers.append(backbone.layer3)
        self.layers.append(backbone.layer4)

        self.dr = nn.ModuleList()

        for i in range(len(self.layers)):
            self.dr.append(
                nn.Conv3d(filters[i], filters[i], kernel_size=(3,1,1), stride=1, padding=0)
            )

    def forward(self, x):

        x = self.conv(x)
        features = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            bs, c, h, w = x.size()
            batch = bs // 3

            feature = x.view(batch, 3, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feature = self.dr[i](feature).squeeze()
            features.append(feature)

        return features

class TAM_Unet(nn.Module):
    '''TAM_Unet model'''

    def __init__(self, img_ch=1, output_ch=1):
        super(TAM_Unet, self).__init__()

        self.weight = InitWeights(init_type='kaiming')

        filters = [64, 128, 256, 512, 1024]

        filters = [x // 4 for x in filters]

        '''第一个编码器'''
        self.encoder1_conv1 = Encoder1(in_ch=img_ch*3, out_ch=filters[0])
        self.encoder1_conv2 = Encoder1(in_ch=filters[0], out_ch=filters[1])
        self.encoder1_conv3 = Encoder1(in_ch=filters[1], out_ch=filters[2])
        self.encoder1_conv4 = Encoder1(in_ch=filters[2], out_ch=filters[3])
        self.encoder1_conv5 = conv_block(in_ch=filters[3], out_ch=filters[4])
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = Encoder2(filters)

        '''定义up_conv过程'''
        self.up_conv5 = up_conv(in_ch=filters[4], out_ch=filters[3])
        self.up_conv4 = up_conv(in_ch=filters[3], out_ch=filters[2])
        self.up_conv3 = up_conv(in_ch=filters[2], out_ch=filters[1])
        self.up_conv2 = up_conv(in_ch=filters[1], out_ch=filters[0])
        
        '''定义decoder中的卷积过程'''
        self.conv4_ = conv_block(in_ch=filters[4], out_ch=filters[3])
        self.conv3_ = conv_block(in_ch=filters[3], out_ch=filters[2])
        self.conv2_ = conv_block(in_ch=filters[2], out_ch=filters[1])
        self.conv1_ = conv_block(in_ch=filters[1], out_ch=filters[0])
        
        '''最后的1X1卷积'''
        self.conv_1x1 = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        self.apply(self.weight)

    def forward(self, x):
        
        slices_num = None
        
        if len(x.shape) == 4:
            batch, slices_num, h, w = x.size()
            _x = x.unsqueeze(1).permute(0, 2, 1, 3, 4).contiguous().view(batch*slices_num, 1, h, w)

            # _x shape: (batch*slices_num, 1, h, w)
        
        # print(_x.shape)
        encoder2_features = self.encoder2(_x)

        res_conv1 = self.encoder1_conv1(x, encoder2_features[0])
        x2 = self.Maxpool(res_conv1)

        res_conv2 = self.encoder1_conv2(x2, encoder2_features[1])
        x3 = self.Maxpool(res_conv2) 

        res_conv3 = self.encoder1_conv3(x3, encoder2_features[2])
        x4 = self.Maxpool(res_conv3) 

        res_conv4 = self.encoder1_conv4(x4, encoder2_features[3])
        x5 = self.Maxpool(res_conv4)

        res_conv5 = self.encoder1_conv5(x5) # 最后一次无池化
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


if __name__ == '__main__':

    
    model = TAM_Unet(1, 1)

    total_params = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total_params/1e6))

    print(model(torch.randn(8, 3, 256, 256)).shape)
    
    pass