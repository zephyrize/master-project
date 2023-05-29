import sys
sys.path.append('../')

import torch
import torch.nn as nn
from networks.init_weight import *
from networks.utils_layers import *

class Unet(nn.Module):
    '''U-net model'''

    def __init__(self, img_ch=1, output_ch=1):
        super(Unet, self).__init__()

        self.weight = InitWeights(init_type='kaiming')

        filters = [64, 128, 256, 512, 1024]

        filters = [x // 4 for x in filters]

        '''定义encoder中的卷积过程'''
        self.conv1 = conv_block(in_ch=img_ch, out_ch=filters[0])
        self.conv2 = conv_block(in_ch=filters[0], out_ch=filters[1])
        self.conv3 = conv_block(in_ch=filters[1], out_ch=filters[2])
        self.conv4 = conv_block(in_ch=filters[2], out_ch=filters[3])
        self.conv5 = conv_block(in_ch=filters[3], out_ch=filters[4])
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
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
        # encoder
        res_conv1 = self.conv1(x)
        x2 = self.Maxpool(res_conv1)

        res_conv2 = self.conv2(x2)
        x3 = self.Maxpool(res_conv2) 

        res_conv3 = self.conv3(x3)
        x4 = self.Maxpool(res_conv3) 

        res_conv4 = self.conv4(x4)
        x5 = self.Maxpool(res_conv4)

        res_conv5 = self.conv5(x5) # 最后一次无池化
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




from torchstat import stat

if __name__ == '__main__':

    
    model = Unet(3, 1)

    total_params = sum([param.nelement() for param in model.parameters()])
    

    print(model(torch.randn(1, 3, 256, 256)).shape)

    input_size = (3, 256, 256)
    
    # stat(model, input_size)
    print("Number of parameter: %.2fM" % (total_params/1e6))

    model = model.cuda()
    model.eval()

    print('fps: ', measure_inference_speed(model, (torch.randn(1, 3, 256, 256).cuda(),)))
    
    pass