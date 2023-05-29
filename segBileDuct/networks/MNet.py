import sys
sys.path.append('../')

import torch
import torch.nn as nn
from networks.init_weight import *
from networks.utils_layers import *




class Encoder1(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super(Encoder1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class Encoder2(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super(Encoder2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=2,dilation=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=2,dilation=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip_feature):
        x = torch.cat([x, skip_feature], dim=1)
        x = self.conv(x)
        return x

class Decoder1(nn.Module):

    '''逆卷积'''
    def __init__(self, in_ch, out_ch):
        super(Decoder1, self).__init__()
 
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_feature):

        x = torch.cat([x, skip_feature], dim=1)
        x = self.conv(x)
        return x

class Decoder2(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super(Decoder2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip_feature):
        x = torch.cat([x, skip_feature], dim=1)
        x = self.conv(x)
        return x

class MNet(nn.Module):


    def __init__(self, img_ch=1, output_ch=1):
        super(MNet, self).__init__()

        self.weight = InitWeights(init_type='kaiming')

        filters = [64, 128, 256, 512, 1024]

        filters = [x // 4 for x in filters] # 16 32 64 128 256

        # encoder 1
        self.encoder1 = nn.ModuleList([])
        pre_ch = img_ch
        for channel in filters[:-1]:
            self.encoder1.append(Encoder1(pre_ch, channel))
            pre_ch = channel

        self.bottom1 = nn.Sequential(
            nn.Conv2d(filters[-2], filters[-1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filters[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[-1], filters[-1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[-1]),
            nn.ReLU(inplace=True)
        )

        # decoder 1
        self.decoder1 = nn.ModuleList([])
        pre_ch = filters[-1]
        for channel in reversed(filters[:-1]):
            self.decoder1.append(Decoder1(pre_ch+channel, channel))
            pre_ch = channel

        self.bottom2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        
        # encoder 2
        self.encoder2 = nn.ModuleList([])
        pre_ch = filters[0]
        for channel in filters[1:-1]:
            self.encoder2.append(Encoder2(pre_ch+channel, channel))
            pre_ch = channel

        self.bottom3 = nn.Sequential(
            nn.Conv2d(filters[-2], filters[-1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[-1], filters[-1], kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(filters[-1]),
            nn.ReLU(inplace=True)
        )

        # decoder 2
        self.decoder2 = nn.ModuleList([])
        pre_ch = filters[-1]
        for channel in reversed(filters[:-1]):
            self.decoder2.append(Decoder2(pre_ch+channel, channel))
            pre_ch = channel

        self.maxPooling = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_1x1 = nn.Conv2d(filters[0], output_ch, kernel_size=1, padding=0, stride=1)
        self.activation = nn.Sigmoid()

        self.apply(self.weight)

    def forward(self, x):

        features = []
        temp_x = x
        for block in self.encoder1:
            temp_x = block(temp_x)
            features.append(temp_x)
            temp_x = self.maxPooling(temp_x)
        
        temp_x = self.bottom1(temp_x)
        
        for idx, block in enumerate(self.decoder1):
            temp_x = self.upsample(temp_x)
            temp_x = block(temp_x, features[-1-idx])

        temp_x = self.bottom2(temp_x)
        
        for idx, block in enumerate(self.encoder2):
            temp_x = self.maxPooling(temp_x)
            temp_x = block(temp_x, features[idx+1])
        
        temp_x = self.bottom3(self.maxPooling(temp_x))

        for idx, block in enumerate(self.decoder2):
            temp_x = self.upsample(temp_x)
            temp_x = block(temp_x, features[-1-idx])


        res = self.conv_1x1(temp_x)
        output = self.activation(res)

        return output
    


from torchstat import stat
from networks.utils_layers import measure_inference_speed

if __name__ == '__main__':

    x = torch.rand([1, 3, 256, 256])
    model = MNet(3, 1)

    total_params = sum([param.nelement() for param in model.parameters()])

    # stat(model, (3,256,256))
    print("Number of parameter: %.2fM" % (total_params/1e6))

    model = model.cuda()
    model.eval()

    print('fps: ', measure_inference_speed(model, (x.cuda(),)))


        


        
