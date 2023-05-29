import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
from networks.init_weight import *
from networks.utils_layers import *

class AttUnet(nn.Module):
    def __init__(self,img_ch=3, output_ch=1):
        super(AttUnet,self).__init__()
        
        self.weight = InitWeights(init_type='kaiming')


        filters = [64, 128, 256, 512, 1024]

        filters = [x // 4 for x in filters]

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(in_ch=img_ch,out_ch=filters[0])
        self.Conv2 = conv_block(in_ch=filters[0],out_ch=filters[1])
        self.Conv3 = conv_block(in_ch=filters[1],out_ch=filters[2])
        self.Conv4 = conv_block(in_ch=filters[2],out_ch=filters[3])
        self.Conv5 = conv_block(in_ch=filters[3],out_ch=filters[4])

        self.Up5 = up_conv(in_ch=filters[4],out_ch=filters[3])
        self.Att5 = Attention_block(F_g=filters[3],F_l=filters[3],F_int=filters[2])
        self.Up_conv5 = conv_block(in_ch=filters[4], out_ch=filters[3])

        self.Up4 = up_conv(in_ch=filters[3],out_ch=filters[2])
        self.Att4 = Attention_block(F_g=filters[2],F_l=filters[2],F_int=filters[1])
        self.Up_conv4 = conv_block(in_ch=filters[3], out_ch=filters[2])
        
        self.Up3 = up_conv(in_ch=filters[2],out_ch=filters[1])
        self.Att3 = Attention_block(F_g=filters[1],F_l=filters[1],F_int=filters[0])
        self.Up_conv3 = conv_block(in_ch=filters[2], out_ch=filters[1])
        
        self.Up2 = up_conv(in_ch=filters[1],out_ch=filters[0])
        self.Att2 = Attention_block(F_g=filters[0],F_l=filters[0],F_int=filters[0] // 2)
        self.Up_conv2 = conv_block(in_ch=filters[1], out_ch=filters[0])

        self.Conv_1x1 = nn.Conv2d(filters[0],output_ch,kernel_size=1,stride=1,padding=0)

        self.activation = nn.Sigmoid()
        self.apply(self.weight)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.activation(d1)



from torchstat import stat

if __name__ == '__main__':
    
    
    
    model = AttUnet(3, 1)

    total_params = sum([param.nelement() for param in model.parameters()])

    # print(model(torch.randn(8, 3, 256, 256)).shape)
    
    # stat(model, (3,256,256))
    print("Number of parameter: %.2fM" % (total_params/1e6))
    
    model = model.cuda()
    model.eval()

    print('fps: ', measure_inference_speed(model, (torch.randn(1, 3, 256, 256).cuda(),)))
    
    
    pass