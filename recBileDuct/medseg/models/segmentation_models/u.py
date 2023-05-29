import math
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d, activation=nn.ReLU, bias=True, dropout=None):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, norm=norm, activation=activation, bias=bias)
        self.dropout = dropout
        if not self.dropout is None:
            self.drop = nn.Dropout2d(p=dropout)
            print('enable dropout')

    def forward(self, x):
        x = self.conv(x)
        if not self.dropout is None:
            x = self.drop(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d, activation=nn.ReLU, bias=True):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
            norm(out_ch),
            activation(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
            norm(out_ch),
            activation(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

class down(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d, activation=nn.ReLU, bias=True, dropout=None):
        super(down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, norm=norm, activation=activation, bias=bias)
        )
        self.dropout = dropout
        if not self.dropout is None:
            self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.mpconv(x)
        if not self.dropout is None:
            x = self.drop(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, type='bilinear', dropout=None, norm=nn.BatchNorm2d, activation=nn.ReLU):
        super(up, self).__init__()
        self.type = type
        if type == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif type == 'deconv':
            self.up = nn.ConvTranspose2d((in_ch_1 + in_ch_2) // 2, (in_ch_1 + in_ch_2) // 2, 2, stride=2)
        elif type == 'nearest':
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up = None

        self.conv = double_conv(in_ch_1 + in_ch_2, out_ch, norm=norm, activation=activation)
        self.dropout = dropout
        if not self.dropout is None:
            self.drop = nn.Dropout2d(p=dropout)
            print('enable dropout')

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        
        x = torch.cat([x2, x1], dim=1)
        if (not self.dropout is None):
            x = self.drop(x)
        x = self.conv(x)
        return x
    
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, input_channel, num_classes, feature_scale=1, encoder_dropout=None, decoder_dropout=None, norm=nn.BatchNorm2d, last_layer_act=None):
        super(UNet, self).__init__()
        self.inc = inconv(input_channel, 64 // feature_scale, norm=norm, dropout=encoder_dropout)
        self.down1 = down(64 // feature_scale, 128 // feature_scale, norm=norm, dropout=encoder_dropout)
        self.down2 = down(128 // feature_scale, 256 // feature_scale, norm=norm, dropout=encoder_dropout)
        self.down3 = down(256 // feature_scale, 512 // feature_scale, norm=norm, dropout=encoder_dropout)
        self.down4 = down(512 // feature_scale, 512 // feature_scale, norm=norm, dropout=encoder_dropout)
        self.up1 = up(512 // feature_scale, 512 // feature_scale, 256 //
                      feature_scale, norm=norm, dropout=decoder_dropout)
        self.up2 = up(256 // feature_scale, 256 // feature_scale, 128 //
                      feature_scale, norm=norm, dropout=decoder_dropout)
        self.up3 = up(128 // feature_scale, 128 // feature_scale, 64 //
                      feature_scale, norm=norm, dropout=decoder_dropout)
        self.up4 = up(64 // feature_scale, 64 // feature_scale, 64 // feature_scale,
                      norm=norm, dropout=decoder_dropout)

        self.outc = outconv(64 // feature_scale, num_classes)
        self.n_classes = num_classes
        self.attention_map = None
        self.last_act = last_layer_act

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        self.hidden_feature = x5
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        if not self.last_act is None:
            x = self.last_act(x)

        return x
    





if __name__ == '__main__':

    input = torch.rand([8,1,256,256])
    unet = UNet(input_channel=1,
                num_classes=1,
                feature_scale=1,
                last_layer_act=nn.Sigmoid())
    
    output = unet(input)

    print(output.shape)