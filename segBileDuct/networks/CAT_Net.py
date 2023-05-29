import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class CrossSliceAttention(nn.Module):
    def __init__(self,input_channels):
        super(CrossSliceAttention,self).__init__()
        self.linear_q=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1),bias=False)
        self.linear_k=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1),bias=False)
        self.linear_v=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1),bias=False)

    def forward(self,pooled_features,features):
        q=self.linear_q(pooled_features)
        q=q.view(q.size(0),-1)
        k=self.linear_k(pooled_features)
        k=k.view(k.size(0),-1)
        v=self.linear_v(features)
        x=torch.matmul(q,k.permute(1,0))/np.sqrt(q.size(1))
        x=torch.softmax(x,dim=1)
        out=torch.zeros_like(v)
        for i in range(x.size(0)):
            temp=x[i,:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out[i,:,:,:]=torch.sum(temp*v,dim=0).clone()
        return out


class MultiHeadedCrossSliceAttentionModule(nn.Module):
    def __init__(self,input_channels,heads=3,pool_kernel_size=(4,4),input_size=(1128,128),batch_size=20,pool_method='avgpool'):
        super(MultiHeadedCrossSliceAttentionModule,self).__init__()
        self.attentions=[]
        self.linear1=nn.Conv2d(in_channels=heads*input_channels,out_channels=input_channels,kernel_size=(1,1))
        self.norm1=nn.LayerNorm([batch_size,input_channels,input_size[0],input_size[1]])
        self.linear2=nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=(1,1))
        self.norm2=nn.LayerNorm([batch_size,input_channels,input_size[0],input_size[1]])

        if pool_method=="maxpool":
            self.pool=nn.MaxPool2d(kernel_size=pool_kernel_size)
        elif pool_method=="avgpool":
            self.pool=nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            assert (False)  # not implemented yet

        for i in range(heads):
            self.attentions.append(CrossSliceAttention(input_channels))
        self.attentions=nn.Sequential(*self.attentions)

    def forward(self,pooled_features,features):

        for i in range(len(self.attentions)):
            x_=self.attentions[i](pooled_features,features)
            if i==0:
                x=x_
            else:
                x=torch.cat((x,x_),dim=1)
        out=self.linear1(x)
        x=F.gelu(out)+features
        out_=self.norm1(x)
        out=self.linear2(out_)
        x=F.gelu(out)+out_
        out=self.norm2(x)
        pooled_out=self.pool(out)
        return pooled_out,out


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,is_pe_learnable=True,max_len=20):
        super(PositionalEncoding,self).__init__()

        position=torch.arange(max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe=torch.zeros(max_len,d_model,1,1)
        pe[:,0::2,0,0]=torch.sin(position*div_term)
        pe[:,1::2,0,0]=torch.cos(position*div_term)
        self.pe=nn.Parameter(pe.clone(),is_pe_learnable)
        #self.register_buffer('pe',self.pe)

    def forward(self,x):
        return x+self.pe[:x.size(0),:,:,:]

    def get_pe(self):
        return self.pe[:,:,0,0]


class ConvBlock(nn.Module):
    def __init__(self,input_channels,output_channels,max_pool,return_single=False):
        super(ConvBlock,self).__init__()
        self.max_pool=max_pool
        self.conv=[]
        self.conv.append(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.InstanceNorm2d(output_channels))
        self.conv.append(nn.LeakyReLU())
        self.conv.append(nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1))
        self.conv.append(nn.InstanceNorm2d(output_channels))
        self.conv.append(nn.LeakyReLU())
        self.return_single=return_single
        if max_pool:
            self.pool=nn.MaxPool2d(2,stride=2,dilation=(1,1))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x):
        x=self.conv(x)
        b=x
        if self.max_pool:
            x=self.pool(x)
        if self.return_single:
            return x
        else:
            return x,b


class DeconvBlock(nn.Module):
    def __init__(self,input_channels,output_channels,intermediate_channels=-1):
        super(DeconvBlock,self).__init__()
        input_channels=int(input_channels)
        output_channels=int(output_channels)
        if intermediate_channels<0:
            intermediate_channels=output_channels*2
        else:
            intermediate_channels=input_channels
        self.upconv=[]
        self.upconv.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.upconv.append(nn.Conv2d(in_channels=input_channels,out_channels=intermediate_channels//2,kernel_size=3,stride=1,padding=1))
        self.conv=ConvBlock(intermediate_channels,output_channels,False)
        self.upconv=nn.Sequential(*self.upconv)

    def forward(self,x,b):
        x=self.upconv(x)
        x=torch.cat((x,b),dim=1)
        x,_=self.conv(x)
        return x

class UNetDecoder(nn.Module):
    def __init__(self,num_layers,base_num):
        super(UNetDecoder,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        for i in range(num_layers-1,0,-1):
            self.conv.append(DeconvBlock(base_num*(2**i),base_num*(2**(i-1))))
        self.conv=nn.Sequential(*self.conv)

    def forward(self,x,b):
        for i in range(self.num_layers-1):
            x=self.conv[i](x,b[i])
        return x

class CrossSliceUNetEncoder(nn.Module):
    def __init__(self,input_channels,num_layers,base_num,num_attention_blocks=3,heads=4,pool_kernel_size=(4,4),input_size=(128,128),batch_size=20,pool_method='avgpool',is_pe_learnable=True):
        super(CrossSliceUNetEncoder,self).__init__()
        self.conv=[]
        self.num_layers=num_layers
        self.num_attention_blocks=num_attention_blocks
        for i in range(num_layers):
            if i==0:
                self.conv.append(ConvBlock(input_channels,base_num,True))
            else:
                self.conv.append(ConvBlock(base_num*(2**(i-1)),base_num*(2**i),(i!=num_layers-1)))
        self.conv=nn.Sequential(*self.conv)
        self.pools=[]
        self.pes=[]
        self.attentions=[]
        for i in range(num_layers):
            if pool_method=='maxpool':
                self.pools.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
            elif pool_method=='avgpool':
                self.pools.append(nn.AvgPool2d(kernel_size=pool_kernel_size))
            else:
                assert (False)  # not implemented yet

            self.pes.append(PositionalEncoding(base_num*(2**i),is_pe_learnable))
            temp=[]
            for j in range(num_attention_blocks):
                temp.append(MultiHeadedCrossSliceAttentionModule(base_num*(2**i),heads,pool_kernel_size,input_size,batch_size,pool_method))
            input_size=(input_size[0]//2,input_size[1]//2)
            self.attentions.append(nn.Sequential(*temp))
        self.attentions=nn.Sequential(*self.attentions)
        self.pes=nn.Sequential(*self.pes)

    def forward(self,x):
        b=[]
        for i in range(self.num_layers):
            x,block=self.conv[i](x)
            if i!=self.num_layers-1:
                block=self.pes[i](block)
                block_pool=self.pools[i](block)
                for j in range(self.num_attention_blocks):
                    block_pool,block=self.attentions[i][j](block_pool,block)
            else:
                x=self.pes[i](x)
                x_pool=self.pools[i](x)
                for j in range(self.num_attention_blocks):
                    x_pool,x=self.attentions[i][j](x_pool,x)
            b.append(block)
        b=b[:-1]
        b=b[::-1]
        return x,b



class CrossSliceAttentionUNet(nn.Module):
    def __init__(self,input_channels,num_classes,num_layers,heads=3,num_attention_blocks=2,base_num=64,pool_kernel_size=(4,4),input_size=(128,128),batch_size=20,pool_method="avgpool",is_pe_learnable=True):
        super(CrossSliceAttentionUNet,self).__init__()
        self.encoder=CrossSliceUNetEncoder(input_channels,num_layers,base_num,num_attention_blocks,heads,pool_kernel_size,input_size,batch_size,pool_method,is_pe_learnable)
        self.decoder=UNetDecoder(num_layers,base_num)
        self.base_num=base_num
        self.input_channels=input_channels
        self.num_classes=num_classes
        self.conv_final=nn.Conv2d(in_channels=base_num,out_channels=num_classes,kernel_size=(1,1))

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x,b=self.encoder(x)
        x=self.decoder(x,b)
        x=self.conv_final(x)
        return self.sigmoid(x)



class CrossSliceUNetPlusPlus(nn.Module):
    def __init__(self,input_channels,num_classes,num_layers,heads=3,num_attention_blocks=2,base_num=64,pool_kernel_size=(4,4),input_size=(128,128),batch_size=20,pool_method="maxpool",is_pe_learnable=True):
        super().__init__()
        self.num_layers=num_layers
        self.num_attention_blocks=num_attention_blocks
        nb_filter=[]
        for i in range(num_layers):
            nb_filter.append(base_num*(2**i))
        self.pool=nn.MaxPool2d(2,2)
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv=[]
        for i in range(num_layers):
            temp_conv=[]
            for j in range(num_layers-i):
                if j==0:
                    if i==0:
                        inp=input_channels
                    else:
                        inp=nb_filter[i-1]
                else:
                    inp=nb_filter[i]*j+nb_filter[i+1]
                temp_conv.append(ConvBlock(inp,nb_filter[i],False,True))
            self.conv.append(nn.Sequential(*temp_conv))
        self.conv=nn.Sequential(*self.conv)
        self.pools=[]
        self.pes=[]
        self.attentions=[]
        for i in range(num_layers):
            if pool_method=='maxpool':
                self.pools.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
            elif pool_method=='avgpool':
                self.pools.append(nn.AvgPool2d(kernel_size=pool_kernel_size))
            else:
                assert (False)  # not implemented yet

            self.pes.append(PositionalEncoding(base_num*(2**i),is_pe_learnable))
            temp=[]
            for j in range(num_attention_blocks):
                temp.append(MultiHeadedCrossSliceAttentionModule(base_num*(2**i),heads,pool_kernel_size,input_size,batch_size,pool_method))
            input_size=(input_size[0]//2,input_size[1]//2)
            self.attentions.append(nn.Sequential(*temp))
        self.attentions=nn.Sequential(*self.attentions)
        self.pes=nn.Sequential(*self.pes)
        self.final=[]
        for i in range(num_layers-1):
            self.final.append(nn.Conv2d(nb_filter[0],num_classes,kernel_size=(1,1)))
        self.final=nn.Sequential(*self.final)

    def forward(self,inputs):
        x=[]
        for i in range(self.num_layers):
            temp=[]
            for j in range(self.num_layers-i):
                temp.append([])
            x.append(temp)
        x[0][0].append(self.conv[0][0](inputs))
        for s in range(1,self.num_layers):
            for i in range(s+1):
                if i==0:
                    x[s-i][i].append(self.conv[s-i][i](self.pool(x[s-i-1][i][0])))
                else:
                    for j in range(i):
                        if j==0:
                            block=x[s-i][j][0]
                            block_pool=self.pools[s-i](block)
                            for k in range(self.num_attention_blocks):
                                block_pool,block=self.attentions[s-i][k](block_pool,block)
                            temp_x=block
                            #print(s-i,j)
                        else:
                            temp_x=torch.cat((temp_x,x[s-i][j][0]),dim=1)
                            #print(s-i,j)
                    temp_x=torch.cat((temp_x,self.up(x[s-i+1][i-1][0])),dim=1)
                    #print('up',s-i+1,i-1,temp_x.size(),self.up(x[s-i+1][i-1][0]).size())
                    x[s-i][i].append(self.conv[s-i][i](temp_x))
        if self.training:
            res=[]
            for i in range(self.num_layers-1):
                res.append(self.final[i](x[0][i+1][0]))
            return res
        else:
            return self.final[-1](x[0][-1][0])
        

def get_cat_net(img_ch=3, batch_size=8):

    model = CrossSliceAttentionUNet(input_channels=img_ch, num_classes=1, input_size=(256,256), num_layers=4, base_num=16, batch_size=batch_size)

    return model


from torchstat import stat
from networks.utils_layers import measure_inference_speed
if __name__ == '__main__':

    batch_size = 1
    x = torch.rand([1, 3, 256, 256])

    model = CrossSliceAttentionUNet(input_channels=3, num_classes=1, input_size=(256,256), num_layers=4, base_num=16, batch_size=batch_size)

    total_params = sum([param.nelement() for param in model.parameters()])
    
    # stat(model, (3,256,256))
    model = model.cuda()
    model.eval()
    print("Number of parameter: %.2fM" % (total_params/1e6))    

    print('fps: ', measure_inference_speed(model, (x.cuda(),)))
