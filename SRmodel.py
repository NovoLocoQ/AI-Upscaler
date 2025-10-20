import torch
from torch import nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_chan,out_chan):
        super().__init__()
        self.Layer1=nn.Sequential(
            nn.Conv2d(in_channels=in_chan,out_channels=out_chan,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_chan,out_channels=out_chan,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.Layer1(x)

class DownSample(nn.Module):
    def __init__(self,in_chan,out_chan):
        super().__init__()
        self.layer1=nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_chan=in_chan,out_chan=out_chan)
        )
        
    def forward(self,x):
        x1=self.layer1(x)
        return x1
    
class UpScale(nn.Module):
    def __init__(self, in_dec_chan, in_enc_chan, out_chan):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dec_chan, in_dec_chan // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_dec_chan // 2 + in_enc_chan, out_chan)
        
    def forward(self, x_dec, x_enc):
        x_dec = self.up(x_dec)
        diffY = x_enc.size()[2] - x_dec.size()[2]
        diffX = x_enc.size()[3] - x_dec.size()[3]
        x_dec = F.pad(x_dec, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_dec, x_enc], dim=1)
        return self.conv(x)
class UNETPixelShuffle(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,b_features=64,scale=4):
        super().__init__()
        self.scale=scale
        #Encoder
        self.start=DoubleConv(in_chan=in_channels,out_chan=b_features)
        self.d1=DownSample(in_chan=b_features,out_chan=b_features*2)
        self.d2=DownSample(in_chan=b_features*2,out_chan=b_features*4)
        self.d3=DownSample(in_chan=b_features*4,out_chan=b_features*8)
        self.d4=DownSample(in_chan=b_features*8,out_chan=b_features*16)
        #Bottleneck
        self.bottleneck=DoubleConv(b_features*16,b_features*16)
        #Decoder
        self.u4 = UpScale(in_dec_chan=b_features*16, in_enc_chan=b_features*8, out_chan=b_features*8)
        self.u3 = UpScale(in_dec_chan=b_features*8, in_enc_chan=b_features*4, out_chan=b_features*4)
        self.u2 = UpScale(in_dec_chan=b_features*4, in_enc_chan=b_features*2, out_chan=b_features*2)
        self.u1 = UpScale(in_dec_chan=b_features*2, in_enc_chan=b_features, out_chan=b_features)
        #Final Upscaling to required resolution
        self.finalConv=DoubleConv(in_chan=b_features,out_chan=out_channels*(scale**2))
        self.finalScale=nn.PixelShuffle(scale)
    def forward(self,x):
        x1=self.start(x)
        x2=self.d1(x1)
        x3=self.d2(x2)
        x4=self.d3(x3)
        x5=self.d4(x4)

        bn=self.bottleneck(x5)

        y1=self.u4(bn,x4)
        y2=self.u3(y1,x3)
        y3=self.u2(y2,x2)
        y4=self.u1(y3,x1)
        
        out=self.finalConv(y4)
        out=self.finalScale(out)
        return out



        