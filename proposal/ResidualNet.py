import torch
from proposal.AttentionLayers import *
from proposal.GDN import GDN,GSDN
class Dense(torch.nn.Module):
    def __init__(self,ch):
        super(Dense, self).__init__()
        self.dense_ch = 32
        self.conv1 = torch.nn.Conv2d(ch, self.dense_ch, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(self.dense_ch+self.dense_ch, self.dense_ch, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(self.dense_ch + (self.dense_ch * 2), self.dense_ch, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(self.dense_ch + (self.dense_ch * 3), ch, 3, stride=1, padding=1)
        self.lrelu = torch.nn.LeakyReLU(inplace=True)
    def forward(self,input):
        x1 = self.lrelu(self.conv1(input))
        x2 = self.lrelu(self.conv2(torch.cat((input, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((input, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((input, x1, x2, x3), dim=1)))
        return x4

class DenseT(torch.nn.Module):
    def __init__(self,ch):
        super(DenseT, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(ch, ch, 3, stride=1, padding=1)
        self.lrelu1 = torch.nn.LeakyReLU(inplace=True)
        self.conv2 = torch.nn.ConvTranspose2d(ch+ch, ch, 3, stride=1, padding=1)
        self.lrelu2 = torch.nn.LeakyReLU(inplace=True)
        self.conv3 = torch.nn.ConvTranspose2d(ch+ch+ch, ch, 3, stride=1, padding=1)
        self.lrelu3 = torch.nn.LeakyReLU(inplace=True)
        self.conv4 = torch.nn.ConvTranspose2d(ch+ch+ch+ch, ch, 3, stride=1, padding=1)
        self.lrelu4 = torch.nn.LeakyReLU(inplace=True)
    def forward(self,input):
        x1 = self.lrelu1(self.conv1(input))
        x2 = self.lrelu2(self.conv2(torch.cat((input, x1), dim=1)))
        x3 = self.lrelu3(self.conv3(torch.cat((input, x1, x2), dim=1)))
        x4 = self.lrelu4(self.conv4(torch.cat((input, x1, x2, x3), dim=1)))
        return x4

class RES(torch.nn.Module):
    def __init__(self,ch):
        super(RES, self).__init__()
        self.conv1 = torch.nn.Conv2d(ch, ch, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(ch, ch, 3, stride=1, padding=1)
        self.lrelu1 = torch.nn.LeakyReLU(inplace=True)
        self.lrelu2 = torch.nn.LeakyReLU(inplace=True)
    def forward(self,input):
        short = input
        x = self.conv1(input)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = x + short
        return x 

class REST(torch.nn.Module):
    def __init__(self,ch):
        super(REST, self).__init__()
        self.convt1 = torch.nn.ConvTranspose2d(ch, ch, 3, stride=1, padding=1)
        self.convt2 = torch.nn.ConvTranspose2d(ch, ch, 3, stride=1, padding=1)
        self.lrelu1 = torch.nn.LeakyReLU(inplace=True)
        self.lrelu2 = torch.nn.LeakyReLU(inplace=True)
    def forward(self,input):
        short = input
        x = self.convt1(input)
        x = self.lrelu1(x)
        x = self.convt2(x)
        x = self.lrelu2(x)
        x = x + short
        return x 


class UpDownResNeXt(torch.nn.Module):
    def __init__(self,ch,kernel=3,inverse = False):
        super(UpDownResNeXt, self).__init__()
        self.half_ch = ch//2
        self.group = 32
        if inverse == False:
            self.First_conv = torch.nn.Conv2d(ch, self.half_ch, 1, stride=1)
            self.Middle_conv = torch.nn.Conv2d(self.half_ch, self.half_ch, kernel, stride=2, padding=kernel//2,groups = self.group)
            self.End_conv = torch.nn.Conv2d(self.half_ch, ch, 1, stride=1)
            self.GDN = GDN(ch)
        else:
            self.First_conv = torch.nn.ConvTranspose2d(ch, self.half_ch, 1, stride=1)
            self.Middle_conv = torch.nn.ConvTranspose2d(self.half_ch, self.half_ch, kernel, stride=2,output_padding=1, padding=kernel//2 ,groups = self.group)
            self.End_conv = torch.nn.ConvTranspose2d(self.half_ch, ch, 1, stride=1)
            self.GDN = GDN(ch,inverse = True)
        self.lrelu1 = torch.nn.LeakyReLU(inplace=True)
        self.lrelu2 = torch.nn.LeakyReLU(inplace=True)
        self.SE = resSELayer(ch)
        self.inverse = inverse
    def forward(self,input):
        x = self.lrelu1(self.First_conv(input))
        x = self.lrelu2(self.Middle_conv(x))
        x = self.GDN(self.End_conv(x))
        x = self.SE(x)
        return x 

class ResNeXt(torch.nn.Module):
    def __init__(self,ch,kernel=3,inverse = False):
        super(ResNeXt, self).__init__()
        self.half_ch = ch//2
        self.group = 32
        if inverse == False:
            self.First_conv = torch.nn.Conv2d(ch, self.half_ch, 1, stride=1)
            self.Middle_conv = torch.nn.Conv2d(self.half_ch, self.half_ch, kernel, stride=1, padding=kernel//2,groups = self.group)
            self.End_conv = torch.nn.Conv2d(self.half_ch, ch, 1, stride=1)
        else:
            self.First_conv = torch.nn.ConvTranspose2d(ch, self.half_ch, 1, stride=1)
            self.Middle_conv = torch.nn.ConvTranspose2d(self.half_ch, self.half_ch, kernel, stride=1, padding=kernel//2 ,groups = self.group)
            self.End_conv = torch.nn.ConvTranspose2d(self.half_ch, ch, 1, stride=1)
        self.lrelu1 = torch.nn.LeakyReLU(inplace=True)
        self.lrelu2 = torch.nn.LeakyReLU(inplace=True)
        self.lrelu3 = torch.nn.LeakyReLU(inplace=True)
        #self.CBAM = CBAM(ch)
        self.SE = resSELayer(ch)
        self.inverse = inverse
    def forward(self,input):
        short = input
        x = self.lrelu1(self.First_conv(input))
        x = self.lrelu2(self.Middle_conv(x))
        x = self.lrelu3(self.End_conv(x))
        x = self.SE(x)
        x += short
        return x 

#Concatenated Residual Modules 
class CRM(torch.nn.Module):
    def __init__(self,ch,inverse = False):
        super(CRM, self).__init__()
        if inverse == False:
            self.RES1 = ResNeXt(ch,kernel=3)
            self.RES2 = ResNeXt(ch,kernel=3)
            self.RES3 = ResNeXt(ch,kernel=3)
        else:
            self.RES1 = ResNeXt(ch,kernel=3,inverse=True)
            self.RES2 = ResNeXt(ch,kernel=3,inverse=True)
            self.RES3 = ResNeXt(ch,kernel=3,inverse=True)
        
    def forward(self,input):
        x = self.RES1(input)
        x = self.RES2(x)
        x = self.RES3(x)
        return x 

from torchsummary import summary 
if __name__ == "__main__":
    model = ResNeXt(256,inverse=True)
    model.cuda()
    summary(model,(256,128,128))  
