import torch
import torch.nn as nn
import math
from proposal_low.GDN import GDN
from proposal_low.ResidualNet import *
from proposal_low.Attention import *
from proposal_low.SELayer import SELayer

class LOWEnc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convA_skip = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.DownA = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1)
        )
        
        self.gdn1 = GDN(256)
        self.res1 = RES(256)

        self.convB_skip = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.DownB = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1)
        )

        self.gdn2 = GDN(256)
        self.res2 = RES(256)

        self.Attention1 = Attention(256)
        #self.SE1 = SELayer(256)

        self.convC_skip = nn.Conv2d(256, 568, 3, stride=2, padding=1)
        self.DownC = nn.Sequential(
            nn.Conv2d(256, 568, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(568, 568, 3, stride=1, padding=1)
        )
        self.gdn3 = GDN(568)

        self.Attention2 = Attention(568)
        self.SE2 = SELayer(568)

    def forward(self, inputs):
        shortA = self.convA_skip(inputs)
        x = self.DownA(inputs)
        x = x + shortA
        x = self.gdn1(x)
        x = self.res1(x)

        shortB = self.convB_skip(x)
        x = self.DownB(x)
        x = x + shortB
        x = self.gdn2(x)
        x = self.res2(x)

        x = self.Attention1(x)
        #x = self.SE1(x)
        shortC = self.convC_skip(x)
        x = self.DownC(x)
        x = x + shortC
        x = self.gdn3(x)

        x =  self.Attention2(x)
        x = self.SE2(x)
        return x

class LOWDec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Attention1 = Attention(568)
        self.res1 = REST(568)

        self.SkipA = nn.Sequential(
            nn.ConvTranspose2d(568,256,3,stride=2,padding=1,output_padding=1)
        )
        self.UpLayerA = nn.Sequential(
            nn.ConvTranspose2d(568,256,3,stride=2,padding=1,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256,256,3,stride=1,padding=1),
        )
        
        self.igdn1 = GDN(256,inverse=True)
        self.res2 = REST(256)
        self.Attention2 = Attention(256)
        
        self.SkipB = nn.Sequential(
            nn.ConvTranspose2d(256,256,3,stride=2,padding=1,output_padding=1)
        )
        self.UpLayerB = nn.Sequential(
            nn.ConvTranspose2d(256,256,3,stride=2,padding=1,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256,256,3,stride=1,padding=1),
        )
        self.igdn2 = GDN(256,inverse=True)
        self.res3 = REST(256)

        self.SkipC = nn.Sequential(
            nn.ConvTranspose2d(256,256,3,stride=2,padding=1,output_padding=1)
        )
        self.UpLayerC = nn.Sequential(
            nn.ConvTranspose2d(256,256,3,stride=2,padding=1,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256,256,3,stride=1,padding=1),
        )

    
    def forward(self, inputs):
        x = self.Attention1(inputs)
        x = self.res1(x)

        shortA = self.SkipA(x)
        x = self.UpLayerA(x)
        x = x + shortA
        x = self.igdn1(x)
        x = self.res2(x)
        x = self.Attention2(x)

        shortB = self.SkipB(x)
        x = self.UpLayerB(x)
        x = x + shortB
        x = self.igdn2(x)
        x = self.res3(x)

        shortC = self.SkipC(x)
        x = self.UpLayerC(x)
        x = x + shortC
        
        return x