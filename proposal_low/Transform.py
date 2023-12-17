import torch
import torch.nn as nn
import math
from proposal_low.GDN import GDN,GSDN
from proposal_low.ResidualNet import *
from proposal_low.Attention import *
from proposal_low.SELayer import SELayer
from proposal_low.AttentionLayers import CBAM,resSELayer
from proposal_low.Enhancement import *

class Analysis_transform(torch.nn.Module):
    def __init__(self,num_filters=256,out_filters=320):
        super(Analysis_transform,self).__init__()

        self.convA_skip = nn.Conv2d(3, 256, 3, stride=2, padding=1)
        self.DownA = nn.Sequential(
            nn.Conv2d(3, num_filters, 1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 5,groups=num_filters, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            resSELayer(num_filters),
            nn.Conv2d(num_filters, num_filters, 1, stride=1),
            GDN(num_filters)
        )
        
        self.res1 = RES(num_filters)

        self.convB_skip = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)
        self.DownB = nn.Sequential(
            nn.Conv2d(num_filters, num_filters,1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 5,groups=num_filters, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            resSELayer(num_filters),
            nn.Conv2d(num_filters, num_filters, 1, stride=1),
            GDN(num_filters)
        )

        self.res2 = RES(num_filters)

        self.Attention1 = Attention(num_filters)
      
        self.convC_skip = nn.Conv2d(num_filters, out_filters, 3, stride=2, padding=1)
        self.DownC = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 5,groups=num_filters, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            resSELayer(num_filters),
            nn.Conv2d(num_filters, out_filters, 1, stride=1),
            GDN(out_filters)
        )
        

        self.Attention2 = Attention(out_filters)
        
    def forward(self, inputs):
        shortA = self.convA_skip(inputs)
        x = self.DownA(inputs)
        x = x + shortA
      
        x = self.res1(x)

        shortB = self.convB_skip(x)
        x = self.DownB(x)
        x = x + shortB
    
        x = self.res2(x)

        x = self.Attention1(x)
   
        shortC = self.convC_skip(x)
        x = self.DownC(x)
        x = x + shortC

        x =  self.Attention2(x)

        return x
    

class Synthesis_transform(torch.nn.Module):
    def __init__(self,num_filters=256,out_filters=320):
        super(Synthesis_transform, self).__init__()
        self.Attention1 = Attention(out_filters)
        self.res1 = REST(out_filters)

        self.SkipA = nn.Sequential(
            nn.ConvTranspose2d(out_filters,num_filters,3,stride=2,padding=1,output_padding=1)
        )
        self.UpLayerA = nn.Sequential(
            nn.ConvTranspose2d(out_filters,num_filters,1,stride=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(num_filters,num_filters,5,groups=num_filters,stride=2,padding=2,output_padding=1),
            nn.LeakyReLU(inplace=True),
            resSELayer(num_filters),
            nn.ConvTranspose2d(num_filters,num_filters,1,stride=1),
            GDN(num_filters,inverse=True)
        )
        
       
        self.res2 = REST(num_filters)
        self.Attention2 = Attention(num_filters)
        
        self.SkipB = nn.Sequential(
            nn.ConvTranspose2d(num_filters,num_filters,3,stride=2,padding=1,output_padding=1)
        )
        self.UpLayerB = nn.Sequential(
            nn.ConvTranspose2d(num_filters,num_filters,1,stride=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(num_filters,num_filters,5,groups=num_filters,stride=2,padding=2,output_padding=1),
            nn.LeakyReLU(inplace=True),
            resSELayer(num_filters),
            nn.ConvTranspose2d(num_filters,num_filters,1,stride=1),
            GDN(num_filters,inverse=True)
        )
        
        self.res3 = REST(num_filters)

        self.SkipC = nn.Sequential(
            nn.ConvTranspose2d(num_filters,3,3,stride=2,padding=1,output_padding=1)
        )
        self.UpLayerC = nn.Sequential(
            nn.ConvTranspose2d(num_filters,num_filters,1,stride=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(num_filters,num_filters,5,groups=num_filters,stride=2,padding=2,output_padding=1),
            nn.LeakyReLU(inplace=True),
            resSELayer(num_filters),
            nn.ConvTranspose2d(num_filters,3,1,stride=1),
            GDN(3,inverse=True)
        )
        
        self.enhance = decoder_side_enhancement()
        
    
    def forward(self, inputs):
        x = self.Attention1(inputs)
        x = self.res1(x)

        shortA = self.SkipA(x)
        x = self.UpLayerA(x)
        x = x + shortA
  
        x = self.res2(x)
        x = self.Attention2(x)

        shortB = self.SkipB(x)
        x = self.UpLayerB(x)
        x = x + shortB
     
        x = self.res3(x)

        shortC = self.SkipC(x)
        x = self.UpLayerC(x)
        x = x + shortC
        
        x = self.enhance(x)
        return x
    