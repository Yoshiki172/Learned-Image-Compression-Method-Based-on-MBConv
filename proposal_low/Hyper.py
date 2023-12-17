import torch
import torch.nn as nn
import math
from proposal_low.GDN import GDN
from proposal_low.ResidualNet import RES
from proposal_low.AttentionLayers import CBAM,resSELayer,EEM
from proposal_low.Attention import *

class Analysis_Hyper(torch.nn.Module):
    def __init__(self,num_filters=192,filters=320):
        super(Analysis_Hyper,self).__init__()
        self.h_a = nn.Sequential(
            nn.Conv2d(filters, num_filters, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1),
        )
        #self.CBAM = CBAM(num_filters)
        #self.SE = resSELayer(num_filters)
        self.Attention = Attention(num_filters)

    def forward(self, inputs):
        x = self.h_a(inputs)
        #x = self.CBAM(x)
        #x = self.Attention(x)
        return x 
    
class Synthesis_Hyper(torch.nn.Module):
    def __init__(self,num_filters=192,filters=320):
        super(Synthesis_Hyper,self).__init__()
        self.Attention = Attention(num_filters)
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, padding=1,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, int(num_filters*1.5), 3, stride=2, padding=1,output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(int(num_filters*1.5), int(num_filters*1.5), 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(int(num_filters*1.5), int(filters*2), 3, stride=2, padding=1, output_padding=1)
        )
    def forward(self, inputs):
        #x = self.Attention(inputs)
        x = self.h_s(inputs)
        return x
    