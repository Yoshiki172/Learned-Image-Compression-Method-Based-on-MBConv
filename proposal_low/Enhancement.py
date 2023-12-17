import torch
import torch.nn as nn
from proposal_low.AttentionLayers import *

class EnhancementBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64,64,3,stride=1,padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(64,64,3,stride=1,padding=1)

    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        return x + inputs

class decoder_side_enhancement(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = torch.nn.Conv2d(3,64, 1, stride=1)
        self.enhancement_block1 = EnhancementBlock()
        self.enhancement_block2 = EnhancementBlock()
        self.enhancement_block3 = EnhancementBlock()
        self.relu = torch.nn.ReLU(inplace=True)
        self.last_conv = torch.nn.Conv2d(64,3, 1, stride=1)

    def forward(self, inputs):
        out = self.first_conv(inputs)
        x = self.enhancement_block1(out)
        x = self.enhancement_block2(x)
        x = self.enhancement_block3(x)
        x = out + x
        output = self.last_conv(x)
        return output + inputs
"""
class decoder_side_enhancement(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = torch.nn.Conv2d(3,64, 1, stride=1)
        self.conv = torch.nn.Conv2d(64,64, 3, stride=1,padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.last_conv = torch.nn.Conv2d(64,3, 1, stride=1)

    def enhancement_block(self,input):
        x = self.conv(input)
        x = self.relu(x)
        x = self.conv(x)
        return x + input

    def forward(self, inputs):
        out = self.first_conv(inputs)
        x = self.enhancement_block(out)
        x = self.enhancement_block(x)
        x = self.enhancement_block(x)
        x = out + x
        output = self.last_conv(x)
        return output + inputs
"""
class EnhBlock(nn.Module):
    def __init__(self, nf):
        super(EnhBlock, self).__init__()
        self.layers = nn.Sequential(
            DenseBlock(3, nf),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            DenseBlock(nf, 3)
        )

    def forward(self, x):
        return x + self.layers(x) * 0.2

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5
        
import torch.nn.init as init

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)