import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        b, c, _, _ = input.size()
        y = self.avg_pool(input).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = input * y.expand_as(input)
        return y

class resSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(resSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        b, c, _, _ = input.size()
        y = self.avg_pool(input).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = input * y.expand_as(input)
        y = y + input
        return y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Simple_CBAM(nn.Module):
    def __init__(self, filter):
        super(Simple_CBAM, self).__init__()
        self.ca = ChannelAttention(filter)
        self.sa = SpatialAttention()

    def forward(self, input):
        x = self.ca(input) * input
        x = self.sa(x) * x
        return x

class CBAM(nn.Module):
    def __init__(self, filter):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(filter)
        self.sa = SpatialAttention()

    def forward(self, input):
        short = input
        x = self.ca(input) * input
        x = self.sa(x) * x
        x = x + short
        return x

import torch.nn.functional as F


class EEM(nn.Module):
    def __init__(self,num_filters):
        super(EEM, self).__init__()
        self.laplace_kernel = torch.FloatTensor([[1, 1, 1], 
                                                [1, -8, 1], 
                                                [1, 1, 1]])
        self.kernel1 = torch.FloatTensor([[1, 0, -1], 
                                        [2, 0, -2], 
                                        [1, 0, -1]])
        self.kernel2 = torch.FloatTensor([[1, 2, 1], 
                                        [0, 0, 0], 
                                        [-1, -2, -1]])
        self.laplace_filter = self.laplace_kernel.expand(num_filters, num_filters, 3, 3).cuda()
        self.sobel_filter1 = self.kernel1.expand(num_filters, num_filters, 3, 3).cuda()
        self.sobel_filter2 = self.kernel2.expand(num_filters, num_filters, 3, 3).cuda()
        self.conv = torch.nn.Conv2d(num_filters,num_filters,3,stride=1,padding=2).cuda()
        torch.nn.init.xavier_uniform_(self.conv.weight.data, gain=1)
        torch.nn.init.constant_(self.conv.bias.data, 0.0)
    def forward(self, input):
        lap_img = F.conv2d(input,self.laplace_filter)
        
        result = F.leaky_relu(self.conv(lap_img))
        return result

if __name__ == "__main__":
    z = torch.zeros([8,192,4,4])
    entropy = SELayer(192)
    x = entropy(z)
    print(x.shape)