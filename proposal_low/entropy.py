import torch
import torch.nn as nn
import math

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class Entropy(nn.Module):
    def __init__(self,input_filters):
        super(Entropy, self).__init__()
        
        self.maskedconv = MaskedConv2d('A', input_filters, input_filters*2, 5, stride=1, padding=2)
        torch.nn.init.xavier_uniform_(self.maskedconv.weight.data, gain=1)
        torch.nn.init.constant_(self.maskedconv.bias.data, 0.0)
        self.conv1 = nn.Conv2d(input_filters*4,640, 1, stride=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(640, 640, 1, stride=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(640, input_filters*9, 1, stride=1)
        self.softmax = nn.Softmax(dim=-1)
        """
        self.conv1 = nn.Conv2d(input_filters*2,640, 1, stride=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(640, 640, 1, stride=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(640, input_filters*9, 1, stride=1)
        self.softmax = nn.Softmax(dim=-1)
        """

        

    def forward(self, sigma,y):
        
        y = self.maskedconv(y)
        x = torch.cat([y, sigma], dim=1)
       

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))   
        x = self.conv3(x)
        """
        x = self.relu1(self.conv1(sigma))
        x = self.relu2(self.conv2(x))   
        x = self.conv3(x)
        """
        # print("split_size: ", x.shape[1])
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = \
            torch.split(x, split_size_or_sections=int(x.shape[1]/9), dim=1)
        scale0 = torch.abs(scale0)
        scale1 = torch.abs(scale1)
        scale2 = torch.abs(scale2)
        probs = torch.stack([prob0, prob1, prob2], dim=-1)
        # print("probs shape: ", probs.shape)
        probs = self.softmax(probs)
        # probs = torch.nn.Softmax(dim=-1)(probs)
        means = torch.stack([mean0, mean1, mean2], dim=-1)
        variances = torch.stack([scale0, scale1, scale2], dim=-1)

        return means, variances, probs

class Entropy2(nn.Module):
    def __init__(self,input_filters):
        super(Entropy2, self).__init__()
        self.maskedconv = MaskedConv2d('A', input_filters, input_filters*2, 7, stride=1, padding=3)
        torch.nn.init.xavier_uniform_(self.maskedconv.weight.data, gain=1)
        torch.nn.init.constant_(self.maskedconv.bias.data, 0.0)
        self.conv1 = nn.Conv2d(input_filters*4,640, 1, stride=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(640, 640, 1, stride=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(640, input_filters*30, 1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sigma,y):
        y = self.maskedconv(y)
        x = torch.cat([y, sigma], dim=1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))   
        x = self.conv3(x)
        # print("split_size: ", x.shape[1])
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2, prob3, mean3, scale3, prob4, mean4, scale4, prob5, mean5, scale5, prob6, mean6, scale6, prob7, mean7, scale7, prob8, mean8, scale8, prob_m0, prob_m1, prob_m2 = \
            torch.split(x, split_size_or_sections=int(x.shape[1]/30), dim=1)
        scale0 = torch.abs(scale0)
        scale1 = torch.abs(scale1)
        scale2 = torch.abs(scale2)
        scale3 = torch.abs(scale3)
        scale4 = torch.abs(scale4)
        scale5 = torch.abs(scale5)
        scale6 = torch.abs(scale6)
        scale7 = torch.abs(scale7)
        scale8 = torch.abs(scale8)
        probs = torch.stack([prob0, prob1, prob2, prob3, prob4, prob5, prob6, prob7, prob8], dim=-1)
        # print("probs shape: ", probs.shape)
        probs = self.softmax(probs)
        # probs = torch.nn.Softmax(dim=-1)(probs)
        means = torch.stack([mean0, mean1, mean2, mean3, mean4, mean5 ,mean6, mean7, mean8], dim=-1)
        variances = torch.stack([scale0, scale1, scale2, scale3, scale4, scale5, scale6, scale7, scale8], dim=-1)
        probs_mix = torch.stack([prob_m0, prob_m1, prob_m2],dim=-1)
        probs_mix = self.softmax(probs_mix)
        return means, variances, probs, probs_mix

if __name__ == "__main__":
    z = torch.zeros([8,60,32,32])
    entropy = Entropy(60)
    means, variances, probs = entropy(z)
    print("means: ", means.shape)
    print("variances: ", variances.shape)
    print("probs: ", probs.shape)