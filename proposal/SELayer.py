import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
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


if __name__ == "__main__":
    z = torch.zeros([8,192,4,4])
    entropy = SELayer(192)
    x = entropy(z)
    print(x.shape)