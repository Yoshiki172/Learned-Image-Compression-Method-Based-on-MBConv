import torch

class ResidualDenseBlock(torch.nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(nf, gc, 3, padding=1, bias=bias)
        self.conv2 = torch.nn.Conv2d(nf + gc, gc, 3, padding=1, bias=bias)
        self.conv3 = torch.nn.Conv2d(nf + 2 * gc, gc, 3, padding=1, bias=bias)
        self.conv4 = torch.nn.Conv2d(nf + 3 * gc, gc, 3, padding=1, bias=bias)
        self.conv5 = torch.nn.Conv2d(nf + 4 * gc, nf, 3, padding=1, bias=bias)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x
