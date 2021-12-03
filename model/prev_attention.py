import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class PrevAttention(nn.Module):
    def __init__(self, indim, outdim):
        super(Self_Attention, self).__init__()
        self.attention_conv = ResBlock(indim, outdim)

    def forward(self, feature, mask):
        concat_fm = torch.cat((feature, mask),dim=1)
        fm = self.attention_conv(concat_fm)
        fm = torch.sigmoid(fm)
        return feature * fm
