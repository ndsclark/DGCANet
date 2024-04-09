import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = ['CRALayer']


class CRALayer(nn.Module):
    def __init__(self, channels, pooling_size=4):
        super(CRALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(pooling_size)
        self.conv = nn.Conv2d(channels, channels, kernel_size=pooling_size, groups=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv(out)
        out = self.sigmoid(out)

        return x * out.expand_as(x)

