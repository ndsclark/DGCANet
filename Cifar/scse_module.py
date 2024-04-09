import torch
import torch.nn as nn

__all__ = ['SCSELayer']


class ChannelSELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelSELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return x * out.expand_as(x)


class SpatialSELayer(nn.Module):
    def __init__(self, in_channels):
        super(SpatialSELayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.conv(x)
        out = self.sigmoid(out)

        return x * out.expand_as(x)


class SCSELayer(nn.Module):
    def __init__(self, num_channels, spatial=True, channel=True, fusion_types='max'):
        super(SCSELayer, self).__init__()

        assert fusion_types in ['add', 'mul', 'max']
        self.fusion_types = fusion_types

        if spatial:
            self.sSE = SpatialSELayer(num_channels)
        else:
            self.sSE = None

        if channel:
            self.cSE = ChannelSELayer(num_channels)
        else:
            self.cSE = None

    def forward(self, x):

        if self.sSE is not None:
            s_out = self.sSE(x)

        if self.cSE is not None:
            c_out = self.cSE(x)

        if (self.sSE is not None) and (self.cSE is not None):
            if self.fusion_types == 'add':
                out = s_out + c_out
            elif self.fusion_types == 'mul':
                out = torch.mul(s_out, c_out)
            elif self.fusion_types == 'max':
                out = torch.max(s_out, c_out)
            else:
                assert False, "invalid fusion_types"
        elif self.sSE is not None:
            out = s_out
        elif self.cSE is not None:
            out = c_out
        else:
            assert False, "invalid SCSELayer"

        return out
