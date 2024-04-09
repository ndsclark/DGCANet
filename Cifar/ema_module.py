import torch
import torch.nn as nn

__all__ = ['EMALayer']


class EMALayer(nn.Module):
    def __init__(self, in_channels, groups=32):
        super(EMALayer, self).__init__()

        self.groups = groups
        assert in_channels // groups > 0
        channels = in_channels // groups

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.gn = nn.GroupNorm(channels, channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # [b, c, h, w]
        b, c, h, w = x.size()
        # [b * g, c // g, h, w]
        group_x = x.view(b * self.groups, -1, h, w)

        # [b * g, c // g, h, 1]
        x_h = self.pool_h(group_x)
        # [b * g, c // g, w, 1]
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # [b * g, c // g, h + w, 1]
        y = torch.cat([x_h, x_w], dim=2)
        # [b * g, c // g, h + w, 1]
        y = self.conv1x1(y)

        # [b * g, c // g, h, 1], [b * g, c // g, w, 1]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        # [b * g, c // g, h, 1]
        x_h = x_h.sigmoid()
        # [b * g, c // g, 1, w]
        x_w = x_w.permute(0, 1, 3, 2).sigmoid()

        # [b * g, c // g, h, w]  相当于IN
        x1 = self.gn(group_x * x_h * x_w)
        # [b * g, 1, c // g]
        x1_1 = self.softmax(self.avg_pool(x1).view(b * self.groups, -1, 1).permute(0, 2, 1))
        # [b * g, c // g, h * w]
        x1_2 = x1.view(b * self.groups, c // self.groups, -1)

        # [b * g, c // g, h, w]
        x2 = self.conv3x3(group_x)
        # [b * g, 1, c // g]
        x2_2 = self.softmax(self.avg_pool(x2).view(b * self.groups, -1, 1).permute(0, 2, 1))
        # [b * g, c // g, h * w]
        x2_1 = x2.view(b * self.groups, c // self.groups, -1)

        # [b * g, 1, h * w]
        out1 = torch.matmul(x1_1, x2_1)
        # [b * g, 1, h * w]
        out2 = torch.matmul(x2_2, x1_2)

        # [b * g, 1, h, w]
        weights = (out1 + out2).view(b * self.groups, 1, h, w)
        # [b * g, c // g, h, w]
        out = group_x * weights.sigmoid()
        # [b, c, h, w]
        out = out.view(b, c, h, w)

        return out

