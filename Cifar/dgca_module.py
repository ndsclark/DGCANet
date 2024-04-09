import torch
import torch.nn as nn


__all__ = ['DGCALayer']


class GSA_Module(nn.Module):
    """ Spatial attention module"""
    def __init__(self, inplanes, fusion_manner='add'):

        super(GSA_Module, self).__init__()

        assert fusion_manner in ['add', 'mul']
        self.fusion_manner = fusion_manner

        self.inplanes = inplanes

        self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        self.transform = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, groups=self.inplanes),
            nn.LayerNorm([self.inplanes, 1, 1]))

        self.alpha = nn.Parameter(torch.zeros(1))

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()

        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):

        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        # [N, C, 1, 1]
        attention_term = self.transform(context)

        if self.fusion_manner == 'add':
            out = x + self.alpha * attention_term
        elif self.fusion_manner == 'mul':
            mul_term = torch.sigmoid(attention_term)
            out = x * mul_term
        else:
            assert False, "invalid fusion_manner"

        return out


class GCA_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim, fusion_manner='add'):

        super(GCA_Module, self).__init__()

        assert fusion_manner in ['add', 'mul']
        self.fusion_manner = fusion_manner

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=-1)

        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        batch, C, height, width = x.size()
        # [N, C, 1, 1]
        context = self.avg_pool(x)
        # [N, 1, C]
        context = context.view(batch, C, -1).permute(0, 2, 1)
        # [N, 1, C]
        attention = self.softmax(context)
        # [N, C, H * W]
        proj_value = x.view(batch, C, -1)
        # [N, 1, H * W]
        attention_term = torch.bmm(attention, proj_value)
        # [N, 1, H, W]
        attention_term = attention_term.view(batch, 1, height, width)

        if self.fusion_manner == 'add':
            out = x + self.beta * attention_term
        elif self.fusion_manner == 'mul':
            mul_term = torch.sigmoid(attention_term)
            out = x * mul_term
        else:
            assert False, "invalid fusion_manner"

        return out


class DGCALayer(nn.Module):
    def __init__(self, inp, spatial=True, channel=True, fusion_types='add'):

        super(DGCALayer, self).__init__()

        assert fusion_types in ['add', 'mul', 'max']
        self.fusion_types = fusion_types

        if spatial:
            self.spatial_attention = GSA_Module(inp)
        else:
            self.spatial_attention = None

        if channel:
            self.channel_attention = GCA_Module(inp)
        else:
            self.channel_attention = None

    def forward(self, x):

        if self.spatial_attention is not None:
            s_out = self.spatial_attention(x)

        if self.channel_attention is not None:
            c_out = self.channel_attention(x)

        if (self.spatial_attention is not None) and (self.channel_attention is not None):
            if self.fusion_types == 'add':
                out = s_out + c_out
            elif self.fusion_types == 'mul':
                out = s_out * c_out
            elif self.fusion_types == 'max':
                out = torch.max(s_out, c_out)
            else:
                assert False, "invalid fusion_types"
        elif self.spatial_attention is not None:
            out = s_out
        elif self.channel_attention is not None:
            out = c_out
        else:
            assert False, "invalid DGALayer"

        return out
