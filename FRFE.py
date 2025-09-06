"""
https://arxiv.org/abs/2303.03667
<<Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks>>
"""

from collections import OrderedDict
from functools import partial
from typing import List

from timm.models.layers import DropPath

import torch
import torch.nn as nn

from torch import Tensor

from utils import Interpolation_Coefficient
from wtconv import WTConv2d
from SConv import SConv_2d
from attention_blocks import CBAM
class PAConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 n_div: int = 4,
                 forward: str = 'split_cat'):
        super(PAConv2d, self).__init__()
        assert in_channels > 4, "in_channels should > 4, but got {} instead.".format(in_channels)
        self.dim_conv = in_channels // n_div
        self.dim_untouched = in_channels - self.dim_conv

        self.conv = nn.Conv2d(in_channels=self.dim_conv,
                              out_channels=self.dim_conv,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.CBAM = CBAM(in_channels)

        if forward == 'slicing':
            self.forward = self.forward_slicing

        elif forward == 'split_cat':
            self.forward = self.forward_split_cat

        else:
            raise NotImplementedError("forward method: {} is not implemented.".format(forward))

    def forward_slicing(self, x: Tensor) -> Tensor:
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        x = self.CBAM(x)


        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.CBAM(x)

        return x


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 act: str = 'GELU'):
        super(ConvBNLayer, self).__init__()
        assert act in ('ReLU', 'GELU')

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = getattr(nn, act)()

    def _fuse_bn_tensor(self) -> None:
        kernel = self.conv.weight
        bias = self.conv.bias if hasattr(self.conv, 'bias') and self.conv.bias is not None else 0
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        self.conv.weight.data = kernel * t
        self.conv.bias = nn.Parameter(beta - (running_mean - bias) * gamma / std, requires_grad=False)
        self.bn = nn.Identity()
        return self.conv.weight.data, self.conv.bias.data

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x
#通道变化，尺寸减半旋转卷积归一化激活层
class SCConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 act: str = 'GELU',
                 Coefficient_3=None):
        super(SCConvBNLayer, self).__init__()
        assert act in ('ReLU', 'GELU')

        self.Coefficient_3 = Coefficient_3
        self.SCin_out_2 = SConv_2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3,stride=2,padding=3,same=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = getattr(nn, act)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.SCin_out_2(x,self.Coefficient_3)
        x = self.bn(x)
        x = self.act(x)

        return x


class PAConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 inner_channels: int = None,
                 kernel_size: int = 3,
                 bias=False,
                 act: str = 'GELU',
                 n_div: int = 4,
                 forward: str = 'split_cat',
                 drop_path: float = 0.,
                 ):
        super(PAConvBlock, self).__init__()
        inner_channels = inner_channels or in_channels * 2
        self.conv1 = PAConv2d(in_channels,
                              kernel_size,
                              n_div,
                              forward)
        self.conv2 = ConvBNLayer(in_channels,
                                 inner_channels,
                                 bias=bias,
                                 act=act)
        self.conv3 = nn.Conv2d(inner_channels,
                               in_channels,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        return x + self.drop_path(y)



class Extractor(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1000,
                 last_channels=1280,
                 inner_channels: list = [40, 80, 160, 320],
                 blocks: list = [1, 2, 8, 2],
                 bias=False,
                 act='GELU',
                 n_div=4,
                 forward='split_cat',
                 drop_path=0.,
                 wt_levels=(5, 4, 3, 2),
                 Coefficient_3=None
                 ):
        super(Extractor, self).__init__()

        self.Coefficient_3=Coefficient_3

        self.WTConv0 = WTConv2d(in_channels, in_channels, kernel_size=5, wt_levels=wt_levels[0])

        self.WTConv1 = WTConv2d(inner_channels[0], inner_channels[0], kernel_size=5, wt_levels=wt_levels[1])

        self.WTConv2 = WTConv2d(inner_channels[1], inner_channels[1], kernel_size=5, wt_levels=wt_levels[2])

        self.WTConv3 = WTConv2d(inner_channels[2], inner_channels[2], kernel_size=5, wt_levels=wt_levels[3])

        self.embedding = SCConvBNLayer(in_channels,
                                       inner_channels[0],Coefficient_3=self.Coefficient_3)

        self.stage1 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             PAConvBlock(inner_channels[0],
                         bias=bias,
                         act=act,
                         n_div=n_div,
                         forward=forward,
                         drop_path=drop_path)) for idx in range(blocks[0])]))

        self.merging1 = SCConvBNLayer(inner_channels[0],
                                      inner_channels[1],Coefficient_3=self.Coefficient_3)

        self.stage2 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             PAConvBlock(inner_channels[1],
                         bias=bias,
                         act=act,
                         n_div=n_div,
                         forward=forward,
                         drop_path=drop_path)) for idx in range(blocks[1])]))

        self.merging2 = SCConvBNLayer(inner_channels[1],
                                      inner_channels[2],Coefficient_3=self.Coefficient_3)

        self.stage3 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             PAConvBlock(inner_channels[2],
                         bias=bias,
                         act=act,
                         n_div=n_div,
                         forward=forward,
                         drop_path=drop_path)) for idx in range(blocks[2])]))

        self.merging3 = SCConvBNLayer(inner_channels[2],
                                      inner_channels[3],Coefficient_3=self.Coefficient_3)

        self.stage4 = nn.Sequential(OrderedDict([
            ('block{}'.format(idx),
             PAConvBlock(inner_channels[3],
                         bias=bias,
                         act=act,
                         n_div=n_div,
                         forward=forward,
                         drop_path=drop_path)) for idx in range(blocks[3])]))

        self.classifier = nn.Sequential(OrderedDict([
            ('global_average_pooling', nn.AdaptiveAvgPool2d(1)),
            ('conv', nn.Conv2d(inner_channels[-1], last_channels, kernel_size=1, bias=False)),
            ('act', getattr(nn, act)()),
            ('flat', nn.Flatten()),
            ('fc', nn.Linear(last_channels, out_channels, bias=True))
        ]))
        self.feature_channels = inner_channels



    def fuse_bn_tensor(self):
        for m in self.modules():
            if isinstance(m, ConvBNLayer):
                m._fuse_bn_tensor()

    def forward(self, x: Tensor) -> List[Tensor]:

        x=self.WTConv0(x)


        x1 = self.stage1(self.embedding(x))


        x1 = self.WTConv1(x1)


        x2 = self.stage2(self.merging1(x1))


        x2 = self.WTConv2(x2)


        x3 = self.stage3(self.merging2(x2))


        x3 = self.WTConv3(x3)

        x4 = self.stage4(self.merging3(x3))
        return x1, x2, x3, x4




FRFE = partial(Extractor, inner_channels=[40, 80, 160, 160], blocks=[2, 2, 3, 3], act='GELU', drop_path=0.)
