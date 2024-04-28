import torch
import torch.nn as nn
from torch import Size
from srec.utils.utils import InitDevice


class ConvBatchNorm(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            stride: Size,
    ) -> None:
        super(ConvBatchNorm, self).__init__()

        self.__conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False
        )

        self.__bn = nn.BatchNorm2d(
            out_channels
        )

    def forward(self, x):

        return self.__bn(self.__conv(x))
