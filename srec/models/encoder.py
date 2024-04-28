import torch
import torch.nn as nn
from torch import Tensor
from srec.arch import EDDModule
from srec.modules.conv_bn import ConvBatchNorm
from typing import List


class HREncoder(EDDModule):

    @EDDModule.INIT_DEVICE.move
    def __init__(self, configs) -> None:
        super().__init__(configs)

        self.num_layers = configs.num_conv_bn

        channels = [
            configs.in_channels,
            configs.out_channels // 2,
            configs.out_channels // 2,
            configs.out_channels // 2
        ]

        down_scale = 1
        for s in configs.strides:
            down_scale *= s

        self._adapter = nn.Conv2d(
            configs.in_channels,
            configs.out_channels // 2,
            kernel_size=1,
            stride=down_scale
        )

        self._cbs = nn.ModuleList([
            ConvBatchNorm(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=configs.kernel_sizes[i],
                padding=configs.paddings[i],
                stride=configs.strides[i]
            ) for i in range(self.num_layers)
        ])

        self._relus = nn.ModuleList([
            nn.ReLU() for i in range(self.num_layers)
        ])

        self._out_conv = nn.Conv2d(
            configs.out_channels,
            configs.out_channels,
            kernel_size=configs.kernel_sizes[-1],
            padding=configs.paddings[-1],
            stride=1
        )

    @torch.inference_mode()
    def infer(self, x):
        return self.forward(x)

    @torch.inference_mode()
    def evaluate(self, x):
        return self.forward(x)

    def forward(
        self,
        x
    ):
        conv_x = x.clone()
        x = self._adapter(x)
        for lid in range(self.num_layers):
            conv_x = self._cbs[lid](conv_x)
            conv_x = self._relus[lid](conv_x)
        # TODO: add
        x = torch.concat([x, conv_x], dim=1)
        x = self._out_conv(x)
        return x
