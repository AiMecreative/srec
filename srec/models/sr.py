import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from srec.arch import EDDModule
from srec.modules.srb import SequentialResidualBlock
from srec.modules.conv_bn import ConvBatchNorm
from srec.utils.utils import InitDevice, merge
from loss.losses import GradientProfileLoss


class SR(EDDModule):
    """
    param len`channels` == `num_srb`, the last channel repeats twice
    """

    @EDDModule.INIT_DEVICE.move
    def __init__(
        self,
        configs
    ) -> None:
        super().__init__()

        channels = [configs.channels] * (configs.num_srbs + 1)

        self._srb_pipeline = nn.ModuleList([
            SequentialResidualBlock(
                channels[i],
                channels[i + 1],
            ) for i in range(configs.num_srbs)
        ])

        self._conv_bn = ConvBatchNorm(
            channels[-1],
            channels[-1],
            kernel_size=3,
            padding=1,
            stride=1
        )

        pix_out_channels = channels[-1] // (configs.up_scale ** 2)

        self._ps = nn.PixelShuffle(configs.up_scale)

        self._out_conv = nn.Conv2d(
            pix_out_channels,
            configs.out_channels,
            kernel_size=3,
            padding=1
        )

        self._act = torch.tanh

        self._sr_loss_fn = merge(
            [nn.MSELoss(), GradientProfileLoss()],
            weights=[20, 1e-4],
            device=self._device
        )

    def _get_loss(self, gt, pred):
        return self._sr_loss_fn(gt, pred)

    @torch.inference_mode()
    def infer(self, x):
        return self._forward_step(x)

    @torch.inference_mode()
    def evaluate(self, x):
        return self._forward_step(x)

    def _forward_step(
        self,
        x: Tensor
    ):
        srb_x = x.clone()
        for srb in self._srb_pipeline:
            srb_x = srb(srb_x)
        self._conv_bn(x)
        x = x + srb_x
        x = self._ps(x)
        x = self._out_conv(x)
        x = self._act(x)
        return x

    def forward(self, x, imgs):
        x = self._forward_step(x)
        return self._get_loss(imgs, x)
