import torch
import torch.nn as nn
from torch import Tensor
from srec.utils.utils import InitDevice


class SequentialResidualBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,  # same with the gru in_channels
    ) -> None:
        super(SequentialResidualBlock, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.gru1 = nn.GRU(
            out_channels,
            out_channels // 2,  # for bidirectional
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.gru2 = nn.GRU(
            out_channels,
            out_channels // 2,  # for bidirectional
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x: Tensor):
        x = self.convs(x)
        x = x.permute(0, 2, 3, 1)
        x_shape = x.shape
        x = x.reshape([-1, x_shape[2], x_shape[3]])
        gru_x = self.gru1(x)[0]
        x = self.gru2(x + gru_x)[0]
        x = x.reshape([x_shape[0], x_shape[1], x_shape[2], x_shape[3]])
        x = x.permute(0, 3, 1, 2)
        return x
