import torch
import torch.nn as nn
from torch import Tensor


class GRUBlock(nn.Module):

    def __init__(
        self,
        model_channels: int
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(model_channels, model_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(model_channels, model_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        x, _ = self.gru(x)
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


class SequentialResidualBlock(nn.Module):
    def __init__(self, channels):
        super(SequentialResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GRUBlock(channels)
        self.mish = nn.Mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GRUBlock(channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.mish(res)
        res = self.conv2(res)
        res = self.bn2(res)
        res = res.transpose(-1, -2)
        res = self.gru1(res)
        res = res.transpose(-1, -2)
        x = x + res
        x = self.gru2(x)
        return x


class PixelShuffleBlock(nn.Module):

    def __init__(
        self,
        model_channels,
        scale
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(model_channels, model_channels * scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.mish = nn.Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.mish(x)
        return x
