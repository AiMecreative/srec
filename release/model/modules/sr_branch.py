"""
Augmentation module aims to boost the LR features' qualities. 
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
from model.modules.srb import SequentialResidualBlock, PixelShuffleBlock
from model.losses import GradientProfileLoss
from model.utils.utils import merge
from model.modules.tps_spatial_transformer import TPSSpatialTransformer
from model.modules.stn_head import STNHead
from functools import partial


class SRBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        img_size: List[int] = [32, 128],
        model_channels=64,
        num_srbs=5,
        scale=2,
    ):
        super(SRBranch, self).__init__()

        # use mask
        self.num_srbs = num_srbs

        self.tps_insize = [img_size[0], img_size[1] // scale]
        self.tps_outsize = [img_size[0] // scale, img_size[1] // scale]
        self.resize = partial(
            F.interpolate, size=self.tps_insize,
            mode='bilinear',
            align_corners=True
        )

        self.tps = TPSSpatialTransformer(
            outsize=self.tps_outsize,
            num_ctrlpoints=20,
            margins=[0.05, 0.05]
        )

        self.stn_head = STNHead(
            in_channels=in_channels,
            num_ctrlpoints=20,
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, model_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )

        for i in range(2, self.num_srbs + 2):
            setattr(self, f'block{i}', SequentialResidualBlock(model_channels))

        self.block7 = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(model_channels)
        )

        self.block8 = nn.Sequential(
            PixelShuffleBlock(model_channels, scale),
            nn.Conv2d(model_channels, out_channels, kernel_size=9, padding=4)
        )

        self.activate = torch.tanh

        # self._load_pretrain()

    def load_pretrain(self, file: str, origin: bool = False, map_location: str = None):
        state_dicts = torch.load(file, map_location=map_location)
        if origin:
            state_dicts = state_dicts['state_dict_G']
        self.load_state_dict(state_dicts)

    def forward(self, x: Tensor):
        if self.training:
            x = self.resize(x)
            _, ctrlpoints = self.stn_head(x)
            x, _ = self.tps(x, ctrlpoints)
        x = self.block1(x)
        res = x.clone()
        for i in range(2, self.num_srbs + 2):
            res = getattr(self, f'block{i}')(res)
        res = self.block7(res)
        x = x + res
        x = self.block8(x)
        x = self.activate(x)
        return x


class SRWrapper(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        super().__init__()

        self.module = module

    def load_pretrain(self, file: str, origin: bool = False, map_location: str = None):
        state_dicts = torch.load(file, map_location=map_location)
        self.load_state_dict(state_dicts)

    def forward(self, x: Tensor):
        return self.module.forward(x)
