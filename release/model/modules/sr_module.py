"""
SR module is designed to parse features from encoded image.
The aim of this module is to reconstruct the image and produce
super-resolution image.
"""

import copy
import torch
import torch.nn as nn
from torch import Tensor
from model.utils.utils import merge
from model.losses import GradientProfileLoss
from model.modules.srb import SequentialResidualBlock, PixelShuffleBlock


class SRer(nn.Module):

    def __init__(
        self,
        model_channels: int = 64,
        out_channels: int = 4,
        scale: int = 2,
        num_block_shifts: int = 7
    ) -> None:
        super(SRer, self).__init__()

        self.index = 8

        setattr(self, f'block{8}', nn.Sequential(
            PixelShuffleBlock(model_channels, scale),
            nn.Conv2d(model_channels, out_channels, kernel_size=9, padding=4),
        ))
        self.activate = torch.tanh

    def load_pretrain(self, file, map_location):
        state_dicts = torch.load(file, map_location)
        self_state_dict_keys = self.state_dict().keys()
        new_states = {}
        for k in self_state_dict_keys:
            new_states[k] = state_dicts[f'module.{k}']
        self.state_dict().update(new_states)

    def forward(self, x: Tensor):
        """
        [b,c,h,w]->[b,c,h,w]
        use bilineared HR to train
        """
        x = getattr(self, f'block{8}')(x)
        x = self.activate(x)
        return x


class SRerWrapper(nn.Module):

    def __init__(self, module) -> None:
        super().__init__()

        self.module = module

    def load_pretrain(self, file, map_location):
        state_dicts = torch.load(file, map_location)
        self.load_state_dict(state_dicts, strict=False)

    def forward(self, x):
        return self.module.forward(x)
