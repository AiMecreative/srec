"""
Augmentation module aims to boost the LR features' qualities.
"""
import copy
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


class PostAugment(nn.Module):
    def __init__(
        self,
        model_channels=64,
        num_srbs=3,
        num_srb_shifts: int = 3
    ):
        super(PostAugment, self).__init__()

        self.index_range = list(range(num_srb_shifts + 1, min(num_srb_shifts + num_srbs + 1, 7)))
        if num_srbs > 3:
            self.index_range.extend(list(range(9, 9 + num_srbs - 3)))
        
        for idx in self.index_range:
            setattr(self, f'block{idx}', SequentialResidualBlock(model_channels))

        setattr(self, f'block{7}', nn.Sequential(
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(model_channels)
        ))

    def load_pretrain(self, file, map_location):
        state_dicts = torch.load(file, map_location)
        self_state_dict_keys = self.state_dict().keys()
        new_states = {}
        for k in self_state_dict_keys:
            new_states[k] = state_dicts[f'module.{k}']
        self.state_dict().update(new_states)

    def forward(self, enc_x: Tensor):
        for idx in self.index_range:
            enc_x = getattr(self, f'block{idx}')(enc_x)
        enc_x = getattr(self, f'block{7}')(enc_x)
        return enc_x


class AugmentWrapper(nn.Module):

    def __init__(self, module) -> None:
        super().__init__()

        self.module = module

    def load_pretrain(self, file, map_location):
        state_dicts = torch.load(file, map_location)
        self.load_state_dict(state_dicts, strict=False)

    def forward(self, x):
        return self.module.forward(x)
