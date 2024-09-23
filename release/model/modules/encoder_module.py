"""
Encoder module can be the first `k` layers of recognizer,
for this work, we choose the first cnn layer as encoder layer.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.utils import save_image
from torchvision.transforms import transforms, InterpolationMode
from model.modules.tps_spatial_transformer import TPSSpatialTransformer
from model.modules.stn_head import STNHead
from model.modules.srb import SequentialResidualBlock
from functools import partial


class Encoder(nn.Module):
    """
    tps + stn and an encoder backbone
    add masks on input images
    """

    def __init__(
        self,
        img_channels: int = 4,
        img_size=[32, 128],
        model_channels: int = 64,
        scale: int = 2,
    ) -> None:
        super(Encoder, self).__init__()

        self.stn_head = STNHead(
            in_channels=img_channels,
            num_ctrlpoints=20
        )

        tps_insize = [img_size[0], img_size[1] // scale]
        tps_outsize = [img_size[0] // scale, img_size[1] // scale]
        self.tps = TPSSpatialTransformer(
            outsize=tps_outsize,
            num_ctrlpoints=20,
            margins=[0.05, 0.05]
        )

        self.in_resize = transforms.Resize(tps_insize, InterpolationMode.BICUBIC)

        self.block1 = nn.Sequential(
            nn.Conv2d(img_channels, model_channels, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        self.block2 = SequentialResidualBlock(model_channels)
        self.block3 = SequentialResidualBlock(model_channels)

    def load_pretrain(self, file, map_location):
        state_dicts = torch.load(file, map_location)
        self_state_dict_keys = self.state_dict().keys()
        new_states = {}
        for k in self_state_dict_keys:
            new_states[k] = state_dicts[f'module.{k}']
        self.state_dict().update(new_states)

    def forward(self, imgs: Tensor):
        """
        x -> masked_x, gray_x -> conv2 + conv1
        [b,ci,h,w] -> [b,co,h,w]
        """
        # HR: [b,c,h,w] -> [b,c,h/2,w/2]
        # LR: [b,c,h,w] -> [b,c,h,w]
        if self.training:
            imgs = self.in_resize(imgs)
            _, ctrlpoints = self.stn_head(imgs)
            imgs, _ = self.tps(imgs, ctrlpoints)
        x = self.block1(imgs)
        enc_x = x.clone()
        enc_x = self.block2(enc_x)
        enc_x = self.block3(enc_x)
        return enc_x, x


class EncoderWrapper(nn.Module):

    def __init__(self, module) -> None:
        super().__init__()

        self.module = module

    def load_pretrain(self, file, map_location):
        state_dicts = torch.load(file, map_location)
        self.load_state_dict(state_dicts, strict=False)
    
    def forward(self, x):
        return self.module.forward(x)