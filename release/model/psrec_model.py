"""
Paralell Super-resolution and Recognizer model(PSRec)
"""

import torch
import torch.nn as nn
from torch import Tensor
from model.model_wrapper import ModelWrapper
from model.modules.augment_module import PostAugment
from model.modules.encoder_module import Encoder
from model.modules.recognizer_module import CRNNRecognizer, CRNNBased
from model.modules.sr_module import SRer
from model.losses import MultitaskLoss
from torchvision.transforms import transforms, InterpolationMode

from configs.mappings import (
    PSRecConfigs, EncoderConfigs,
    AugmentConfigs, RecognizerConfigs, SRConfigs
)


class PSRec(nn.Module):
    """
    Mainly used for passing features between moduels,
    handling losses and returning them.

    The training loop, inferring loop and testing loop have different maners
    """

    def __init__(
        self,
        configs: PSRecConfigs
    ) -> None:
        super(PSRec, self).__init__()

        self.encoder = self._get_encoder(configs.encoder)
        self.augment = self._get_augment(configs.augment)
        self.recognizer = self._get_recognizer(configs.recognizer)
        self.sr = self._get_sr(configs.sr)

        self.requires_augment = configs.requires_augment

        self.down_sample = transforms.Resize(
            [configs.image_size[0] // configs.scale, configs.image_size[1] // configs.scale],
            InterpolationMode.BILINEAR
        )

    def _get_encoder(self, configs: EncoderConfigs):
        return ModelWrapper(Encoder(
            img_channels=configs.img_channels,
            img_size=[configs.img_size[0], configs.img_size[1]],
            model_channels=configs.model_channels,
            scale=configs.scale,
        ))

    def _get_augment(self, configs: AugmentConfigs):
        return ModelWrapper(PostAugment(
            model_channels=configs.model_channels,
            num_srbs=configs.num_srbs,
        ))

    def _get_recognizer(self, configs: RecognizerConfigs):
        return ModelWrapper(CRNNBased(
            img_channel=configs.img_channel,
            img_height=configs.img_height,
            img_width=configs.img_width,
            num_classes=configs.num_classes,
            seq_hidden=configs.seq_hidden,
            rnn_hidden=configs.rnn_hidden,
            model_channels=configs.model_channels,
            scale=configs.scale,
            out_channels=configs.out_channels,
            mlp_classifier=configs.mlp_classifier,
            side_rnn=configs.side_rnn,
            input_type=configs.input_type,
        ))

    def _get_sr(self, configs: SRConfigs):
        return ModelWrapper(SRer(
            model_channels=configs.model_channels,
            out_channels=configs.out_channels,
            scale=configs.scale,
        ))

    @torch.inference_mode()
    def infer(
        self,
        imgs: Tensor,
        requires_augment: bool = False
    ):
        enc_x, cnn_x = self.encoder(imgs)
        if requires_augment:
            enc_x = self.augment(enc_x)
        enc_x = enc_x + cnn_x
        logits = self.recognizer(enc_x)
        srs = self.sr(enc_x)
        return logits, srs

    @torch.inference_mode()
    def evaluate(
        self,
        hr_imgs: Tensor,
        lr_imgs: Tensor = None,
        requires_augment: bool = False
    ):
        """
        the behavior is same with `train` except for gradient=False
        used only on training loop
        """
        if requires_augment:
            enc_x, cnn_x = self.encoder(lr_imgs)
            enc_x = self.augment(enc_x)
        else:
            down_hr = self.down_sample(hr_imgs)
            enc_x, cnn_x = self.encoder(down_hr)
        enc_x = enc_x + cnn_x
        logits = self.recognizer(enc_x)
        srs = self.sr(enc_x)
        return logits, srs

    def forward(
        self,
        hr_imgs: Tensor,
        lr_imgs: Tensor = None,
        requires_augment: bool = False
    ):
        """
        if requires_augment=False, lr_imgs can be None, because it will not be used
        if requires_augment=True, lr_imgs can't be None,
        hr_imgs can't be None for any time
        """
        if requires_augment:
            enc_x, cnn_x = self.encoder(lr_imgs)
            enc_x = self.augment(enc_x)
        else:
            down_hr = self.down_sample(hr_imgs)
            enc_x, cnn_x = self.encoder(down_hr)
        enc_x = cnn_x + enc_x
        logits = self.recognizer(enc_x)
        srs = self.sr(enc_x)
        logit_lengths = torch.full(
            (logits.shape[1],),
            fill_value=logits.shape[0],
            dtype=torch.long,
            device=logits.device
        )

        return (logits, logit_lengths), srs
