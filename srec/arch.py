import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from srec.utils.utils import InitDevice


class EDDModule(nn.Module):

    DEVICE = ''
    INIT_DEVICE = InitDevice(DEVICE)

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super(EDDModule, self).__init__()

        self._device = self.DEVICE

    def resume(self, state_dicts: str, per_layer: bool = False):
        """
        model resume from state_dicts
        where state_dicts is a path to .pth file
        """
        if not per_layer:
            self.load_state_dict(state_dicts)
        else:
            self_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dicts.items() if k in self_dict}
            self_dict.update(pretrained_dict)
            self.load_state_dict(pretrained_dict)

    def save(self, state_dicts: str):
        """
        save model state_dicts,
        where state_dicts is a path to .pth file
        """
        if state_dicts is not None and state_dicts != "":
            torch.save(self.state_dict(), state_dicts)
            return
        return self.state_dict()

    @torch.inference_mode()
    def infer(
        self,
        *args,
        **kwargs
    ):
        raise NotImplementedError("EDDArch infer has not been implemented")

    @torch.inference_mode()
    def evaluate(
        self,
        *args,
        **kwargs
    ):
        raise NotImplementedError("EDDArch evaluate has not been implemented")

    def forward(
        self,
        *args,
        **kwargs
    ):
        raise NotImplementedError("EDDArch forward has not been implemented")


class EDDArch(EDDModule):

    @EDDModule.INIT_DEVICE.move
    def __init__(
            self,
            encoder: EDDModule,
            augment: EDDModule,
            recognizer: EDDModule,
            sr: EDDModule
    ) -> None:
        """
        Encoder-Diffusion-Decoders architecture:
        where the encoder is unified for every inputs,
        the augmentation module will boost the weak features
        (in this work, the augmentation module is realized 
        by a mini-diffusion),
        the decoders contains recognizer and sr modules,
        for recognition task and super-resolution task respectively in this work
        """
        super().__init__(
            encoder,
            augment,
            recognizer,
            sr
        )

        self.encoder = encoder
        self.augment = augment
        self.recognizer = recognizer
        self.sr = sr

    def resume(self, state_dicts: str):
        state_dicts = torch.load(state_dicts)
        self.encoder.resume(state_dicts["encoder"])
        self.augment.resume(state_dicts["augment"])
        self.recognizer.resume(state_dicts["recognizer"])
        self.sr.resume(state_dicts["sr"])

    def save(self, state_dicts: str):
        torch.save(
            {
                "encoder": self.encoder.save(""),
                "augment": self.augment.save(""),
                "recognizer": self.recognizer.save(""),
                "sr": self.sr.save("")
            },
            state_dicts
        )

    @torch.inference_mode()
    def infer(
        self,
        imgs,
        requires_augment: bool
    ):
        """
        model inference
            use LR images only to predict labels and generate SR images
        input: 
            imgs: LR images or HR images
            requires_augment: if enabled, the encoded image features will be
                passed on model augmentation module
        output: 
            predicted labels and generated SR images
        """
        raise NotImplementedError("EDDArch infer has not been implemented")

    @torch.inference_mode()
    def evaluate(
        self,
        labels,
        hr_imgs,
        lr_imgs=None,
        requires_augment: bool = False
    ):
        """
        model evaluating
            use HR images and/or LR images to evaluate model
        input:
            lables: tokenized characters
            hr_imgs: HR images
            lr_imgs: LR images or None
            requres_augment: if use augmentation module,
                if enabled, the lr_imgs should not be None
        output:
            multi-task (combination of recognize loss and sr loss) loss,
            recognize loss,
            sr loss,
            loss weights,
            or feature loss if requires_augment is enabled,
            recognize accuracy,
            psnr between sr_imgs and hr_imgs,
            ssim between sr_igms and hr_imges,
        """
        raise NotImplementedError("EDDArch evaluate has not been implemented")

    def forward(
        self,
        labels,
        hr_imgs,
        lr_imgs=None,
        requires_augment: bool = False
    ):
        """
        model training
            use HR images and/or LR images to train model,
            if requires augmentation, the LR images will be
            passed into the model to train the augmentation module
        input:
            labels: tokenized characters
            hr_imgs: HR images
            lr_imgs: LR images or None
            requres_augment: if use augmentation module,
                if enabled, the lr_imgs should not be None
        output:
            multi-task (combination of recognize loss and sr loss) loss,
            recognize loss,
            sr loss,
            loss weights,
            or feature loss if requires_augment is enabled
        """
        raise NotImplementedError("EDDArch forward has not been implemented")
