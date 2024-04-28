import torch
import torch.nn as nn
from hydra.utils import instantiate
from srec.arch import EDDModule, EDDArch
from srec.utils.utils import InitDevice, merge
from loss.losses import UncertaintyWeightLoss, GradientProfileLoss


class Srec(EDDArch):

    @EDDModule.INIT_DEVICE.move
    def __init__(
        self,
        encoder: EDDModule,
        augment: EDDModule,
        recognizer: EDDModule,
        sr: EDDModule,
        load_pretrain: bool = False,
        pretrain_weights: str = '',
    ) -> None:
        super().__init__(
            encoder,
            augment,
            recognizer,
            sr
        )

        if load_pretrain:
            self.resume(pretrain_weights)

        self._mulitask_loss_fn = UncertaintyWeightLoss(
            device=self._device,
            num_losses=2
        )

    @torch.inference_mode()
    def infer(
        self,
        imgs,
        requires_augment: bool = False
    ):
        feat = self.encoder(imgs)
        if requires_augment:
            feat = self.augment.infer(feat)
        pred_labels = self.recognizer.infer(feat)
        pred_srs = self.sr.infer(feat)
        return pred_labels, pred_srs

    @torch.inference_mode()
    def evaluate(
        self,
        labels,
        hr_imgs,
        lr_imgs=None,
        requires_augment: bool = False
    ):
        feat_hr = self.encoder(hr_imgs)
        feat = feat_hr
        if requires_augment:
            feat_lr = self.encoder(lr_imgs)
            feat_sr = self.augment.evaluate(feat_lr)
            feat = feat_sr
        pred_labels = self.recognizer.evaluate(feat)
        pred_srs = self.sr.evaluate(feat)
        return pred_labels, pred_srs

    def forward(
        self,
        labels,
        lengths,
        hr_imgs,
        lr_imgs=None,
        requires_augment: bool = False
    ):
        """
        output:
            loss, (rec_loss, sr_loss), loss_weight
            or feat_loss only if requires_augment == True
        """
        feat_hr = self.encoder(hr_imgs)
        if requires_augment:
            feat_lr = self.encoder(lr_imgs)
            feat_loss = self.augment(feat_lr, feat_hr)
            return feat_loss, (0, 0), 1
        # no augmentation
        rec_loss = self.recognizer(feat_hr, labels, lengths)
        sr_loss = self.sr(feat_hr, hr_imgs) * 10

        loss = self._mulitask_loss_fn([rec_loss, sr_loss])
        loss_weights = self._mulitask_loss_fn.sigmas_no_grad()

        return (loss, (rec_loss, sr_loss), loss_weights)
