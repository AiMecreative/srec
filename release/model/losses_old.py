import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CTCLoss, CrossEntropyLoss, MSELoss
from typing import List


def generate_gradient_field(img: Tensor) -> Tensor:
    _pad_right = (0, 1, 0, 0)
    _pad_left = (1, 0, 0, 0)
    _pad_top = (0, 0, 1, 0)
    _pad_bottom = (0, 0, 0, 1)

    b, c, h, w = img.shape
    right_shifted = F.pad(img, _pad_right)[:, :, :, 1:]
    left_shifted = F.pad(img, _pad_left)[:, :, :, :w]
    top_shifted = F.pad(img, _pad_top)[:, :, :h, :]
    bottom_shifted = F.pad(img, _pad_bottom)[:, :, 1:, :]

    grad_field = 0.5 * ((right_shifted - left_shifted) ** 2 + (top_shifted - bottom_shifted) ** 2 + 1e-6) ** 0.5

    return grad_field


class GradientProfileLoss(nn.Module):
    """
    L_{GP} = E[nabla I_{SR} - nabla I_{HR}]_1
    """

    def __init__(self) -> None:
        super(GradientProfileLoss, self).__init__()

        self.metric = nn.L1Loss()

    def forward(self, sr_imgs: Tensor, hr_imgs: Tensor) -> float:
        sr_grad_field = generate_gradient_field(sr_imgs[:, :3, :, :])
        hr_grad_field = generate_gradient_field(hr_imgs[:, :3, :, :])
        return self.metric(sr_grad_field, hr_grad_field)


class UncertaintyWeightLoss(nn.Module):
    """
    uncertainty loss used for multitask
    at least 2 losses should be provided
    $L = 1/(2 \sigma_1) * l1 + 1/(2 \sigma_2) * l2 + ... + log (1 + \sigma_1 \sigma_2 ...)$
    """

    def __init__(
        self,
        loss_types,
        init_weights,
        weight_decay: float = 0.01
    ) -> None:
        super(UncertaintyWeightLoss, self).__init__()

        self.loss_types = loss_types
        self.weight_decay = weight_decay
        self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float, requires_grad=True), requires_grad=True)

    @torch.no_grad()
    def get_weights(self):
        weight_dict = {}
        for idx, t in enumerate(self.loss_types):
            weight_dict[t] = self.weights[idx] ** 2
        return weight_dict

    def forward(self, losses):
        loss_sum = 0.
        for idx, loss in enumerate(losses):
            loss_sum = (
                loss_sum
                + self.weights[idx] ** 2 * loss
                + self.weight_decay * torch.log(1 + 1 / (self.weights[idx] ** 2))
            )
        return loss_sum


class SRLoss(nn.Module):

    def __init__(self, weights: List[float] = [20, 1e-4], requires_gp: bool = True) -> None:
        super().__init__()

        self.mse_loss = nn.MSELoss()
        self.gp_loss = GradientProfileLoss()
        self.weights = weights
        self.requires_gp = requires_gp

    def forward(self, sr_imgs, hr_imgs):
        if self.requires_gp:
            return (
                self.weights[0] * self.mse_loss(sr_imgs, hr_imgs)
                + self.weights[1] * self.gp_loss(sr_imgs, hr_imgs)
            )
        return self.mse_loss(sr_imgs, hr_imgs)


class RecLoss(nn.Module):

    def __init__(self, blank_id: int = 0) -> None:
        super().__init__()

        self.ctc_loss = nn.CTCLoss(blank_id, zero_infinity=True)

    def forward(self, preds, pred_lengths, targets, target_lengths):
        return self.ctc_loss(preds, targets, pred_lengths, target_lengths)


class MultitaskLoss(nn.Module):

    def __init__(
        self,
        init_weights: List[Tensor],
        learn_weights: bool = False,
        loss_types: List[str] = ['sr'],
        weight_decay: float = 0.01,
        sr_loss_weights: List[float] = [20, 1e-4],
        sr_requires_gp: bool = True,
        rec_blank_id: int = 0,
    ) -> None:
        super().__init__()

        self.loss_type = loss_types
        self.init_weights = torch.tensor(init_weights, dtype=torch.float)
        self.learn_weights = learn_weights

        if 'sr' in loss_types:
            self.sr_loss_fn = SRLoss(sr_loss_weights, sr_requires_gp)
        if 'rec' in loss_types:
            self.rec_loss_fn = RecLoss(rec_blank_id)
        if learn_weights:
            self.uncertainty = UncertaintyWeightLoss(loss_types, init_weights, weight_decay)

    def forward(
        self,
        sr_imgs=None,
        hr_imgs=None,
        preds=None,
        pred_lengths=None,
        targets=None,
        target_lengths=None
    ):
        losses = []
        loss_dict = {}
        if 'sr' in self.loss_type:
            sr_loss = self.sr_loss_fn(sr_imgs, hr_imgs)
            losses.append(sr_loss)
            loss_dict['sr'] = sr_loss.detach()
        if 'rec' in self.loss_type:
            rec_loss = self.rec_loss_fn(preds, pred_lengths, targets, target_lengths)
            losses.append(rec_loss)
            loss_dict['rec'] = rec_loss.detach()
        if self.learn_weights:
            return self.uncertainty(losses), loss_dict, self.uncertainty.get_weights()
        loss = 0
        for idx in range(len(losses)):
            loss = loss + self.init_weights[idx] * losses[idx]
        weight_dict = {}
        for idx, t in enumerate(self.loss_type):
            weight_dict[t] = self.init_weights[idx]
        return loss, loss_dict, weight_dict
