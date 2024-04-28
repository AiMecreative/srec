import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CTCLoss, CrossEntropyLoss, MSELoss


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
        sr_grad_field = generate_gradient_field(sr_imgs)
        hr_grad_field = generate_gradient_field(hr_imgs)
        return self.metric(sr_grad_field, hr_grad_field)


class UncertaintyWeightLoss(nn.Module):
    """
    uncertainty loss used for multitask
    at least 2 losses should be provided
    $L = 1/(2 \sigma_1) * l1 + 1/(2 \sigma_2) * l2 + ... + log (1 + \sigma_1 \sigma_2 ...)$
    """

    def __init__(
        self,
        device: str,
        num_losses: int = 2,
    ) -> None:
        super(UncertaintyWeightLoss, self).__init__()

        __sigmas = torch.ones(num_losses, requires_grad=True)
        self.__sigmas = nn.Parameter(__sigmas)

        for m in self.modules():
            m = m.to(device)

    def sigmas_no_grad(self):
        sigmas_ng = []
        for sig in self.__sigmas:
            sig_no_grad: Tensor = sig.detach().requires_grad_(False)
            sigmas_ng.append(sig_no_grad)
        return sigmas_ng

    def forward(self, losses):
        loss_sum = 0.
        for idx, loss in enumerate(losses):
            loss_sum = (
                loss_sum
                + 0.5 * 1 / (self.__sigmas[idx] ** 2) * loss
                + torch.log(1 + self.__sigmas[idx] ** 2)
            )
        return loss_sum
