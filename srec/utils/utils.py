import time
import math
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, Size
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from functools import wraps
from typing import List, Tuple, Callable


class InitDevice:

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device

    def move(self, init_fn):
        def _wrapper(module, *args, **kwargs):
            init_fn(module, *args, **kwargs)
            for m in module.modules():
                m = m.to(module._device)
        return _wrapper


def merge(fns: List, weights: List = None, device: str = "cuda:0"):
    if weights is None:
        weights = [1. for _ in range(len(fns))]

    def _merge(*args):
        res = 0.
        for idx, loss in enumerate(fns):
            res += weights[idx] * loss(*args)
        return res
    return _merge


def run_timer(cuda_fn: Callable):
    @wraps(cuda_fn)
    def _wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.time()
        cuda_fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()
        print(f"{cuda_fn.__name__} spent {(end - time) / 1000}s")
    return _wrapper


def collate_fn(batch: List, img_size: Size) -> Tuple[str, List[Tensor], List[Tensor]]:
    """
    @param batch: a list of data tuples
        one data tuple contains label, hr_img and lr_img
    """
    img_size = [img_size[0], img_size[1]]
    labels, hr_imgs, lr_imgs = zip(*batch)
    # interpolate to target sizes
    resized_hr = []
    resized_lr = []
    trans = transforms.Compose([
        transforms.Resize(img_size, InterpolationMode.NEAREST)
    ])
    for hr in hr_imgs:
        resized_hr.append(trans(hr))
    for lr in lr_imgs:
        resized_lr.append(trans(lr))
    resized_hr = torch.stack(resized_hr)
    resized_lr = torch.stack(resized_lr)
    return labels, resized_hr, resized_lr


def _gaussian_filter(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    gaussian_kernel = (
        (1./(2.*math.pi*variance))
        * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    filter = nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,
        bias=False,
        padding=kernel_size//2
    )

    filter.weight.data = gaussian_kernel
    filter.weight.requires_grad = False

    return filter


def _defocus_filter(radius: float, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(-radius, radius + 1)
    kernel_size = x_coord.shape[0]
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.T

    blur_kernel = x_grid ** 2 + y_grid ** 2 <= radius ** 2

    # Make sure sum of values in gaussian kernel equals 1.
    blur_kernel = blur_kernel / torch.sum(blur_kernel)

    # Reshape to 2d depthwise convolutional weight
    blur_kernel = blur_kernel.view(1, 1, kernel_size, kernel_size)
    blur_kernel = blur_kernel.repeat(channels, 1, 1, 1)

    filter = nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,
        bias=False,
        padding=kernel_size//2
    )

    filter.weight.data = blur_kernel
    filter.weight.requires_grad = False

    gaussian_conv = _gaussian_filter(kernel_size * 2, sigma, channels)

    return nn.Sequential(
        filter,
        gaussian_conv
    )


class DefocusBlur:

    def __init__(
        self,
        rng: np.random.Generator = np.random.default_rng(),
        kernel_size: int = 5,
        sigma: float = 2.,
        channels: int = 3
    ) -> None:
        self._rng = rng
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels

    def __call__(self, imgs: Tensor, radius: float = 9, sigma: float = 2., threshold: float = 0.5):
        """
        @param threshold: defines whether this batch of imgs should be blurred
            if random number is greater than threshold,
            then this batch should be blurred
        """
        if self._rng.uniform(0, 1) < threshold:
            return imgs
        conv = _defocus_filter(radius, sigma=sigma)
        return conv(imgs)
