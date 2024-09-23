import torch
import numpy as np
from typing import List, Tuple
from torch import Tensor, Size
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from functools import partial


def merge(fns: List, weights: List = None):
    if weights is None:
        weights = [1. for _ in range(len(fns))]

    def _merge(*args):
        res = 0.
        for idx, loss in enumerate(fns):
            res += weights[idx] * loss(*args)
        return res
    return _merge


def collate_fn(
    batch: List,
    img_size: Size,
    mask: bool = True,
    perspectives: List[float] = [0.0, 0.0],
    rotations: List[float] = [0.0, 0.0]
) -> Tuple[str, List[Tensor], List[Tensor]]:
    """
    @param batch: a list of data tuples
        one data tuple contains label, hr_img and lr_img
    """

    def to_tensor(imgs: Image.Image, mask: bool = False):
        _to_tensor = transforms.ToTensor()
        img_tensor = _to_tensor(imgs)
        if mask:
            img_mask = imgs.convert('L')
            threshold = np.array(img_mask).mean()
            img_mask = img_mask.point(lambda x: 0 if x > threshold else 255)
            img_mask = _to_tensor(img_mask)
            img_tensor = torch.concat([img_tensor, img_mask], dim=0)
            # mjsynth ranges
            # img_tensor = img_tensor * 2 - 1.0
        return img_tensor        

    labels, hr_imgs, lr_imgs = zip(*batch)
    hr_size = [img_size[0], img_size[1]]
    lr_size = [img_size[0] // 2, img_size[1] // 2]
    # interpolate to target sizes
    perspective_scale = np.random.choice(np.array(perspectives))
    rotation_degree = np.random.choice(np.array(rotations))
    hr_transform = transforms.Compose([
        transforms.Resize(hr_size, InterpolationMode.BICUBIC),
        partial(to_tensor, mask=mask),
        transforms.RandomPerspective(perspective_scale),
        transforms.RandomRotation(rotation_degree)
    ])
    lr_transform = transforms.Compose([
        transforms.Resize(lr_size, InterpolationMode.BICUBIC),
        partial(to_tensor, mask=mask),
        transforms.RandomPerspective(perspective_scale),
        transforms.RandomRotation(rotation_degree)
    ])
    hr_imgs = [hr_transform(img) for img in hr_imgs]
    lr_imgs = [lr_transform(img) for img in lr_imgs]
    hr_imgs = torch.stack(hr_imgs, dim=0)
    lr_imgs = torch.stack(lr_imgs, dim=0)
    return labels, hr_imgs, lr_imgs
