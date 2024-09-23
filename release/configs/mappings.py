from dataclasses import dataclass
from typing import List


@dataclass
class DataConfigs:

    train_ds: str = '/home/zsplinx2/Project/xr/dataset/textzoom/textzoom/train1'
    eval_ds: str = '/home/zsplinx2/Project/xr/dataset/textzoom/textzoom/train2'
    infer_ds: str = '/home/zsplinx2/Project/xr/dataset/textzoom/textzoom/test/easy'
    charset: str = '1234567890abcdefghijklmnopqrstuvwxyz'
    img_shape: List[int] = None
    read_mode: str = 'RGB'
    img_type: str = 'hrlr'                   # 'hr', 'lr', 'hrlr'
    min_label_len: int = 2
    max_label_len: int = 25
    min_img_len: int = 0
    perspective_scale: List[float] = None  # [0.2, 0.6]
    rotate_degree: List[float] = None      # [2, 10]

    # dataloader configs
    batch_size: int = 48
    shuffle: bool = True
    num_workers: int = 16


@dataclass
class OptimConfigs:

    lr: float = 5e-5
    betas: List[float] = None  # [0.5, 0.999]


@dataclass
class EncoderConfigs:

    img_channels: int = 4
    img_size = [32, 128]
    model_channels: int = 64
    scale: int = 2


@dataclass
class AugmentConfigs:

    model_channels: int = 64
    num_srbs: int = 3


@dataclass
class SRConfigs:

    model_channels: int = 64
    out_channels: int = 4
    scale: int = 2


@dataclass
class RecognizerConfigs:

    in_channels: int = 64
    num_class: int = 37
    mid_linear: int = 64
    rnn_hidden: int = 256
    scale: int = 2


@dataclass
class PSRecConfigs:

    encoder: EncoderConfigs
    augment: AugmentConfigs
    recognizer: RecognizerConfigs
    sr: SRConfigs
    scale: int = 2
    multiloss: List[float] = None
    learn_loss: bool = False
    image_size: List[int] = None  # [64, 256]
    requires_augment: bool = False
    checkpoints: str = ''
