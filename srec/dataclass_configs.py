from dataclasses import dataclass
from typing import List, Union, Callable
from srec.modules.self_attn import LinearAttnFn, FullAttnFn


@dataclass
class TaskConfigs:

    epochs: int = 300,
    device: str = 'cuda:1',
    infer_dir: str = 'infer'
    tensorboard_dir: str = 'tensorboardx',
    save_dir: str = 'save_dir',
    save_interval: int = 10,
    eval_interval: int = 1,
    num_losses: int = 2
    load_pretrain: bool = False
    pretrain_weights: str = ''


@dataclass
class OptimConfigs:

    lr: float = 5e-5
    betas: List[float] = None  # [0.5, 0.999]


@dataclass
class DataConfigs:

    train_ds: str = "/home/zsplinx2/Project/xr/dataset/textzoom/textzoom/train1"
    eval_ds: str = "/home/zsplinx2/Project/xr/dataset/textzoom/textzoom/train2"
    infer_ds: str = "/home/zsplinx2/Project/xr/dataset/textzoom/textzoom/test/easy"
    charset: str = "1234567890abcdefghijklmnopqrstuvwxyz"
    img_shape: List[int] = None
    img_type: str = "RGB"
    min_label_len: int = 2
    max_label_len: int = 25
    min_img_len: int = 0
    min_psnr: float = 16.0
    defocus_blur: bool = False
    rotate_degree: int = 5

    # dataloader configs
    batch_size: int = 48
    shuffle: bool = True
    num_workers: int = 16


@dataclass
class HREncoderConfigs:

    in_channels: int = 3
    out_channels: int = 384
    kernel_sizes: List[int] = None      # = [9, 3, 3]
    paddings: List[int] = None          # = [4, 1, 1]
    strides: List[int] = None           # = [1, 2, 2]
    num_conv_bn: int = 3


@dataclass
class AugmentConfigs:
    in_channels: int = 384
    channel_mults: List[int] = None  # = [2]
    num_mini: int = 1
    num_mids: int = 1
    num_attn_heads: int = 4
    timesteps: int = 2000
    concat_hr_lr: bool = True
    concat_type: str = "mix"
    attn_fn: Callable = LinearAttnFn()


@dataclass
class SRConfigs:

    num_srbs: int = 5
    channels: int = 384
    out_channels: int = 3
    up_scale: int = 4  # defined by strides in `encoder`


# @dataclass
# class RecognizerConfigs:

#     backbone: str = 'parseq'
#     charset_size: int = 39
#     max_label_length: int = 25
#     dim_models: int = 384
#     num_heads: int = 12
#     dim_feedforward: int = 1536
#     device: str = 'cuda:1'
#     max_num_perm: int = 6
#     pretrain: str = ''


@dataclass
class RecognizerConfigs:

    backbone: str = 'crnn'
    max_label_len: int = 25
    in_channels: int = 512
    feature_size: int = None  # [32, 128]
    charset_size: int = 36+1
    mlp_hidden: int = 64
    rnn_hidden: int = 256
    load_pretrain: bool = True
    pretrain_recognizer: str = ''
