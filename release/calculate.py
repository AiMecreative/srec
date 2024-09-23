import os
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile
from time import time
from model.model_scheduler import ModelScheduler
from model.utils.utils import collate_fn
from functools import partial
from data.data_module import DataModule
from hydra.utils import instantiate
from omegaconf import OmegaConf
from configs.mappings import DataConfigs
from model.modules.rec_branch import CRNNBased
from model.modules.sr_branch import SRBranch
from model.losses import MultitaskLoss
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
TEXTZOOM_WEIGHTS = [0.3699, 0.3227, 0.3074]


@hydra.main(config_path='expr/srb3', config_name='srb3', version_base='v1.2')
def main(configs: OmegaConf):

    # configs data
    data_conf: DataConfigs = instantiate(configs.data)
    collate_train = partial(
        collate_fn,
        img_size=data_conf.img_shape,
        perspectives=data_conf.perspective_scale,
        rotations=data_conf.rotate_degree
    )
    collate_eval = partial(
        collate_fn,
        img_size=data_conf.img_shape
    )
    dm = DataModule(data_conf)
    train_dl = dm.data_loader(data_conf.train_ds, collate_fn=collate_train, batch_size=data_conf.batch_size,)
    infer_easy_dl = dm.data_loader(data_conf.infer_ds[0], collate_fn=collate_eval, batch_size=1)
    infer_medium_dl = dm.data_loader(data_conf.infer_ds[1], collate_fn=collate_eval, batch_size=1)
    infer_hard_dl = dm.data_loader(data_conf.infer_ds[2], collate_fn=collate_eval, batch_size=1)

    print(f'load train dataloader: length {len(train_dl)} | batch_size {train_dl.batch_size}')
    print(f'load eval dataloader: length {len(infer_easy_dl)} | batch_size {infer_easy_dl.batch_size}')
    print(f'load eval dataloader: length {len(infer_medium_dl)} | batch_size {infer_medium_dl.batch_size}')
    print(f'load eval dataloader: length {len(infer_hard_dl)} | batch_size {infer_hard_dl.batch_size}')

    scheduler = ModelScheduler(configs)
    scheduler.load_model(configs.pretrain_weights, map_location=configs.device, strict=True, key='model')
    scheduler.freeze_model(configs.freeze)
    res = scheduler.evaluate([infer_easy_dl, infer_medium_dl, infer_hard_dl])
    print(res)

    inputs_hr = torch.randn(1, 4, 32, 128).to(scheduler.device)
    inputs_lr = torch.randn(1, 4, 16, 64).to(scheduler.device)
    flops, params = profile(scheduler.model, (inputs_hr, inputs_lr, True,))
    print(str(flops/1000**3)+'G', str(params/1000**2)+'M')


def weighted_avg(stat_list):
    """
    input List[float] or str like '| float | float | float |'
    """
    avg = 0
    if isinstance(stat_list, str):
        stat_list = stat_list.split(' ')[1::2]
    for s, w in zip(stat_list, TEXTZOOM_WEIGHTS):
        avg += w * float(s)
    return avg


if __name__ == '__main__':
    # res = weighted_avg('| 0.7951 | 0.5326 | 0.6226 |')
    # print(round(res, 4))
    main()
