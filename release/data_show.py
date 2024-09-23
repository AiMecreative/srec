import hydra
import torch
import torch.nn as nn
import torch.optim as optim
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


@hydra.main(config_path='configs', config_name='configs', version_base='v1.2')
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
    eval_dl = dm.data_loader(data_conf.infer_ds, collate_fn=collate_eval, batch_size=1)

    print(f'load train dataloader: length {len(train_dl)} | batch_size {train_dl.batch_size}')
    print(f'load eval dataloader: length {len(eval_dl)} | batch_size {eval_dl.batch_size}')

# easy: 1616
# hard: 1343
# medium: 1410


if __name__ == '__main__':
    main()
