import os
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
from torchvision.transforms import transforms, InterpolationMode


@hydra.main(config_path='configs', config_name='psrec_configs', version_base='v1.2')
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

    # scheduler = ModelScheduler(configs)
    # scheduler.load_model(configs.pretrain_weights, map_location=configs.device, strict=False)
    # scheduler.freeze_model(configs.freeze)
    # scheduler.train(train_dl, [infer_easy_dl, infer_medium_dl, infer_hard_dl])
    # res = scheduler.evaluate([infer_easy_dl, infer_medium_dl, infer_hard_dl])
    # print(res)
    avg_acc = []
    avg_psnr = []
    avg_ssim = []
    total_time = []

    psnr_fn = PeakSignalNoiseRatio(data_range=(0.0, 1.0))
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0))
    resize_fn = transforms.Resize((32, 128), InterpolationMode.BICUBIC)

    def _weighted_avg(stat_list):
        avg = 0
        for s, w in zip(stat_list, [0.3699, 0.3227, 0.3074]):
            avg += w * float(s)
        return avg
    eval_dl = [infer_easy_dl, infer_medium_dl, infer_hard_dl]
    if not isinstance(eval_dl, list):
        eval_dl = [eval_dl]
    for lid, loader in enumerate(eval_dl):
        print(f'\nevaluating loader {lid}')
        psnr = []
        ssim = []
        for eid, batch in enumerate(loader):
            _, hr_img, lr_img = batch
            sr_img = resize_fn(lr_img)
            psnr.append(psnr_fn(sr_img[:, :3, :, :], hr_img[:, :3, :, :]))
            ssim.append(ssim_fn(sr_img[:, :3, :, :], hr_img[:, :3, :, :]))
        psnr = sum(psnr) / len(psnr)
        ssim = sum(ssim) / len(ssim)

        print(f'\nloader {lid}| psnr {psnr} | ssim {ssim}')

        avg_psnr.append(psnr)
        avg_ssim.append(ssim)

    avg_psnr = _weighted_avg(avg_psnr)
    avg_ssim = _weighted_avg(avg_ssim)

    print(avg_psnr, avg_ssim)


if __name__ == '__main__':
    main()
