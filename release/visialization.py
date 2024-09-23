import os
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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
from expr.ablation.tsrn import TSRN
from torchvision.utils import save_image, make_grid
from torchvision.transforms import transforms
from PIL import Image


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

    scheduler = ModelScheduler(configs)
    scheduler.load_model(configs.pretrain_weights, map_location=configs.device, strict=False, key='model')
    scheduler.freeze_model(configs.freeze)
    scheduler.infer([infer_easy_dl, infer_medium_dl, infer_hard_dl])  # , infer_medium_dl, infer_hard_dl])

# easy: 1616
# hard: 1343
# medium: 1410


def show(label1, label2):
    with open(label1) as f1:
        label1_lines = f1.readlines()
    with open(label2) as f2:
        label2_lines = f2.readlines()
    with open('cross_infer/a_labels.txt', 'w') as cf:
        for l1, l2 in zip(label1_lines, label2_lines):
            if '<' in l1.split()[-2] and '<' not in l2.split()[-2]:
                if l1.split()[0] == l2.split()[0]:
                    cf.write(f'{l1.split()[0]}\n')


if __name__ == '__main__':
    # main()
    # show('infer_tsrn_rnn_mlp/a_labels.txt', 'infer_gp_w1e4/a_labels.txt')

    # indices = []
    # level = []
    # folder1 = 'infer_tsrn_rnn_mlp'
    # folder2 = 'infer_gp_w1e4'
    # with open('cross_infer/a_labels.txt') as cf:
    #     for idx in cf.readlines():
    #         indices.append(idx.split()[0].split('_')[-1])
    #         level.append(idx.split()[0].split('_')[0])
    # for i, idx in enumerate(indices):
    #     lr_img = f'{folder1}/{level[i]}_lr_{idx}.png'
    #     hr_img = f'{folder1}/{level[i]}_hr_{idx}.png'
    #     sr_img1 = f'{folder1}/{level[i]}_sr_{idx}.png'
    #     sr_img2 = f'{folder2}/{level[i]}_sr_{idx}.png'
    #     lr_img = Image.open(lr_img).convert('RGB')
    #     hr_img = Image.open(hr_img).convert('RGB')
    #     sr_img1 = Image.open(sr_img1).convert('RGB')
    #     sr_img2 = Image.open(sr_img2).convert('RGB')

    #     lr_img = transforms.PILToTensor()(lr_img) / 255.0
    #     hr_img = transforms.PILToTensor()(hr_img) / 255.0
    #     sr_img1 = transforms.PILToTensor()(sr_img1) / 255.0
    #     sr_img2 = transforms.PILToTensor()(sr_img2) / 255.0

    #     save_image([lr_img, sr_img1, sr_img2, hr_img], f'cross_infer/{level[i]}_lrsr12hr_{idx}.png')
    # lr_img = Image.open('infer_gp_w1e4/easy_hr_6.png').convert('RGB')
    # lr_img = transforms.PILToTensor()(lr_img) / 255.0

    # save_image(lr_img[:3, :, :], 'hr_1.png')
    # save_image(lr_img[-1, :, :], 'hr_mask_1.png')
    def mish(x):

        soft_plus = np.log(1 + np.exp(x))
        return x * np.tanh(soft_plus)

    x_arr = np.linspace(-5, 5, 100)
    y_arr = np.array([mish(x_i) for x_i in x_arr])

    plt.figure()
    plt.title('Mish activation function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_arr, y_arr, linewidth=3, label='$y=tanh(\ln (1+ \exp(x)))$')
    plt.grid()
    plt.legend()
    plt.savefig('mish.svg')
