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
from model.modules.sr_branch import SRBranch, SRWrapper
from model.losses import MultitaskLoss
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


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
    infer_easy_dl = dm.data_loader(data_conf.infer_ds[0], collate_fn=collate_eval, batch_size=1)
    infer_medium_dl = dm.data_loader(data_conf.infer_ds[1], collate_fn=collate_eval, batch_size=1)
    infer_hard_dl = dm.data_loader(data_conf.infer_ds[2], collate_fn=collate_eval, batch_size=1)

    print(f'load eval dataloader: length {len(infer_easy_dl)} | batch_size {infer_easy_dl.batch_size}')
    print(f'load eval dataloader: length {len(infer_medium_dl)} | batch_size {infer_medium_dl.batch_size}')
    print(f'load eval dataloader: length {len(infer_hard_dl)} | batch_size {infer_hard_dl.batch_size}')

    # attributes
    save_dir = configs.save_dir
    infer_dir = configs.infer_dir

    # configs models
    sr_model = SRBranch()

    n_gpu = len(configs.device_ids)
    ddp = n_gpu > 1
    main_device = configs.device

    # if ddp:
    #     sr_model = nn.DataParallel(sr_model)
    # else:
    #     sr_model = SRWrapper(sr_model)
    sr_model.load_state_dict(torch.load('legacy/pretrain_weights/pretrain_sr.pth')['state_dict_G'])
    sr_model = sr_model.to(main_device)


    # frozen_brach = configs.frozen_branch
    # if 'sr' in frozen_brach:
    #     for p in sr_model.parameters():
    #         p.requires_grad_(False)

    # print(f'load models: DDP {ddp} | n_gpu {n_gpu} | frozen {frozen_brach}')

    # configs optim
    optimizers = optim.Adam(
        sr_model.parameters(), lr=5e-4, betas=[0.5, 0.999]
    )

    # configs loss module
    loss_type = ['sr']
    multi_loss = MultitaskLoss(
        init_weights=[0.5],
        loss_types=loss_type,
        learn_weights=False
    )

    if ddp:
        multi_loss = nn.DataParallel(multi_loss)
    multi_loss = multi_loss.to(main_device)

    print(f'init loss functions')

    # configs metrics
    psnr_fn = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(main_device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(main_device)

    start_time = time()
    # train_len = len(train_dl)
    best_psnr = 0
    # for epoch in range(configs.num_epochs):
    #     sr_model.train()
    #     losses = []
    #     for tid, tbatch in enumerate(train_dl):
    #         print(f'\repoch {epoch + 1}/{configs.num_epochs} | tid {tid + 1}/{train_len}', end='', flush=True)
    #         labels, hr_imgs, lr_imgs = tbatch

    #         hr_imgs = hr_imgs.to(main_device)
    #         lr_imgs = lr_imgs.to(main_device)

    #         sr_imgs = sr_model(lr_imgs)

    #         loss = multi_loss(sr_imgs=sr_imgs, hr_imgs=hr_imgs)

    #         optimizers.zero_grad()
    #         loss.sum().backward()
    #         nn.utils.clip_grad.clip_grad_norm_(
    #             sr_model.parameters(),
    #             max_norm=0.25
    #         )
    #         optimizers.step()

    #         losses.append(loss.sum())

    #     avg_loss = sum(losses) / len(losses)
    #     print(f'\nepoch {epoch + 1}/{configs.num_epochs} | loss {avg_loss}')

    with torch.no_grad():
        sr_model.eval()
        psnr = []
        ssim = []
        for dl in [infer_easy_dl, infer_medium_dl, infer_hard_dl]:
            for eid, ebatch in enumerate(dl):
                labels, hr_imgs, lr_imgs = ebatch

                hr_imgs = hr_imgs.to(main_device)
                lr_imgs = lr_imgs.to(main_device)

                sr_imgs = sr_model(lr_imgs)

                psnr.append(psnr_fn(sr_imgs[:, :3, :, :], hr_imgs[:, :3, :, :]))
                ssim.append(ssim_fn(sr_imgs[:, :3, :, :], hr_imgs[:, :3, :, :]))

            avg_psnr = sum(psnr) / len(psnr)
            avg_ssim = sum(ssim) / len(ssim)
            print(f'evaluate epoch {eid} avg_psnr {avg_psnr} / {best_psnr}, avg_ssim {avg_ssim}')

            # if avg_psnr > best_psnr:
            #     print(f'psnr is best')
            #     best_psnr = avg_psnr
            #     save_file = f'{save_dir}/best_psnr_sr_model.pth'
            #     torch.save(sr_model.state_dict(), save_file)

            # if (epoch + 1) % 10 == 0:
            #     save_file = f'{save_dir}/epoch_{epoch}_sr_model.pth'
            #     torch.save(sr_model.state_dict(), save_file)


if __name__ == '__main__':
    main()
