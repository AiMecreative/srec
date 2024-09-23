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
from model.modules.recognizer_module import CRNNBased
from model.modules.sr_branch import SRBranch, SRWrapper
from model.losses import MultitaskLoss
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from model.utils.tokenizer import CTCTokenizer
from torchvision.utils import save_image
from torchvision.transforms import transforms, InterpolationMode
from expr.ablation.tsrn import TSRN



def accuracy(preds, targets):
    acc = 0
    for idx in range(len(preds)):
        if preds[idx] == targets[idx]:
            acc += 1
    return acc / len(preds)


@hydra.main(config_path='expr/crnn_mlp', config_name='crnn_mlp', version_base='v1.2')
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

    # attributes
    save_dir = configs.save_dir
    infer_dir = configs.infer_dir
    tokenizer = CTCTokenizer(data_conf.charset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(infer_dir):
        os.mkdir(infer_dir)

    # configs models
    rec_model = CRNNBased(
        mlp_classifier=True,
        side_rnn=False,
    )
    tsrn = TSRN(STN=True)
    # sr_model = SRBranch()

    # n_gpu = len(configs.device_ids)
    # ddp = n_gpu > 1
    main_device = configs.device

    # if ddp:
    #     rec_model = nn.DataParallel(rec_model)
    #     sr_model = nn.DataParallel(sr_model)
    # else:
    #     sr_model = SRWrapper(sr_model)
    #     rec_model = RecWrapper(rec_model)
    rec_model = rec_model.to(main_device)
    tsrn = tsrn.to(main_device)
    # sr_model = sr_model.to(main_device)

    # if 'sr' in configs.pretrain_branch:
    #     sr_model.load_pretrain(configs.sr_pretrain, origin=False, map_location=main_device)
    # if 'rec' in configs.pretrain_branch:
    rec_model.load_pretrain(configs.pretrain_weights, origin=False, map_location=main_device)
    tsrn.load_state_dict(torch.load('pretrain_weights/pretrain_sr.pth')['state_dict_G'])
    for p in tsrn.parameters():
        p.requires_grad_(False)
    tsrn.eval()

    # frozen_brach = configs.frozen_branch
    # if 'sr' in frozen_brach:
    #     for p in sr_model.parameters():
    #         p.requires_grad_(False)
    # if 'rec' in frozen_brach:
    #     for p in rec_model.parameters():
    #         p.requires_grad_(False)

    # # finetune recognizer
    # if 'side_rnn' in configs.finetune_rec_type:
    #     for p in rec_model.side_rnn.parameters():
    #         p.requires_grad_(True)

    # if 'mlp' in configs.finetune_rec_type:
    #     for p in rec_model.mlp.parameters():
    #         p.requires_grad_(True)

    # if 'dense' in configs.finetune_rec_type:
    #     for p in rec_model.dense.parameters():
    #         p.requires_grad_(True)

    # if 'all' in configs.finetune_rec_type:
    #     for p in rec_model():
    #         p.requires_grad_(True)

    # configs optim
    optimizers = optim.RMSprop(
        rec_model.parameters(), lr=5e-4
    )

    # configs loss module
    loss_type = ['rec']
    multi_loss = MultitaskLoss(
        init_weights=[1],
        loss_types=loss_type,
        learn_weights=False
    )

    multi_loss = multi_loss.to(main_device)

    print(f'init loss functions')

    # configs metrics
    acc_fn = accuracy
    # psnr_fn = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(main_device)
    # ssim_fn = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(main_device)
    def normalize_fn(x, tgt_size):
        resize = transforms.Resize(tgt_size, InterpolationMode.NEAREST)
        x = resize(x)
        x = x * 2 - 1
        return x

    def _weighted_avg(stat_list):
        avg = 0
        for s, w in zip(stat_list, [0.3699, 0.3227, 0.3074]):
            avg += w * float(s)
        return avg
    train_len = len(train_dl)
    best_acc = 0
    best_acc_id = 0
    # for epoch in range(configs.num_epochs):
    #     # sr_model.eval()
    #     rec_model.train()
    #     losses = []
    #     for tid, tbatch in enumerate(train_dl):
    #         print(f'\repoch {epoch + 1}/{configs.num_epochs} | tid {tid + 1}/{train_len}', end=' ')
    #         labels, hr_imgs, lr_imgs = tbatch

    #         tokens, token_lengths = tokenizer.tokenize(labels=labels, device=main_device)
    #         hr_imgs = hr_imgs.to(main_device)
    #         lr_imgs = lr_imgs.to(main_device)

    #         sr_imgs = tsrn(lr_imgs)

    #         sr_imgs = normalize_fn(sr_imgs,(hr_imgs.shape[2],hr_imgs.shape[3]))
    #         sr_imgs = sr_imgs[:, :3, :, :].mean(dim=1, keepdim=True)
    #         # lr_imgs = normalize_fn(lr_imgs, (hr_imgs.shape[2],hr_imgs.shape[3]))
    #         # lr_imgs = lr_imgs[:, :3, :, :].mean(dim=1, keepdim=True)

    #         logits = rec_model(sr_imgs)

    #         # loss = multi_loss(sr_imgs=sr_imgs, hr_imgs=hr_imgs).sum()
    #         logit_lengths = torch.full(
    #             (logits.shape[1],),
    #             fill_value=logits.shape[0],
    #             dtype=torch.long,
    #             device=logits.device
    #         )
    #         loss, _, _ = multi_loss(
    #             preds=logits,
    #             pred_lengths=logit_lengths,
    #             targets=tokens,
    #             target_lengths=token_lengths
    #         )

    #         # optimizers['sr'].zero_grad()
    #         optimizers.zero_grad()
    #         loss.backward()
    #         nn.utils.clip_grad.clip_grad_norm_(
    #             rec_model.parameters(),
    #             max_norm=5
    #         )
    #         # optimizers['sr'].step()
    #         optimizers.step()

    #         losses.append(loss)

    #     avg_loss = sum(losses) / len(losses)
    #     print(f'\nepoch {epoch + 1}/{configs.num_epochs} | loss {avg_loss}')

    with torch.no_grad():
        print(f'evaluating ...')
        # sr_model.eval()
        rec_model.eval()
        avg_acc = []
        for idx, dl in enumerate([infer_easy_dl, infer_medium_dl, infer_hard_dl]):
            acc = []
            for eid, ebatch in enumerate(dl):
                labels, hr_imgs, lr_imgs = ebatch

                tokens, token_lengths = tokenizer.tokenize(labels=labels, device=main_device)
                hr_imgs = hr_imgs.to(main_device)
                lr_imgs = lr_imgs.to(main_device)

                sr_imgs = tsrn(lr_imgs)
                sr_imgs = normalize_fn(sr_imgs, (hr_imgs.shape[2],hr_imgs.shape[3]))
                sr_imgs = sr_imgs[:, :3, :, :].mean(dim=1, keepdim=True)
                # lr_imgs = normalize_fn(lr_imgs, (hr_imgs.shape[2],hr_imgs.shape[3]))
                # lr_imgs = lr_imgs[:, :3, :, :].mean(dim=1, keepdim=True)
                logits = rec_model(sr_imgs)

                pred_tokens = tokenizer.decode_logits(logits)
                pred_labels = tokenizer.untokenize(pred_tokens)

                acc.append(acc_fn(pred_labels, labels))

            acc = sum(acc) / len(acc)
            avg_acc.append(acc)
            print(f'epoch {0}, loader {idx}, avg_acc {acc}')
        avg_acc = _weighted_avg(avg_acc)
        print(f'epoch {0}, avg_acc {avg_acc} / {best_acc} ({best_acc_id})')

            # if avg_acc > best_acc:
            #     print(f'best acc model')
            #     best_acc = avg_acc
            #     best_acc_id = epoch + 1
            #     save_file = f'{save_dir}/best_acc_rec_model.pth'
            #     torch.save(rec_model.state_dict(), save_file)
            # save_file = f'{save_dir}/epoch_{epoch}.pth'
            # torch.save(rec_model.state_dict(), save_file)


if __name__ == '__main__':
    main()
