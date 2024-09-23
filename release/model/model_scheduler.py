import os
import time
import colorama
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from torch.utils.data import DataLoader
from thop import profile
from torch import Tensor
from torchvision.utils import save_image
from datetime import datetime
from model.model_wrapper import ModelWrapper
from model.psrec_model import PSRec
from model.utils.tokenizer import CTCTokenizer
from model.losses import MultitaskLoss
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import transforms, InterpolationMode

from tensorboardX import SummaryWriter


class ModelScheduler(object):

    TEXTZOOM_WEIGHTS = [0.3699, 0.3227, 0.3074]

    def __init__(
        self,
        configs
    ) -> None:

        # attributes configs
        self.device = configs.device
        self.device_ids = configs.device_ids
        self.n_gpu = len(self.device_ids)
        self.ddp = self.n_gpu > 1
        self.num_epochs = configs.num_epochs
        self.num_evaluate = configs.num_evaluate
        self.loss_types = configs.loss.loss_types
        self.learn_weights = configs.loss.learn_weights

        self.infer_dir = configs.infer_dir
        if not os.path.exists(self.infer_dir):
            os.mkdir(self.infer_dir)

        self.save_dir = configs.save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.tensorboard_dir = configs.tensorboard_dir
        if not os.path.exists(self.tensorboard_dir):
            os.mkdir(self.tensorboard_dir)

        self.figs_dir = configs.figs_dir
        if not os.path.exists(self.figs_dir):
            os.mkdir(self.figs_dir)

        self.save_per_epochs = configs.save_per_epochs
        self.requires_augment = configs.psrec.requires_augment

        self.tokenizer = CTCTokenizer(configs.data.charset)

        # model configs
        self.model = PSRec(configs.psrec)
        if self.ddp:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        else:
            self.model = ModelWrapper(self.model)
        self.model = self.model.to(self.device)

        if not self.requires_augment:
            for p in self.model.module.augment.parameters():
                p.requires_grad_(False)

        print(
            f'model initialized, device {self.device}, ' +
            f'DDP {self.ddp}, ' +
            f'requires augment {self.requires_augment}'
        )

        # loss configs
        self.loss_module = self.init_loss(configs.loss)
        if self.ddp:
            self.loss_module = nn.DataParallel(self.loss_module)
        else:
            self.loss_module = ModelWrapper(self.loss_module)
        self.loss_module = self.loss_module.to(self.device)

        # optim configs
        self.optim = self.init_optim(configs.optim)
        print(f'optim initialized {self.optim}, configs {configs.optim}')
        self.lr_scheduler = self.init_lr_scheduler(configs.lr_scheduler)

        self.psnr_fn = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(self.device)
        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(self.device)

        def _accuracy(preds, targets):
            acc = 0
            for idx in range(len(preds)):
                if preds[idx] == targets[idx]:
                    acc += 1
            return acc / len(preds)
        self.acc_fn = _accuracy

        self.tensorboard_working_dir = f'{self.tensorboard_dir}/{datetime.now()}'
        self.writer = SummaryWriter(self.tensorboard_working_dir)

    def freeze_model(self, configs):
        # frozen modules if needed
        if 'encoder' in configs.frozen_modules:
            for p in self.model.module.encoder.parameters():
                p.requires_grad_(False)
        if 'augment' in configs.frozen_modules:
            for p in self.model.module.augment.parameters():
                p.requires_grad_(False)
        if 'sr' in configs.frozen_modules:
            for p in self.model.module.sr.parameters():
                p.requires_grad_(False)
        if 'recognizer' in configs.frozen_modules:
            for p in self.model.module.recognizer.parameters():
                p.requires_grad_(False)

        # recognizer finetune stratage
        # attention: use `module` find named_modules
        if 'side_rnn' in configs.finetune_rec_type:
            for p in self.model.module.recognizer.module.side_rnn.parameters():
                p.requires_grad_(True)

        if 'mlp' in configs.finetune_rec_type:
            for p in self.model.module.recognizer.module.mlp.parameters():
                p.requires_grad_(True)

        if 'head' in configs.finetune_rec_type:
            for p in self.model.module.recognizer.module.dense.parameters():
                p.requires_grad_(True)

        if 'all' in configs.finetune_rec_type:
            for p in self.model.module.recognizer.parameters():
                p.requires_grad_(True)

    def load_model(self, file: str, map_location: str = None, strict: bool = True, use_wrapper: bool = False, key: str = None):
        if use_wrapper:
            self.model.load_pretrain(file, map_location, strict, from_ddp=use_wrapper, key='model')
        else:
            state_dicts = torch.load(file, map_location)
            if key is not None and key != '':
                state_dicts = state_dicts[key]
            self.model.load_state_dict(state_dicts, strict=strict)

    def init_optim(self, configs):
        # params = self.model.parameters()
        # if self.learn_weights:
        #     params = itertools.chain(params, self.loss_module.parameters())
        optim_list = [optim.AdamW(
            self.model.parameters(),
            lr=float(configs.lr),
            betas=configs.betas
        )]
        if self.learn_weights:
            optim_list.append(optim.AdamW(
                self.loss_module.parameters(),
                lr=float(configs.loss_lr),
                # betas=configs.betas
            ))
        return optim_list

    def init_lr_scheduler(self, configs):
        lr_list = [optim.lr_scheduler.StepLR(
            self.optim[0],
            configs.step_size,
            configs.gamma
        )]
        if self.learn_weights:
            # step LR with gamma = 0.8 before
            # lr_list.append(optim.lr_scheduler.CosineAnnealingLR(self.optim[1], configs.step_size // 2, eta_min=1e-5))
            lr_list.append(optim.lr_scheduler.StepLR(
                self.optim[1],
                (configs.step_size * 1.5),
                configs.gamma
            ))
        return lr_list

    def init_loss(self, configs):
        return MultitaskLoss(
            configs.init_weights,
            configs.learn_weights,
            configs.loss_types,
            configs.weight_decay,
            configs.sr_loss_weights,
            configs.sr_requires_gp,
            configs.rec_blank_id
        )

    def _extract_batch(self, batch):
        labels, hr_imgs, lr_imgs = batch
        tokens, lengths = self.tokenizer.tokenize(labels, self.device)
        tokens = tokens.to(self.device)
        hr_imgs = hr_imgs.to(self.device)
        lr_imgs = lr_imgs.to(self.device)
        return labels, tokens, lengths, hr_imgs, lr_imgs

    @torch.no_grad()
    def save(self, file):
        torch.save({
            'encoder': self.model.module.encoder.state_dict(),
            'augment': self.model.module.augment.state_dict(),
            'recognizer': self.model.module.recognizer.state_dict(),
            'sr': self.model.module.sr.state_dict(),
            'model': self.model.state_dict()
        }, file)

    @torch.inference_mode()
    def _eval_step(self, batch):
        self.model.eval()
        labels, tokens, lengths, hr_imgs, lr_imgs = self._extract_batch(batch)
        torch.cuda.synchronize()
        st = time.time()
        logits, srs = self.model.module.evaluate(hr_imgs, lr_imgs, self.requires_augment)
        torch.cuda.synchronize()
        et = time.time()
        pred_tokens = self.tokenizer.decode_logits(logits)
        pred_labels = self.tokenizer.untokenize(pred_tokens)
        return labels, pred_labels, tokens, pred_tokens, hr_imgs, lr_imgs, srs, et - st

    @torch.inference_mode()
    def infer(self, infer_dl: DataLoader, loader_name=['easy', 'medium', 'hard']):
        label_file = f'{self.infer_dir}/a_labels.txt'
        img_name_fmt = 'lrsrhr_{}'
        lr_fmt = 'lr_{}.png'
        hr_fmt = 'hr_{}.png'
        sr_fmt = 'sr_{}.png'
        resize_to_sr = transforms.Resize((32, 128), InterpolationMode.NEAREST)
        if not isinstance(infer_dl, list):
            infer_dl = [infer_dl]
        with open(label_file, 'w') as pf:
            for id, dl in enumerate(infer_dl):
                level = loader_name[id]
                print(f'infer loader {level}')
                for idx, piece in enumerate(dl):
                    print(f'\r{idx+1}/{len(dl)}', end='', flush=True)
                    labels, pred_labels, tokens, pred_tokens, hr_imgs, lr_imgs, sr_imgs, infer_time = self._eval_step(piece)
                    img_name = f'{level}_{img_name_fmt.format(idx)}'
                    lr_file = f'{level}_{lr_fmt.format(idx)}'
                    hr_file = f'{level}_{hr_fmt.format(idx)}'
                    sr_file = f'{level}_{sr_fmt.format(idx)}'
                    lr_imgs = resize_to_sr(lr_imgs)
                    save_image(lr_imgs[0, :3, :, :], f'{self.infer_dir}/{lr_file}')
                    save_image(sr_imgs[0, :3, :, :], f'{self.infer_dir}/{sr_file}')
                    save_image(hr_imgs[0, :3, :, :], f'{self.infer_dir}/{hr_file}')
                    line = f'{img_name} | gt: {labels} | '
                    labels = labels[0]
                    pred_labels = pred_labels[0]
                    if labels != pred_labels:
                        pred_labels = f'{pred_labels} {"<"*25} error'
                    line += f'{pred_labels}\n'
                    pf.write(f'{img_name} | gt: {labels} | pred: {pred_labels}\n')

    @ torch.inference_mode()
    def evaluate(self, eval_dl):
        self.model.eval()
        avg_acc = []
        avg_psnr = []
        avg_ssim = []
        total_time = []

        def _weighted_avg(stat_list):
            avg = 0
            for s, w in zip(stat_list, self.TEXTZOOM_WEIGHTS):
                avg += w * float(s)
            return avg
        resize_to_sr = transforms.Resize((32, 128), InterpolationMode.NEAREST)
        # with open(f'{self.infer_dir}/a_predict.txt', 'w') as pf:
        if not isinstance(eval_dl, list):
            eval_dl = [eval_dl]
        for lid, loader in enumerate(eval_dl):
            print(f'\nevaluating loader {lid}')
            acc = []
            psnr = []
            ssim = []
            for eid, batch in enumerate(loader):
                if eid >= self.num_evaluate:
                    break
                print(f'\r>> {eid+1}/{self.num_evaluate}', end='', flush=True)
                label, pred_label, token, pred_token, hr_img, lr_img, sr_img, infer_time = self._eval_step(batch)
                acc.append(self.acc_fn(pred_label, label))
                total_time.append(infer_time)
                lr_img = resize_to_sr(lr_img)
                # save_image(
                #     [lr_img[0, :3, :, :],
                #      sr_img[0, :3, :, :],
                #      hr_img[0, :3, :, :]],
                #     f'{self.infer_dir}/image_lr_sr_hr_{eid}.png')
                # pf.write(f'image_lr_sr_hr_{eid}.png | gt: {label} | pred: {pred_label}\n')

                psnr.append(self.psnr_fn(sr_img[:, :3, :, :], hr_img[:, :3, :, :]))
                ssim.append(self.ssim_fn(sr_img[:, :3, :, :], hr_img[:, :3, :, :]))
            acc = sum(acc) / len(acc)
            psnr = sum(psnr) / len(psnr)
            ssim = sum(ssim) / len(ssim)

            print(f'\nloader {lid}: acc {acc} | psnr {psnr} | ssim {ssim}')

            avg_acc.append(acc)
            avg_psnr.append(psnr)
            avg_ssim.append(ssim)

        avg_acc = _weighted_avg(avg_acc)
        avg_psnr = _weighted_avg(avg_psnr)
        avg_ssim = _weighted_avg(avg_ssim)
        fps = len(total_time) / sum(total_time)
        return avg_acc, avg_psnr, avg_ssim, fps

    def _train_step(self, batch):
        self.model.train()
        labels, tokens, lengths, hr_imgs, lr_imgs = self._extract_batch(batch)
        pred_logits, sr_imgs = self.model(hr_imgs, lr_imgs, self.requires_augment)
        logits, logit_lengths = pred_logits[0], pred_logits[1]
        return self.loss_module.module(
            sr_imgs=sr_imgs,
            hr_imgs=hr_imgs,
            preds=logits,
            pred_lengths=logit_lengths,
            targets=tokens,
            target_lengths=lengths
        )

    def train(self, train_dl, eval_dl=None):
        print(f'train start')
        requires_eval = eval_dl is not None
        train_df = {'loss': []}
        for t in self.loss_types:
            train_df[f'{t}_loss'] = []
            train_df[f'{t}_weight'] = []
        if requires_eval:
            eval_df = {'acc': [], 'psnr': [], 'ssim': []}
        loader_len = len(train_dl)
        best_acc, best_acc_id = 0, 0
        best_ssim, best_ssim_id = 0, 0
        best_psnr, best_psnr_id = 0, 0
        for epoch_id in range(self.num_epochs):
            avg_loss = []
            avg_losses = {t: [] for t in self.loss_types}
            avg_weights = {t: [] for t in self.loss_types}
            for tid, batch in enumerate(train_dl):
                print(
                    f'\r[{datetime.now()}] ' +
                    f'epoch {epoch_id + 1}/{self.num_epochs} | ' +
                    f'tid {tid + 1}/{loader_len}',
                    end='', flush=True
                )
                # loss: ddp tensor
                # losses: dict
                # weights: dict
                loss, losses, weights = self._train_step(batch)
                loss = loss.sum()

                for optimizer in self.optim:
                    optimizer.zero_grad()
                nn.utils.clip_grad_norm_(self.model.module.recognizer.parameters(), max_norm=5)
                nn.utils.clip_grad_norm_(self.model.module.encoder.parameters(), max_norm=0.25)
                if self.requires_augment:
                    nn.utils.clip_grad_norm_(self.model.module.augment.parameters(), max_norm=0.25)
                nn.utils.clip_grad_norm_(self.model.module.sr.parameters(), max_norm=0.25)
                loss.backward()
                for optimizer in self.optim:
                    optimizer.step()

                avg_loss.append(loss.cpu().item())
                for t in self.loss_types:
                    avg_losses[t].append(losses[t].cpu().item())
                    avg_weights[t].append(weights[t].cpu().item())

            for lr_scheduler in self.lr_scheduler:
                lr_scheduler.step()
            avg_loss = sum(avg_loss) / len(avg_loss)
            train_df['loss'].append(avg_loss)
            for t in self.loss_types:
                avg_losses[t] = sum(avg_losses[t]) / len(avg_losses[t])
                avg_weights[t] = sum(avg_weights[t]) / len(avg_weights[t])
                train_df[f'{t}_loss'].append(avg_losses[t])
                train_df[f'{t}_weight'].append(avg_weights[t])

            print(f'\nloss: {avg_loss}')
            for t in self.loss_types:
                print(
                    f'{t} loss {avg_losses[t]} | ' +
                    f'{t} weights: {avg_weights[t]}'
                )

            # tensorborad record
            step = epoch_id + 1
            self.writer.add_scalar(tag='train/loss', scalar_value=avg_loss, global_step=step)
            self.writer.add_scalars(main_tag='train/losses', tag_scalar_dict=avg_losses, global_step=step)
            self.writer.add_scalars(main_tag='train/weights', tag_scalar_dict=avg_weights, global_step=step)

            # evaluate
            if requires_eval:
                print('>> start evaluating ...')
                acc, psnr, ssim, fps = self.evaluate(eval_dl)
                self.writer.add_scalar(tag='eval/acc', scalar_value=acc, global_step=step)
                self.writer.add_scalar(tag='eval/psnr', scalar_value=psnr, global_step=step)
                self.writer.add_scalar(tag='eval/ssim', scalar_value=ssim, global_step=step)
                eval_df['acc'].append(acc)
                eval_df['psnr'].append(psnr)
                eval_df['ssim'].append(ssim)
                print(
                    f'\n>> acc: {acc} / {best_acc} ({best_acc_id}) \n' +
                    f'psnr {psnr} / {best_psnr} ({best_psnr_id}) \n' +
                    f'ssim {ssim} / {best_ssim} ({best_ssim_id}) \n' +
                    f'fps {fps}) \n' +
                    '>> evaluating end'
                )
                if acc > best_acc:
                    print('>> current model is the best acc model')
                    best_acc = acc
                    best_acc_id = step
                    best_model = f'{self.save_dir}/best_acc_model.pth'
                    self.save(best_model)
                    print(f'>> model saved in {best_model}')
                if psnr > best_psnr:
                    print('>> current model is the best psnr model')
                    best_psnr = psnr
                    best_psnr_id = step
                    best_model = f'{self.save_dir}/best_psnr_model.pth'
                    self.save(best_model)
                    print(f'>> model saved in {best_model}')
                if ssim > best_ssim:
                    best_ssim = ssim
                    best_ssim_id = step
            if step % self.save_per_epochs == 0:
                save_file = f'{self.save_dir}/epoch_{step}.pth'
                self.save(save_file)
                print(f'>> model saved in {save_file}')
            # record information of this epoch
            with open(f'{self.tensorboard_working_dir}/train.txt', 'a') as train_log:
                train_log.write(f'loss {train_df["loss"][-1]} ')
                for t in self.loss_types:
                    train_log.write(f'{t}_loss {train_df[f"{t}_loss"][-1]} ')
                    train_log.write(f'{t}_weight {train_df[f"{t}_weight"][-1]} ')
                train_log.write('\n')
            with open(f'{self.tensorboard_working_dir}/eval.txt', 'a') as eval_log:
                eval_log.write(f'acc {eval_df["acc"][-1]} ')
                eval_log.write(f'psnr {eval_df[f"psnr"][-1]} ')
                eval_log.write(f'ssim {eval_df[f"ssim"][-1]} ')
                eval_log.write('\n')
        # write all log into csv
        train_df = pd.DataFrame(train_df)
        eval_df = pd.DataFrame(eval_df)
        train_df.to_csv(f'{self.tensorboard_working_dir}/train.csv')
        eval_df.to_csv(f'{self.tensorboard_working_dir}/eval.csv')
        print(f'train ending')
