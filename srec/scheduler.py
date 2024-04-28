import os
import torch.nn as nn
import torch.optim as optim
from hydra.utils import instantiate
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from srec.arch import EDDModule
from srec.models.srec_model import Srec
from srec.models.encoder import HREncoder
from srec.models.augment import Augment
from srec.models.recognizer import Recognizer
from srec.models.sr import SR
from srec.utils.utils import InitDevice
from srec.utils.metrics import Metrics
from srec.utils.tokenizer import CTCTokenizer
from tensorboardX import SummaryWriter


class SrecSchduler(object):

    def __init__(
        self,
        configs
    ) -> None:

        self.device = configs.task.device
        self.tensorboard_dir = configs.task.tensorboard_dir
        self.save_dir = configs.task.save_dir
        self.infer_dir = configs.task.infer_dir
        self.epochs = configs.task.epochs
        self.save_interval = configs.task.save_interval
        self.eval_interval = configs.task.eval_interval
        self.encoder_configs = instantiate(configs.encoder)
        self.augment_configs = instantiate(configs.augment)
        self.recognizer_configs = instantiate(configs.recognizer)
        self.sr_configs = instantiate(configs.sr)

        EDDModule.DEVICE = configs.task.device

        self.model = self.get_model(
            configs.task.load_pretrain,
            configs.task.pretrain_weights,
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(configs.optim.lr),
            betas=configs.optim.betas
        )

        self.tokenizer = CTCTokenizer(configs.data.charset)
        self.metric = Metrics(self.device)
        self.writer = SummaryWriter(self.tensorboard_dir)

        if not os.path.exists(self.tensorboard_dir):
            os.mkdir(self.tensorboard_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.exists(self.infer_dir):
            os.mkdir(self.infer_dir)

    def get_model(
        self,
        load_pretrain: bool = False,
        pretrain_weights: str = '',
    ):
        encoder = HREncoder(self.encoder_configs)
        augment = Augment(self.augment_configs)
        recognizer = Recognizer(self.recognizer_configs)
        sr = SR(self.sr_configs)
        return Srec(
            encoder,
            augment,
            recognizer,
            sr,
            load_pretrain,
            pretrain_weights,
        )

    def resume(self, state_dicts, freeze: bool = False):
        self.model.resume(state_dicts)
        if freeze:
            self.model.eval()
            self.model.requires_grad_(False)

    def save(self, state_dicts):
        self.model.save(state_dicts)

    def _extract(self, batch, raw_label: bool = False):
        labels, hr_imgs, lr_imgs = batch
        tokens, lengths = self.tokenizer.tokenize(labels, self.device)
        tokens = tokens.to(self.device)
        lengths = lengths.to(self.device)
        hr_imgs = hr_imgs.to(self.device)
        lr_imgs = lr_imgs.to(self.device)
        if raw_label:
            return labels, tokens, lengths, hr_imgs, lr_imgs
        return tokens, lengths, hr_imgs, lr_imgs

    def _eval_step(self, batch, requires_augment, vis: bool = False):
        labels, tokens, lengths, hr_imgs, lr_imgs = self._extract(batch, raw_label=True)
        pred_logits, pred_srs = self.model.evaluate(tokens, hr_imgs, lr_imgs, requires_augment)
        pred_labels = self.tokenizer.decode_logits(pred_logits)
        pred_labels = self.tokenizer.untokenize(pred_labels)
        if vis:
            print()
            print('=' * 20)
            for idx in range(1, 11):
                print(f'{labels[-idx]} > {pred_labels[-idx]} > {pred_logits[-idx]}')
            print('=' * 20)
        return {
            'acc': self.metric.acc(pred_labels, labels),
            'psnr': self.metric.psnr(pred_srs, hr_imgs),
            'ssim': self.metric.ssim(pred_srs, hr_imgs)
        }

    def evaluate(self, eval_dl: DataLoader, requires_augment: bool, max_num_eval: int = 25):
        acc = 0
        psnr = 0
        ssim = 0
        loader_len = len(eval_dl)
        num_eval = min(max_num_eval, loader_len)
        for eid, batch in enumerate(eval_dl):
            if eid >= num_eval:
                break
            print(f'\reval batch idx {eid + 1}/{loader_len}', end='', flush=True)
            vis = eid == num_eval - 1
            eval_res = self._eval_step(batch, requires_augment, vis)
            acc += eval_res['acc']
            psnr += eval_res['psnr']
            ssim += eval_res['ssim']
        return {
            'acc': acc / loader_len,
            'psnr': psnr / loader_len,
            'ssim': ssim / loader_len
        }

    def infer(
        self,
        infer_dl: DataLoader,
        requires_augment: bool = False,
        use_hr: bool = True
    ):
        image_dir = f'{self.infer_dir}/images'
        label_file = f'{self.infer_dir}/labels.txt'
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        global_idx = 0
        lf = open(label_file, 'a')
        loader_len = len(infer_dl)
        for bid, batch in enumerate(infer_dl):
            print(f'\rinfer batch {bid + 1}/{loader_len}', end='', flush=True)
            gt_labels, _, _, gt_hrs, gt_lrs = self._extract(batch, raw_label=True)
            bs = gt_hrs.shape[0]
            gt_hrs = gt_hrs.to(self.device)
            gt_lrs = gt_lrs.to(self.device)
            imgs = gt_hrs if use_hr else gt_lrs
            pred_labels, pred_srs = self.model.infer(imgs, requires_augment)
            pred_labels = self.tokenizer.decode_logits(pred_labels)
            pred_labels = self.tokenizer.untokenize(pred_labels)

            name_fmt = "{}_{:0>5d}.png"
            for idx in range(bs):
                global_idx += 1
                img_name = name_fmt.format('gt', global_idx)
                sr_name = name_fmt.format('sr', global_idx)
                lf.write(f'{img_name} \t | \t ground truth: {gt_labels[idx]} \t | prediction: \t {pred_labels[idx]}\n')
                save_image(imgs[idx], f'{image_dir}/{img_name}')
                save_image(pred_srs[idx], f'{image_dir}/{sr_name}')

        lf.close()
        print()

    def _train_step(self, batch, requires_augment):
        tokens, lengths, hr_imgs, lr_imgs = self._extract(batch)
        train_res = self.model(tokens, lengths, hr_imgs, lr_imgs, requires_augment)
        return train_res

    def train(self, train_dl, eval_dl=None, requires_augment: bool = False):
        requires_eval = (eval_dl is not None)
        global_step = 0
        best_acc = 0
        loader_len = len(train_dl)
        for epoch in range(self.epochs):
            bs = 0
            avg_loss = 0
            avg_losses = [0, 0]
            avg_weights = [0, 0]
            for tid, batch in enumerate(train_dl):
                self.model.train()
                bs += 1
                global_step += 1
                print(
                    f"\r[{datetime.now()}] " +
                    f"epoch {epoch + 1}/{self.epochs} | " +
                    f"tid {tid + 1}/{loader_len} | " +
                    f"global step {global_step}",
                    end="", flush=True
                )
                loss, losses, weights = self._train_step(batch, requires_augment)
                avg_loss += loss.detach().requires_grad_(False)
                avg_losses[0] += losses[0].detach().requires_grad_(False)
                avg_losses[1] += losses[1].detach().requires_grad_(False)
                avg_weights[0] += weights[0].detach().requires_grad_(False)
                avg_weights[1] += weights[1].detach().requires_grad_(False)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    5
                )
                self.optimizer.step()
            avg_loss /= bs
            avg_losses[0] /= bs
            avg_losses[1] /= bs
            avg_weights[0] /= bs
            avg_weights[1] /= bs

            print(f"\nloss: {avg_loss} | rec loss {avg_losses[0]} | sr loss {avg_losses[1]} | loss weights: {avg_weights}")

            # add to tensorboard
            self.writer.add_scalar(
                tag='train/loss',
                scalar_value=avg_loss,
                global_step=global_step
            )
            self.writer.add_scalars(
                main_tag='train/losses',
                tag_scalar_dict={'rec': avg_losses[0], 'sr': avg_losses[1]},
                global_step=global_step
            )
            self.writer.add_scalars(
                main_tag='train/weights',
                tag_scalar_dict={'rec': avg_weights[0], 'sr': avg_weights[1]},
                global_step=global_step
            )

            if requires_eval and (epoch + 1) % self.eval_interval == 0:
                print(f'eval model:')
                eval_res = self.evaluate(eval_dl, requires_augment)
                if best_acc < eval_res['acc']:
                    best_acc = eval_res['acc']
                    self.save(f'{self.save_dir}/best_model_epoch{epoch + 1}.pth')
                    print(f'save best model in {self.save_dir}/best_model_epoch{epoch + 1}.pth')
                print(f'>> acc: {eval_res["acc"]}\n' +
                      f'>> psnr: {eval_res["psnr"]}\n' +
                      f'>> ssim: {eval_res["ssim"]}')
                # add to tensorboard
                self.writer.add_scalar(
                    tag='eval/acc',
                    scalar_value=eval_res['acc'],
                    global_step=global_step
                )
                self.writer.add_scalar(
                    tag='eval/psnr',
                    scalar_value=eval_res['psnr'],
                    global_step=global_step
                )
                self.writer.add_scalar(
                    tag='eval/ssim',
                    scalar_value=eval_res['ssim'],
                    global_step=global_step
                )

            if (epoch + 1) % self.save_interval == 0:
                self.save(f'{self.save_dir}/epoch{epoch + 1}_statedicts.pth')

                print(f'save model in {self.save_dir}/epoch{epoch + 1}_statedicts.pth')
