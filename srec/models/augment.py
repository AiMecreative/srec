import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, Size
from typing import List, Callable, Tuple
from functools import reduce, partial
from srec.arch import EDDModule
from srec.modules.mini_res import MiniRes
from srec.modules.self_attn import SelfAttn, LinearAttnFn
from srec.modules.embeddings import TimeEmbed


def sample_param(param_sequence: np.ndarray, timestep: Tensor, x_shape: Size) -> Tensor:
    # """
    # used for diffusion parameter selection
    # """
    timestep_cpu = timestep.cpu()
    b, *_ = timestep_cpu.shape
    out = torch.from_numpy(param_sequence[timestep_cpu]).to(timestep.device).float()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class SREcoder(nn.Module):

    def __init__(
            self,
            in_channels: int,
            device: str,
            channel_mults: List[int] = [2],
            num_mini: int = 1,
            num_mids: int = 1,
            num_attn_heads: int = 3,
            timesteps: int = 2000,
            concat_hr_lr: bool = True,
            concat_type: str = "mix",  # "mix" | "concat"
            attn_fn: Callable = LinearAttnFn(),
    ) -> None:
        super(SREcoder, self).__init__()

        if concat_hr_lr:
            in_channels *= 2

        depth = len(channel_mults)
        time_embed_dim = in_channels * 4
        channels = [in_channels]
        channels.extend([in_channels * mult for mult in channel_mults])

        self.__time_embed = TimeEmbed(
            time_embed_dim,
            timesteps,
            device
        )

        self.__downs = nn.ModuleList([])
        self.__mids = nn.ModuleList([])
        self.__ups = nn.ModuleList([])

        for d in range(depth):
            layer = nn.ModuleList([])
            res = nn.ModuleList([
                MiniRes(
                    channels[d],
                    channels[d],
                    device,
                    time_embed_dim=time_embed_dim
                ) for _ in range(num_mini)
            ])
            layer.append(res)
            layer.append(
                SelfAttn(
                    channels[d],
                    num_attn_heads,
                    attn_fn=attn_fn,
                    device=device
                )
            )
            layer.append(
                self.down_sample(
                    channels[d],
                    channels[d + 1]
                )
            )

            self.__downs.append(layer)

        for d in range(num_mids):
            layer = nn.ModuleList([])
            res = nn.ModuleList([
                MiniRes(
                    channels[-1],
                    channels[-1],
                    device,
                    time_embed_dim=time_embed_dim
                ) for _ in range(num_mini)
            ])
            layer.append(res)
            layer.append(
                SelfAttn(
                    channels[-1],
                    num_attn_heads,
                    attn_fn=attn_fn,
                    device=device
                )
            )

            self.__mids.append(layer)

        for d in reversed(list(range(1, depth + 1))):
            layer = nn.ModuleList([])
            res = nn.ModuleList([
                MiniRes(
                    channels[d] + channels[d - 1],
                    channels[d] + channels[d - 1],
                    device,
                    time_embed_dim=time_embed_dim
                ) for _ in range(num_mini)
            ])
            layer.append(res)
            layer.append(
                SelfAttn(
                    channels[d] + channels[d - 1],
                    num_attn_heads,
                    attn_fn=attn_fn,
                    device=device
                )
            )
            layer.append(
                self.up_sample(
                    channels[d] + channels[d - 1],
                    channels[d - 1]
                )
            )

            self.__ups.append(layer)

        self.__final_res = MiniRes(
            channels[0] * 2,
            channels[0],
            device,
            time_embed_dim
        )

        self.__final_conv = nn.Conv2d(
            channels[0],
            channels[0],
            kernel_size=3,
            padding=1
        )

        self.__concat_fn = (
            self.mix_concat
            if concat_type == "mix"
            else partial(torch.concat, dim=1)
        )

        for m in self.modules():
            m = m.to(device)

    def down_sample(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * stride ** 2,
                kernel_size=3,
                padding=1,
                stride=stride
            ),
            nn.Conv2d(
                in_channels * stride ** 2,
                out_channels,
                kernel_size=1
            )
        )

    def up_sample(
        self,
        in_channels: int,
        out_channels: int,
        up_scale: int = 2
    ):
        return nn.Sequential(
            nn.Upsample(
                scale_factor=up_scale,
                mode="nearest"
            ),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def mix_concat(self, hr: Tensor, lr: Tensor):
        mix = torch.concat([hr, lr], dim=2)
        bs, ch, h, w = mix.shape
        mix = mix.reshape(bs, ch * 2, h // 2, w)
        return mix

    def forward(self, fh: Tensor, fl: Tensor, timestep: Tensor = None):
        mix = self.mix_concat(fh, fl)
        res_x = mix.clone()

        t_embed = self.__time_embed(timestep)

        features = []

        # down sample
        for res_layer, attn_block, down_block in self.__downs:
            for res_block in res_layer:
                mix = res_block(mix, t_embed)
            mix = attn_block(mix)
            mix = down_block(mix)
            features.append(mix)

        # mid sample
        for res_layer, attn_block in self.__mids:
            for res_block in res_layer:
                mix = res_block(mix, t_embed)
            mix = attn_block(mix)

        # up sample
        for res_layer, attn_block, up_block in self.__ups:
            for res_block in res_layer:
                mix = res_block(
                    self.__concat_fn(mix, features.pop()),
                    t_embed
                )
            mix = attn_block(mix)
            mix = up_block(mix)

        mix = self.__final_res(
            self.__concat_fn(mix, res_x),
            t_embed
        )

        mix = self.__final_conv(mix)

        return mix


class NoiseDiffusion(object):
    """predict noises"""

    def __init__(
        self,
        diffusion_timesteps: int = 2000,
    ) -> None:
        super(NoiseDiffusion, self).__init__()

        self.diffusion_timesteps = diffusion_timesteps
        self.alpha_cumprod_func = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        self.max_beta = 0.999

        # alpha bar t
        alpha_cumprod_t = []
        # alpha bar t+1
        alpha_cumprod_tp1 = []
        # beta
        betas = []
        for i in range(self.diffusion_timesteps):
            t1 = i / self.diffusion_timesteps
            t2 = (i + 1) / self.diffusion_timesteps
            alpha_cumprod_t.append(self.alpha_cumprod_func(t1))
            alpha_cumprod_tp1.append(self.alpha_cumprod_func(t2))
            betas.append(min(1 - alpha_cumprod_tp1[-1] / alpha_cumprod_t[-1], self.max_beta))

        self.alpha_cumprod_t = np.array(alpha_cumprod_t, dtype=np.float64)
        self.alpha_cumprod_tm1 = np.append(1., self.alpha_cumprod_t[:-1])
        self.alpha_cumprod_tp1 = np.array(alpha_cumprod_tp1, dtype=np.float64)
        self.betas = np.array(betas, dtype=np.float64)

        assert self.alpha_cumprod_t.shape == (self.diffusion_timesteps,)
        assert self.alpha_cumprod_tp1.shape == (self.diffusion_timesteps,)
        assert self.betas.shape == (self.diffusion_timesteps,)

        # diffusion q(x_t | x_{t-1})
        self.sqrt_alpha_cumprod_t = np.sqrt(self.alpha_cumprod_t)
        self.sqrt_1m_alpha_cumprod_t = np.sqrt(1. - self.alpha_cumprod_t)
        self.log_1m_alpha_cumprod_t = np.log(1. - self.alpha_cumprod_t)
        self.sqrt_recip_alpha_cumprod_t = np.sqrt(1. / self.alpha_cumprod_t)
        self.sqrt_recip_m1_alpha_cumprod_t = np.sqrt(1. / self.alpha_cumprod_t - 1.)

        # variance of posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1. - self.alpha_cumprod_tm1) / (1. - self.alpha_cumprod_t)
        )
        # clip log-variance for zero-variance at the beginning
        self.clipped_log_posterior_variance = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        # mean of posterior q(x_{t-1} | x_t, x_0)
        self.posterior_mean_x0_coef = (
            self.betas * np.sqrt(self.alpha_cumprod_tm1) / (1. - self.alpha_cumprod_t)
        )
        self.posterior_mean_xt_coef = (
            (1. - self.alpha_cumprod_tm1) * np.sqrt(1. - self.betas) / (1. - self.alpha_cumprod_t)
        )

    def get_xt_mean_variance_from_x0(self, x0: Tensor, timestep: Tensor) -> Tuple[Tensor]:
        """
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_1m_alpha_cumprod_tm1 * noise
        the shape of x0: [b, patch_embed_dim, hw/64]
        """
        mean = sample_param(self.sqrt_alpha_cumprod_t, timestep, x0.shape) * x0
        variance = sample_param(1. - self.alpha_cumprod_t, timestep, x0.shape)
        log_variance = sample_param(self.log_1m_alpha_cumprod_t, timestep, x0.shape)
        return mean, variance, log_variance

    def get_xt_from_x0(self, x0: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        # noise = torch.randn_like(x0)
        mean, _, log_var = self.get_xt_mean_variance_from_x0(x0, timestep)
        return mean + torch.exp(log_var * 0.5) * noise

    def get_posterior_mean_variance(self, x0: Tensor, xt: Tensor, timestep: Tensor) -> Tuple[Tensor]:
        """q(x_{t-1} | x_t, x_0)"""
        post_mean = (
            sample_param(self.posterior_mean_x0_coef, timestep, x0.shape) * x0
            + sample_param(self.posterior_mean_xt_coef, timestep, xt.shape) * xt
        )
        post_variance = sample_param(self.posterior_variance, timestep, xt.shape)
        clipped_post_variance = sample_param(self.clipped_log_posterior_variance, timestep, xt.shape)
        return post_mean, post_variance, clipped_post_variance

    # use denoise model to predict
    @torch.no_grad()
    def get_x0_from_noise(self, xt: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        """x_0 = 1. / sqrt{alpha_cumprod} * x_t - sqrt{1. / alpha_cumprod - 1.} * noise"""
        return (
            sample_param(self.sqrt_recip_alpha_cumprod_t, timestep, xt.shape) * xt
            - sample_param(self.sqrt_recip_m1_alpha_cumprod_t, timestep, noise.shape) * noise
        )

    @torch.no_grad()
    def get_pred_mean_variance(self, denoise_model: nn.Module, xt: Tensor, condition: Tensor, timestep: Tensor) -> Tuple[Tensor]:
        """denoise_model: p(x_{t-1} | x_t)"""
        pred_noise: Tensor = denoise_model(xt, condition, timestep)
        # pred_noise = pred_noise.reshape(pred_noise.shape[0], pred_noise.shape[1], -1)
        pred_x0: Tensor = self.get_x0_from_noise(xt=xt, timestep=timestep, noise=pred_noise)
        pred_mean, pred_var, pred_log_var = self.get_posterior_mean_variance(x0=pred_x0, xt=xt, timestep=timestep)
        return pred_mean, pred_var, pred_log_var

    @torch.no_grad()
    def get_pred_xtm1(self, denoise_model: nn.Module, xt: Tensor, condition: Tensor, timestep: Tensor) -> Tensor:
        """denoise_mode: x_{t-1} = mean + std * noise"""
        pred_mean, _, pred_log_var = self.get_pred_mean_variance(denoise_model, xt, condition, timestep)
        noise = torch.randn_like(xt)
        return pred_mean + torch.exp(pred_log_var * 0.5) * noise

    @torch.no_grad()
    def sample(self, denoise_model: nn.Module, condition: Tensor) -> Tensor:
        """sample single img"""
        img: Tensor = torch.randn_like(condition, device=denoise_model.device)
        for timestep in range(self.diffusion_timesteps - 1, 0, -1):
            timestep = torch.tensor([timestep,] * img.shape[0], dtype=torch.long, device=denoise_model.device)
            img = self.get_pred_xtm1(denoise_model, xt=img, condition=condition, timestep=timestep)
        return img


class Augment(EDDModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)