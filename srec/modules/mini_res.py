import torch.nn as nn
from torch import Tensor


class MiniRes(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        device: str,
        time_embed_dim: int = None,
        num_groups: int = 8
    ) -> None:
        super(MiniRes, self).__init__()

        self.__conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(
                num_groups,
                out_channels
            ),
        )

        self.__act1 = nn.ReLU()

        self.__conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(
                num_groups,
                out_channels
            ),
        )

        self.__act2 = nn.ReLU()

        self.__conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        self.__mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                time_embed_dim,
                out_channels * 2
            )
        ) if time_embed_dim is not None else None

        for m in self.modules():
            m = m.to(device)

    def forward(self, x: Tensor, time_embed: Tensor = None):
        scale_offset = None
        if time_embed is not None and self.__mlp is not None:
            time_embed = self.__mlp(time_embed)
            time_embed = time_embed.unsqueeze(-1).unsqueeze(-1)
            scale_offset = time_embed.chunk(chunks=2, dim=1)

        conv_x = self.__conv1(x)
        if scale_offset is not None:
            scale, offset = scale_offset
            conv_x = (scale + 1) * conv_x + offset
        conv_x = self.__act1(conv_x)

        conv_x = self.__conv2(conv_x)
        conv_x = self.__act2(conv_x)

        x = conv_x + self.__conv3(x)
        return x
