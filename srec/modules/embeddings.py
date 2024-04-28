import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEmbed(nn.Module):
    """use sin positional encoding"""

    def __init__(
            self,
            dim: int,
            max_seq_length: int,
            device: str = "cuda:0"
    ) -> None:
        super(PositionalEmbed, self).__init__()

        assert dim % 2 == 0

        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = torch.tensor(10000., dtype=torch.float)
        self.device = torch.device(device)

        pos_encode = torch.zeros(self.max_seq_length, self.dim, dtype=torch.float)
        pos = torch.arange(0, self.max_seq_length, dtype=torch.float).unsqueeze(1)
        item = pos * torch.exp(-torch.arange(0, dim, 2, dtype=torch.float) * torch.log(self.base) / self.dim)
        pos_encode[:, 0::2] = torch.sin(item)
        pos_encode[:, 1::2] = torch.cos(item)
        
        pos_encode.unsqueeze_(0)

        # just register buffer size
        self.register_buffer("pos_encode", pos_encode)

        for m in self.modules():
            m = m.to(device=self.device)

    def forward(self, length: int) -> Tensor:
        return self.pos_encode[:length]


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class TimeEmbed(nn.Module):

    def __init__(self, time_embed_dims: int, timesteps: int, device: str = "cuda:0") -> None:
        super(TimeEmbed, self).__init__()

        self.pos_embed = nn.Embedding(num_embeddings=timesteps, embedding_dim=time_embed_dims).to(device=device)

        self.embed = nn.Sequential(
            self.pos_embed,
            nn.Linear(time_embed_dims, time_embed_dims),
            nn.GELU(),
            nn.Linear(time_embed_dims, time_embed_dims)
        ).to(device=device)

    def forward(self, t):
        return self.embed(t)
