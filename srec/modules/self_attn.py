import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Callable


class SelfAttn(nn.Module):

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        attn_fn: Callable,
        device: str,
        dropout: float = 0.01,
    ) -> None:
        super(SelfAttn, self).__init__()

        self.__num_heads = num_heads
        self.__attn_scale = dim_model ** (-0.5)
        self.__attn_fn = attn_fn
        self.__dropout = dropout

        self.__projection = nn.Linear(dim_model, dim_model * 3)

        self.__out_projection = nn.Linear(dim_model, dim_model)

        for m in self.modules():
            m = m.to(device)

    def forward(self, x: Tensor, mask=None):
        """
        input: 
            x:    [bs, seq_len, dim]
            mask: [seq_len, seq_len]
        output:
            attn: [bs, seq_len, dim]
        """
        masking = (mask is not None)
        seq_len, dim = x.shape[1], x.shape[2]
        qkv: Tensor = self.__projection(x)
        qkv = qkv.reshape(-1, seq_len, 3, self.__num_heads, dim // self.__num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.__attn_scale, qkv[1], qkv[2]
        attn: Tensor = self.__attn_fn(q, k, v, mask, masking=masking, dropout=self.__dropout, is_train=self.training)
        return self.__out_projection(attn)


class FullAttnFn:
    """
    same with multihead_attention in torch
    """

    def __call__(self, query, key, value, mask, masking: bool = False, dropout: float = 0.01, is_train: bool = False) -> Any:
        """
        input: 
            q,k,v: [bs, heads, seq_len, dim // heads]
            mask: [seq_len, seq_len]
        output: [bs, seq_len, dim]
        """
        seq_len = query.shape[-2]
        dim = query.shape[-1] * query.shape[1]
        attn = torch.matmul(query, torch.transpose(key, -2, -1))
        if masking:
            attn = attn + mask
        attn = torch.softmax(attn, -1)
        attn = torch.dropout(attn, dropout, is_train)
        attn = torch.matmul(attn, value).permute(0, 2, 1, 3).reshape(-1, seq_len, dim)
        return attn


class LinearAttnFn:
    """
    linear attention without mask
    """

    def __call__(self, query, key, value, mask, masking: bool = False, dropout: float = 0.01, is_train: bool = False):
        """
        input:
            q,k,v: [bs, heads, seq_len, dim // heads]
        output: [bs, seq_len, dim]
        """
        seq_len = query.shape[-2]
        dim = query.shape[-1] * query.shape[1]

        query: Tensor = torch.softmax(query, -2)
        key: Tensor = torch.softmax(key, -1)

        kv: Tensor = torch.einsum('bhkd, bhvd -> bhkv', key, value)
        attn: Tensor = torch.einsum('bhkv, bhkd -> bhvd', kv, query)

        attn = attn.permute(0, 2, 1, 3).reshape(-1, seq_len, dim)

        return attn
