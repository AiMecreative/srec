import torch
import torch.nn as nn
from torch import Tensor, Size
from torch.nn.modules import transformer
from typing import Optional
from srec.utils.utils import InitDevice


class RecLayer(nn.Module):

    def __init__(
        self,
        dim_models: int,
        num_heads: int,
        dim_feedforward: int,
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            dim_models,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim_models,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.linear1 = nn.Linear(dim_models, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_models)

        self.norm1 = nn.LayerNorm(dim_models, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim_models, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(dim_models, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(dim_models, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def _forward_step(
        self,
        token: Tensor,
        normed_token: Tensor,
        key: Tensor,
        value: Tensor,
        token_mask: Optional[Tensor],
        pad_mask: Optional[Tensor]
    ):
        attn_token, sa_weights = self.self_attn(
            normed_token,
            key,
            key,
            attn_mask=token_mask,
            key_padding_mask=pad_mask
        )
        token = token + self.dropout1(attn_token)

        attn_token, ca_weights = self.cross_attn(
            self.norm1(token),
            value,
            value
        )
        token = token + self.dropout2(attn_token)

        attn_token = self.linear1(self.norm2(token))
        attn_token = self.activation(attn_token)
        attn_token = self.linear2(self.dropout(attn_token))
        token = token + self.dropout3(attn_token)

        return token, sa_weights, ca_weights

    def forward(
        self,
        pos_query,
        token,
        img_features,
        pos_query_mask: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        token_pad_mask: Optional[Tensor] = None,
        update_token: bool = True
    ):
        # NOTE LayerNorm before forward, Pre-LN
        # NOTE pre-process of content and position query
        normed_query = self.norm_q(pos_query)
        normed_token = self.norm_c(token)

        pos_query = self._forward_step(
            pos_query,
            normed_query,
            normed_token,
            img_features,
            pos_query_mask,
            token_pad_mask
        )[0]

        if update_token:
            token = self._forward_step(
                token,
                normed_token,
                normed_token,
                img_features,
                token_mask,
                token_pad_mask
            )[0]
        return pos_query, token


class RecBlock(nn.Module):

    def __init__(
        self,
        dim_models: int,
        num_heads: int,
        dim_feedforward: int,
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.layers = transformer._get_clones(RecLayer(
            dim_models,
            num_heads,
            dim_feedforward,
            layer_norm_eps,
            dropout,
        ), 1)

        self.norm = nn.LayerNorm(dim_models)

    def forward(
        self,
        pos_query,
        token,
        img_features,
        pos_query_mask: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        token_pad_mask: Optional[Tensor] = None,
    ):
        img_features = img_features.flatten(2).permute(0, 2, 1)
        for idx, layer in enumerate(self.layers):
            is_last = idx == len(self.layers) - 1
            pos_query, token = layer(
                pos_query,
                token,
                img_features,
                pos_query_mask,
                token_mask,
                token_pad_mask,
                update_token=not is_last
            )
        pos_query = self.norm(pos_query)
        return pos_query
