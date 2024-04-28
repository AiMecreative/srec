import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor, Size
from typing import Optional
from itertools import permutations
from srec.utils.tokenizer import CTCTokenizer
from srec.arch import EDDModule
from srec.modules.rec_layer import RecBlock
from srec.modules.embeddings import TokenEmbedding

_BOS_ID_ = 37
_PAD_ID_ = 38
_EOS_ID_ = 0


class ParseqBackbone(nn.Module):

    def __init__(
        self,
        charset_size: int,
        max_label_length: int,
        dim_models: int,
        num_heads: int,
        dim_feedforward: int,
        device: str,
        max_num_perm: int = 6,
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.1,
        pretrain: str = ''
    ) -> None:
        super().__init__()

        self._device = device
        self.max_num_perm = max_num_perm
        self.max_label_length = max_label_length
        self.rng = np.random.default_rng()
        self.eos_id = _EOS_ID_
        self.pad_id = _PAD_ID_
        self.bos_id = _BOS_ID_

        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, dim_models))

        self.decoder = RecBlock(
            dim_models,
            num_heads,
            dim_feedforward,
            layer_norm_eps,
            dropout,
        )

        self.head = nn.Linear(dim_models, charset_size - 2)
        self.text_embed = TokenEmbedding(charset_size, dim_models)
        self.dropout = nn.Dropout(dropout)

        # self._load_pretrain(pretrain)

    def _load_pretrain(self, checkpoints: str):
        state_dict = torch.load(checkpoints)['recognizer']
        self_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in self_dict}
        self_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict)

    def perms(self, token):
        max_num_chars = token.shape[1] - 2
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)]

        max_perms = math.factorial(max_num_chars)
        max_perms //= 2
        num_gen_perms = min(self.max_num_perm, max_perms)
        if max_num_chars < 5:
            if max_num_chars == 4:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(
                range(max_num_chars), max_num_chars)), device=self._device)[selector]
            perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend(
                [torch.randperm(max_num_chars, device=self._device)
                 for _ in range(num_gen_perms - len(perms))]
            )
            perms = torch.stack(perms)
        comp = perms.flip(-1)
        perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def attn_masks(self, perm):
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = -torch.inf
        token_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = -torch.inf  # mask "self"
        query_mask = mask[1:, :-1]
        return token_mask, query_mask

    def _decode_step(
        self,
        token: torch.Tensor,
        img_features: torch.Tensor,
        pos_query: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        pad_mask: Optional[Tensor] = None,
        pos_query_mask: Optional[Tensor] = None
    ):
        """
        在decoder的基础上进一步封装 添加了文本的position和token的结合处理
        """
        bs, length = token.shape
        bos = self.text_embed(token[:, :1])
        token_embed = self.pos_queries[:, :length - 1] + self.text_embed(token[:, 1:])
        token_embed = self.dropout(torch.cat([bos, token_embed], dim=1))
        if pos_query is None:
            pos_query = self.pos_queries[:, :length].expand(bs, -1, -1)
        pos_query = self.dropout(pos_query)
        return self.decoder(
            pos_query=pos_query,
            token=token_embed,
            img_features=img_features,
            pos_query_mask=pos_query_mask,
            token_mask=token_mask,
            token_pad_mask=pad_mask
        )

    @torch.inference_mode()
    def infer(
        self,
        img_features: Tensor,
        max_length: int = None,
    ):
        test_mode = max_length is None
        max_length = (
            self.max_label_length
            if max_length is None
            else min(max_length, self.max_label_length)
        )
        bs = img_features.shape[0]
        # +1 for eos at end of sequence.
        num_steps = max_length + 1
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)
        token_mask = query_mask = torch.triu(
            torch.full(
                (num_steps, num_steps),
                float('-inf'), device=self._device
            ),
            diagonal=1
        )
        # if self.decode_ar:
        in_token = torch.full((bs, num_steps), self.pad_id,
                              dtype=torch.long, device=self._device)
        in_token[:, 0] = self.bos_id

        logits = []
        for i in range(num_steps):
            j = i + 1  # next token index
            out_token = self._decode_step(
                token=in_token[:, :j],
                img_features=img_features,
                pos_query=pos_queries[:, i:j],
                token_mask=token_mask[:j, :j],
                pos_query_mask=query_mask[i:j, :j]
            )
            p_i = self.head(out_token)
            logits.append(p_i)
            if j < num_steps:
                # greedy decode. add the next token index to the target input
                in_token[:, j] = p_i.squeeze().argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if test_mode and (in_token == self.eos_id).any(dim=-1).all():
                    break

        logits = torch.cat(logits, dim=1)
        # if self.refine_iters:
        query_mask[
            torch.triu(
                torch.ones(
                    num_steps, num_steps,
                    dtype=torch.bool,
                    device=self._device
                ),
                diagonal=2
            )] = 0
        bos = torch.full(
            (bs, 1),
            self.bos_id,
            dtype=torch.long,
            device=self._device
        )
        # for i in range(self.refine_iters):
        in_token = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
        tgt_padding_mask = (
            (in_token == self.eos_id).int().cumsum(-1) > 0)
        out_token = self._decode_step(
            token=in_token,
            img_features=img_features,
            pos_query=pos_queries,
            token_mask=token_mask,
            pad_mask=tgt_padding_mask,
            pos_query_mask=query_mask[:, :in_token.shape[1]]
        )
        logits = self.head(out_token)
        return logits

    def forward(
        self,
        token: Tensor,
        img_features: Tensor,
    ):
        perms = self.perms(token)
        in_token = token[:, :-1]
        out_token = token[:, 1:]
        pad_mask = (in_token == self.pad_id) | (in_token == self.eos_id)

        loss = 0
        loss_numel = 0
        n = (out_token != self.pad_id).sum().item()
        for i, perm in enumerate(perms):
            token_mask, query_mask = self.attn_masks(perm)
            out = self._decode_step(
                token=in_token,
                img_features=img_features,
                token_mask=token_mask,
                pad_mask=pad_mask,
                pos_query_mask=query_mask)
            logits = self.head(out).flatten(end_dim=1)
            out_token = out_token.flatten()
            loss += n * F.cross_entropy(logits, out_token, ignore_index=self.pad_id)
            loss_numel += n
            if i == 1:
                out_token = torch.where(
                    out_token == self.eos_id, self.pad_id, out_token)
                n = (out_token != self.pad_id).sum().item()
        loss /= loss_numel
        return loss


"""
class CRNNBackbone(nn.Module):

    class _cnn(nn.Module):

        def __init__(
            self,
        ) -> None:
            super().__init__()

            self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 1))
            self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
            self.batchnorm4 = nn.BatchNorm2d(512)
            self.relu1 = nn.ReLU()
            self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.batchnorm5 = nn.BatchNorm2d(512)
            self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 1))
            self.relu2 = nn.ReLU()
            self.conv6 = nn.Conv2d(512, 512, 2, stride=1, padding=0)
            self.relu3 = nn.ReLU()

            self.out_channels = 512
            self.out_size_factors = (0.125, 1)
            self.out_size_offset = (-1, -1)

        def forward(self, img_features: Tensor):
            x = self.max_pool1(img_features)
            x = self.conv4(x)
            x = self.batchnorm4(x)
            x = self.relu1(x)
            x = self.conv5(x)
            x = self.max_pool2(x)
            x = self.batchnorm5(x)
            x = self.relu2(x)
            x = self.conv6(x)
            x = self.relu3(x)
            return x

    def __init__(
        self,
        in_channels: int,
        feat_size: Size,
        charset_size: int,
        mlp_hidden: int = 64,
        rnn_hidden: int = 256
    ):
        super().__init__()
        assert in_channels == 256

        self.cnn = self._cnn()
        self.cnn_out_channels = self.cnn.out_channels
        cnn_feat_size = (
            feat_size[0] * self.cnn.out_size_factors[0] + self.cnn.out_size_offset[0],
            feat_size[1] * self.cnn.out_size_factors[1] + self.cnn.out_size_offset[1],
        )

        self.map_to_seq = nn.Linear(int(self.cnn_out_channels * cnn_feat_size[0]), mlp_hidden)

        self.rnn1 = nn.LSTM(mlp_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        # self.rnn1 = nn.GRU(mlp_hidden, rnn_hidden, bidirectional=True)
        # self.rnn2 = nn.GRU(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, charset_size)

    def _load_pretrain(self, checkpoints: str):
        state_dict = torch.load(checkpoints)
        self_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in self_dict}
        self_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict)

    @torch.inference_mode()
    def infer(self, img_features: Tensor):
        return self.forward(img_features)

    def forward(self, img_features: Tensor):
        x = self.cnn(img_features)  # [48, 512, 7, 63]
        b, c, h, w = x.shape

        x = x.view(b, c * h, w)
        x = x.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(x)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        output = torch.nn.functional.log_softmax(output, dim=2)
        return output  # shape: (seq_len, batch, num_class)
"""


class CRNNBackbone(nn.Module):

    class _cnn(nn.Module):

        def __init__(
            self,
        ) -> None:
            super().__init__()

            self.max_pool1 = nn.AvgPool2d(kernel_size=(2, 1))
            self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
            self.batchnorm4 = nn.BatchNorm2d(512)
            self.relu1 = nn.ReLU()
            self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.batchnorm5 = nn.BatchNorm2d(512)
            self.max_pool2 = nn.AvgPool2d(kernel_size=(2, 1))
            self.relu2 = nn.ReLU()
            self.conv6 = nn.Conv2d(512, 512, 2, stride=1, padding=0)
            self.relu3 = nn.ReLU()

            self.out_channels = 512
            self.out_size_factors = (0.125, 1)
            self.out_size_offset = (-1, -1)

        def forward(self, img_features: Tensor):
            x = self.max_pool1(img_features)
            x = self.conv4(x)
            x = self.batchnorm4(x)
            x = self.relu1(x)
            x = self.conv5(x)
            x = self.max_pool2(x)
            x = self.batchnorm5(x)
            x = self.relu2(x)
            x = self.conv6(x)
            x = self.relu3(x)
            return x

    def __init__(
        self,
        in_channels: int,
        feat_size: Size,
        charset_size: int,
        mlp_hidden: int = 128,
        rnn_hidden: int = 256
    ):
        super().__init__()
        assert in_channels == 256

        self.cnn = self._cnn()
        self.cnn_out_channels = self.cnn.out_channels
        cnn_feat_size = (
            feat_size[0] * self.cnn.out_size_factors[0] + self.cnn.out_size_offset[0],
            feat_size[1] * self.cnn.out_size_factors[1] + self.cnn.out_size_offset[1],
        )

        self.map_to_seq = nn.Linear(int(self.cnn_out_channels * cnn_feat_size[0]), mlp_hidden)

        # self.rnn1 = nn.LSTM(mlp_hidden, rnn_hidden, bidirectional=True)
        # self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.rnn1 = nn.GRU(mlp_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.GRU(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, charset_size)

    def _load_pretrain(self, checkpoints: str):
        state_dict = torch.load(checkpoints)
        self_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in self_dict}
        self_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict)

    @torch.inference_mode()
    def infer(self, img_features: Tensor):
        return self.forward(img_features)

    def forward(self, img_features: Tensor):
        x = self.cnn(img_features)  # [48, 512, 7, 63]
        b, c, h, w = x.shape

        x = x.view(b, c * h, w)
        x = x.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(x)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        output = torch.nn.functional.log_softmax(output, dim=2)
        return output  # shape: (seq_len, batch, num_class)


class Recognizer(EDDModule):

    @EDDModule.INIT_DEVICE.move
    def __init__(
        self,
        configs
    ) -> None:
        super().__init__(configs)

        # self._backbone = ParseqBackbone(
        #     charset_size=configs.charset_size,
        #     max_label_length=configs.max_label_length,
        #     dim_models=configs.dim_models,
        #     num_heads=configs.num_heads,
        #     dim_feedforward=configs.dim_feedforward,
        #     device=configs.device,
        #     max_num_perm=configs.max_num_perm,
        #     pretrain=configs.pretrain
        # )
        self.max_label_len = configs.max_label_len

        self.conv = nn.Conv2d(
            configs.in_channels,
            configs.in_channels,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 1))

        self._backbone = CRNNBackbone(
            in_channels=configs.in_channels,
            feat_size=configs.feature_size,
            charset_size=configs.charset_size,
            mlp_hidden=configs.mlp_hidden,
            rnn_hidden=configs.rnn_hidden
        )

        self._rec_loss_fn = nn.CTCLoss(blank=CTCTokenizer._BLANK_ID_)

        if configs.load_pretrain:
            self._backbone._load_pretrain(configs.pretrain_recognizer)

    def infer(
        self,
        img_features: Tensor
    ):
        img_features = self.conv(img_features)
        img_features = self.relu(img_features)
        img_features = self.max_pool(img_features)
        logits = self._backbone.infer(img_features)
        logits = logits.permute(1, 0, 2)
        return logits

    def evaluate(
        self,
        img_features: Tensor
    ):
        img_features = self.conv(img_features)
        img_features = self.relu(img_features)
        img_features = self.max_pool(img_features)
        logits = self._backbone.infer(img_features)
        logits = logits.permute(1, 0, 2)
        return logits

    def _forward_step(
        self,
        img_features: Tensor
    ):
        img_features = self.conv(img_features)
        img_features = self.relu(img_features)
        img_features = self.max_pool(img_features)
        return self._backbone(img_features)

    def _get_loss(self, preds, target, pred_lengths, target_lengths):
        return self._rec_loss_fn(preds, target, pred_lengths, target_lengths)

    def forward(self, img_features, target_labels, target_lengths):
        bs = img_features.shape[0]
        preds: Tensor = self._forward_step(img_features)
        pred_lengths = torch.full((bs,), fill_value=self.max_label_len, dtype=torch.long, device=preds.device)
        return self._get_loss(preds, target_labels, pred_lengths, target_lengths)
