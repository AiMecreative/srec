import torch
import torch.nn as nn
from torch import Tensor
from bidict import bidict
from typing import List
from torch.nn.utils.rnn import pad_sequence


class Tokenizer(object):

    _EOS_ = '[E]'
    _PAD_ = '[P]'
    _BOS_ = '[B]'

    def __init__(
        self,
        charset: str,
    ) -> None:
        charset = [self._EOS_] + list(charset) + [self._BOS_, self._PAD_]
        char_map = {}
        for idx, c in enumerate(charset):
            char_map[c] = idx
        self._charset_map = bidict(char_map)

    def tokenize(
        self,
        label_chars: List,
        device: str
    ):
        def _label2idx(label): return [self._charset_map[c] for c in label]
        label_batch = ([torch.tensor(
            [self._charset_map[self._BOS_]] + _label2idx(l) + [self._charset_map[self._EOS_]],
            dtype=torch.long,
            device=device)
            for l in label_chars
        ])
        label_batch = pad_sequence(label_batch, batch_first=True, padding_value=self._charset_map[self._PAD_])
        return label_batch

    def untokenize(self, logits):
        def _idx2label(indices): return (
            [self._charset_map.inv[i.item()]
             for i in indices]
        )
        batched_tokens = []
        for logit in logits:
            _, tokens = logit.max(dim=-1)
            batched_tokens.append(_idx2label(tokens))
        merged = []
        for tok in batched_tokens:
            chars = ""
            for c in tok:
                if c in [self._BOS_, self._EOS_]:
                    break
                if c in [self._PAD_]:
                    continue
                chars += c
            merged.append(chars)
        return merged


class CTCTokenizer(object):

    _BLANK_ = '[B]'
    _BLANK_ID_ = 0

    def __init__(
        self,
        charset: str
    ) -> None:
        charset = [self._BLANK_] + list(charset)
        char_map = {}
        for idx, c in enumerate(charset):
            char_map[c] = idx
        self._charset_map = bidict(char_map)

    def _reconstruct(self, tokens):
        res = []
        # merge same labels
        previous = None
        for l in tokens:
            if l != previous:
                res.append(l)
                previous = l
        # delete blank
        res = [l for l in res if l != self._BLANK_ID_]
        return res

    def tokenize(self, labels, device: str):
        lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)
        def _label2idx(label): return [self._charset_map[c] for c in label]
        # label_batch = ([torch.tensor(
        #     _label2idx(l),
        #     dtype=torch.long,
        #     device=device)
        #     for l in labels
        # ])
        # label_batch = pad_sequence(label_batch, batch_first=True, padding_value=self._charset_map[self._BLANK_])
        label_batch = torch.tensor([tok for l in labels for tok in _label2idx(l)], dtype=torch.long, device=device)
        return label_batch, lengths

    def decode_logits(self, logits: Tensor):
        token_li = []
        tokens = torch.argmax(logits, dim=-1)
        for token in tokens:
            token = self._reconstruct(token)
            token_li.append(token)
        return token_li

    def untokenize(self, token_list):
        def _idx2label(indices): return ''.join([self._charset_map.inv[i.detach().cpu().item()] for i in indices])
        labels = []
        for tokens in token_list:
            labels.append(_idx2label(tokens))
        return labels
