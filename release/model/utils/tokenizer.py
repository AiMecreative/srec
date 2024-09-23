import torch
from bidict import bidict
from torch import Tensor


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
        label_batch = torch.tensor([tok for l in labels for tok in _label2idx(l)], dtype=torch.long, device=device)
        return label_batch, lengths

    def decode_logits(self, logits: Tensor):
        logits = logits.permute(1, 0, 2)
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
