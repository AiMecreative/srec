import torch
import torch.nn as nn


class ModelWrapper(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        super().__init__()

        self.module = module

    def load_pretrain(
        self,
        file: str,
        map_location: str = None,
        strict: bool = True,
        use_wrapper: bool = False,
        key: str = None
    ):
        state_dicts = torch.load(file, map_location)
        if use_wrapper:
            self.load_state_dict(state_dicts, strict=strict)[key]
        else:
            self.module.load_state_dict(state_dicts, strict=strict)[key]

    def forward(self, *args):
        return self.module.forward(*args)
