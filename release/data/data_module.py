from typing import Callable
from torch.utils.data import Dataset, DataLoader
from data.datasets import TextZoomDataset, STIDataset
from configs.mappings import DataConfigs


class DataModule(object):

    def __init__(
            self,
            data_configs: DataConfigs
    ) -> None:
        self.configs = data_configs

    def dataset(self, ds_file: str = None, is_init: bool = False) -> Dataset:
        # ds_file = ds_file if ds_file is not None else self.configs.ds_file
        ds_file = [ds_file] if isinstance(ds_file, str) else ds_file
        ds_li = []
        for file in ds_file:
            ds_li.append(
                TextZoomDataset(
                    file,
                    self.configs.charset,
                    self.configs.read_mode,
                    self.configs.img_type,
                    self.configs.min_label_len,
                    self.configs.max_label_len,
                    self.configs.min_img_len,
                    self.configs.perspective_scale,
                    self.configs.rotate_degree,
                )
            )
        return STIDataset(ds_li)

    def data_loader(
            self,
            ds_file: str = None,
            dataset: Dataset = None,
            collate_fn: Callable = None,
            batch_size: int = 1,
    ):
        if dataset is None:
            ds_file = ds_file if ds_file is not None else self.configs.ds_file
            dataset = self.dataset(ds_file)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=self.configs.shuffle,
            num_workers=self.configs.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )
