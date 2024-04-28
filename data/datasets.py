import os
import re
import unicodedata
import lmdb
import bisect
import numpy as np
from io import BytesIO
from typing import Tuple, List
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from srec.utils.utils import DefocusBlur
from PIL import Image


class TextZoomDataset(Dataset):
    """
    TextZoom dataset has `train1`, `train2` and `test`
    each database has the key `label-**`, `image_hr-**` and `image_lr-**`
    the return sequence is (`label-**`, `image_hr-**` and `image_lr-**`)
    """

    def __init__(
        self,
        db_path: str,
        target_charset: str,
        img_type: str = "RGB",
        min_label_length: int = 2,
        max_label_length: int = 25,
        min_img_dim: int = 0,
        min_psnr: float = 16.0,
        defocus_blur: bool = False,
        rotate_degrees: int = 0
    ) -> None:
        super(TextZoomDataset, self).__init__()
        self.db_path: str = db_path
        self.env = lmdb.Environment(
            path=self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        # data augmentation
        self.defocus_blur = defocus_blur
        self.rotate_degrees = rotate_degrees
        self._rng = np.random.default_rng()

        transform_fn = [transforms.ToTensor()]
        if self.defocus_blur:
            transform_fn.append(DefocusBlur(rng=self._rng))
        if self.rotate_degrees != 0:
            transform_fn.append(transforms.RandomRotation(self.rotate_degrees))
        self.transform = transforms.Compose(transform_fn)

        self.img_type = img_type
        self.labels = []
        self.clean_data_idx = []

        self.num_samples = self._filtrate(max_label_length, min_label_length, min_img_dim, target_charset)

    def _binary_2_PIL(self, img_binary_stream: bytes) -> Image:
        return Image.open(BytesIO(img_binary_stream)).convert(self.img_type)

    def _filtrate(
        self, max_label_length: int,
        min_label_length: int,
        min_img_dim: int,
        target_charset: str, whitespace: bool = False,
        unicode: bool = True
    ) -> int:
        """
        filtrate data, including labels and imgs
        for labels: ignore the labels whose length is greater than `max_label_length`,
            if `target_charset` exists, we will igonre the chars that are not in `target_charset`
        for imgs: ignore the imgs whose widths or heights are less than the `min_img_dim`
        the clean piece of data are stored in list using their indices
        """
        lower_case = target_charset.islower()
        upper_case = target_charset.isupper()
        unsupport = re.compile(f'[^{re.escape(target_charset)}]')
        with self.env.begin(write=False) as txn:
            total_length = int(txn.get("num-samples".encode()))
            for idx in range(1, total_length + 1):
                # label filter
                label_key = "label-{:0>9d}".format(idx).encode()
                label: str = txn.get(label_key).decode()
                if not whitespace:
                    label = ''.join(label.split())
                if unicode:
                    label = unicodedata.normalize("NFKD", label).encode("ascii", errors="ignore").decode()
                if len(label) > max_label_length or len(label) < min_label_length:
                    continue
                # convert the label into target charset
                if lower_case:
                    label = label.lower()
                if upper_case:
                    label = label.upper()
                # check if the label chars are all in charset
                label = unsupport.sub('', label)
                if not label:
                    # illegal char(s), ignore this piece of data
                    continue

                # img filter
                if min_img_dim > 0:
                    img_hr_key = "image_hr-{:0>9d}".format(idx)
                    img_lr_key = "image_lr-{:0>9d}".format(idx)
                    img_hr: Image = self._binary_2_PIL(txn.get(img_hr_key.encode()))
                    img_lr: Image = self._binary_2_PIL(txn.get(img_lr_key.encode()))
                    hr_w, hr_h = img_hr.size
                    lr_w, lr_h = img_lr.size
                    if (
                        hr_w < min_img_dim or hr_h < min_img_dim
                        or lr_w < min_img_dim or lr_h < min_img_dim
                    ):
                        continue
                self.labels.append(label)
                self.clean_data_idx.append(idx)
        return len(self.clean_data_idx)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor]:
        """
        the index of mdb file ranges from [1, max_length]
        but we index the dataset from [0, max_length - 1]
        """
        assert 0 <= index < self.num_samples, "index range error in TextZoomDataset"
        label = self.labels[index]
        index = self.clean_data_idx[index]
        with self.env.begin(write=False) as txn:
            img_hr_key = "image_hr-{:0>9d}".format(index)
            img_lr_key = "image_lr-{:0>9d}".format(index)

            img_hr = self._binary_2_PIL(txn.get(img_hr_key.encode()))
            img_lr = self._binary_2_PIL(txn.get(img_lr_key.encode()))

        return label, self.transform(img_hr), self.transform(img_lr)


class STIDataset(Dataset):
    """
    concatenated dataset, used for STI tasks
    """

    def __init__(self, ds_list: List):
        super(STIDataset, self).__init__()
        self.ds_list = ds_list
        self.ds_len = []
        self.ds_cumsum = []

        for ds in self.ds_list:
            self.ds_len.append(len(ds))

        cumsum = 0
        for l in self.ds_len:
            cumsum += l
            self.ds_cumsum.append(cumsum)

    def __len__(self):
        return sum(self.ds_len)

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor]:
        # insert by the left
        ds_idx = bisect.bisect_right(self.ds_cumsum, index)
        items = self.ds_list[ds_idx][index - (self.ds_cumsum[ds_idx] - self.ds_len[ds_idx])]
        return items
