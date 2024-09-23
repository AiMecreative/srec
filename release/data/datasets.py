import os
import re
import unicodedata
import lmdb
import bisect
import numpy as np
from io import BytesIO
from typing import Tuple, List, Dict
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile


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
        read_mode: str = "RGB",
        img_type: str = 'hr',
        min_label_length: int = 2,
        max_label_length: int = 25,
        min_img_dim: int = 0,
        perspective_scale: List[float] = None,
        rotate_degrees: int = 0,
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
        self._perspective = perspective_scale != 0
        self._rotate = rotate_degrees != 0
        self._percpective_scale = perspective_scale
        self._rotate_degrees = rotate_degrees
        self._rng = np.random.default_rng()

        self._img_type = img_type
        self.read_mode = read_mode
        self.labels = []
        self.clean_data_idx = []

        self.num_samples = self._filtrate(max_label_length, min_label_length, min_img_dim, target_charset)

    def _binary_2_PIL(self, img_binary_stream: bytes) -> Image:
        return Image.open(BytesIO(img_binary_stream)).convert(self.read_mode)

    def _filtrate(
        self,
        max_label_length: int,
        min_label_length: int,
        min_img_dim: int,
        target_charset: str,
        whitespace: bool = False,
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

            img_hr: Image.Image = self._binary_2_PIL(txn.get(img_hr_key.encode()))
            img_lr: Image.Image = self._binary_2_PIL(txn.get(img_lr_key.encode()))

        return label, img_hr, img_lr


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


class MJSynthDataset(Dataset):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __init__(
        self,
        db_path: str,  # data/dataset/mnt/ramdisk/max/90kDICT32px
        mode: str,
        target_charset: str,
        img_type: str = "RGB",
        min_label_length: int = 2,
        max_label_length: int = 25,
        min_img_dim: int = 0,
        min_psnr: float = 16.0,
        perspective_scale: List[float] = None,
        rotate_degrees: int = 0,
        is_init: bool = False
    ) -> None:
        super(MJSynthDataset, self).__init__()

        self._db_path: str = db_path
        self._mode = mode
        self._annotation_path = f'{self._db_path}/annotation_{self._mode}.txt'

        self._data_path = f'{self._db_path}/{mode}_data.txt'
        self._label_path = f'{self._db_path}/{mode}_label.txt'

        # data augmentation
        self._perspective = perspective_scale is not None
        self._rotate = rotate_degrees != 0
        self._percpective_scale = perspective_scale
        self._rotate_degrees = rotate_degrees

        self._rng = np.random.default_rng()

        self.img_type = img_type

        if is_init:
            # load label mappings
            self._data_paths = []
            self._labels = []
            label_map = {}
            with open(f'{self._db_path}/lexicon.txt', 'r') as lex:
                lex_li = lex.readlines()
            for label_idx, label in enumerate(lex_li):
                label_map[label_idx] = label

            # load labels with filtration
            self._num_samples = self._filtrate(
                max_label_length,
                min_label_length,
                min_img_dim,
                target_charset,
                label_map
            )

            with open(self._data_path, 'w') as df:
                df.writelines(self._data_paths)
            with open(self._label_path, 'w') as lf:
                lf.writelines(self._labels)

        if not is_init:
            with open(self._label_path, 'r') as lf:
                self._labels = lf.readlines()
            with open(self._data_path, 'r') as df:
                self._data_paths = df.readlines()
            self._num_samples = len(self._labels)

    def _filtrate(
        self,
        max_label_length: int,
        min_label_length: int,
        min_img_dim: int,
        target_charset: str,
        label_map: Dict[int, str],
        whitespace: bool = False,
        unicode: bool = True
    ):
        lower_case = target_charset.islower()
        upper_case = target_charset.isupper()
        unsupport = re.compile(f'[^{re.escape(target_charset)}]')
        total_num = 0
        with open(self._annotation_path, 'r') as lf:
            for piece in lf.readlines():
                path, label_idx = piece.split(' ')
                # read label
                label_idx = int(label_idx)
                label = label_map[label_idx]
                if not whitespace:
                    label = ''.join(label.split())
                if unicode:
                    label = unicodedata.normalize("NFKD", label).encode("ascii", errors="ignore").decode()
                if len(label) < min_label_length or max_label_length < len(label):
                    continue
                if lower_case:
                    label = label.lower()
                if upper_case:
                    label = label.upper()
                label = unsupport.sub('', label)
                if not label:
                    continue

                # check image
                image_path = f'{self._db_path}/{path}'
                # print(image_path)
                try:
                    image = Image.open(image_path)
                    image = transforms.PILToTensor()(image)
                    image = image / 255.0
                    c, h, w = image.shape
                    if h < min_img_dim or w < min_img_dim:
                        continue
                except IOError:
                    continue

                self._data_paths.append(image_path)
                self._labels.append(label)
                total_num += 1
                if total_num % 100 == 0:
                    print(f'\rload data num: {total_num}', end='', flush=True)
            print(f'finish loading data: total num: {len(self._labels)}')
        return len(self._labels)

    def __len__(self): return self._num_samples

    def __getitem__(self, index):
        label = self._labels[index]
        image_path = self._data_paths[index]
        image = Image.open(image_path).convert(self.img_type)
        image = transforms.PILToTensor()(image)
        image = image / 255.0
        # random choose augmentation param
        transform = []
        if self._perspective:
            perspective_scale = np.random.choice(np.array(self._percpective_scale))
            transform.append(transforms.RandomPerspective(perspective_scale, p=0.5))
        if self._rotate:
            rotate_degree = np.random.choice(np.array(self._rotate_degrees))
            transform.append(transforms.RandomRotation(rotate_degree))
        return label, len(label), image
