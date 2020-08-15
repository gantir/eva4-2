import numpy as np
import torch
from albumentations import CoarseDropout
from albumentations import Compose
from albumentations import HorizontalFlip
from albumentations import HueSaturationValue
from albumentations import Normalize
from albumentations import Rotate
from albumentations.pytorch import ToTensor
from evalib import datasets
from torch import cuda
from torch.utils.data import DataLoader


class TinyImageNet:
    def __init__(self, args):
        super(TinyImageNet, self).__init__()

        self._args = args
        self._prepare_dataloader()

    def _train_mean(self):
        return (0.4914, 0.4822, 0.4465)

    def _train_std(self):
        return (0.247, 0.243, 0.261)

    def _val_mean(self):
        return (0.4914, 0.4822, 0.4465)

    def _val_std(self):
        return (0.247, 0.243, 0.261)

    def _prepare_data(self):
        # Augumentation used only while training
        train_transforms_album = Compose(
            [
                HueSaturationValue(p=0.25),
                HorizontalFlip(p=0.5),
                Rotate(limit=15),
                CoarseDropout(
                    max_holes=1,
                    max_height=16,
                    max_width=16,
                    min_height=4,
                    min_width=4,
                    fill_value=np.array(self._train_mean()) * 255.0,
                    p=0.75,
                ),
                Normalize(mean=self._train_mean(), std=self._train_std()),
                ToTensor(),
            ]
        )

        def train_transforms(img):
            return train_transforms_album(image=np.array(img))["image"]

        # train_transforms = transforms.Compose(
        #     [
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(self._train_mean(), self._train_std()),
        #     ]
        # )
        train_dataset = datasets.TinyImageNet(
            root=self._args.data_path, split="train", transform=train_transforms, download=True,
        )

        # No Augumentation while Validation
        val_transforms_album = Compose(
            [
                Normalize(mean=self._val_mean(), std=self._val_std()),
                ToTensor()
                # transforms.ToTensor(),
                # transforms.Normalize(self._val_mean(), self._val_std()),
            ]
        )

        def val_transforms(img):
            return val_transforms_album(image=np.array(img))["image"]

        # Pytorch default approach
        # val_transforms = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(self._val_mean(), self._val_std()),
        #     ]
        # )
        val_dataset = datasets.TinyImageNet(
            root=self._args.data_path, split="val", transform=val_transforms, download=True,
        )

        return train_dataset, val_dataset

    def _prepare_dataloader(self):
        train_dataset, val_dataset = self._prepare_data()

        torch.manual_seed(self._args.SEED)
        dlargs = {"batch_size": self._args.batch_size}
        if cuda.is_available():
            torch.cuda.manual_seed(self._args.SEED)
            dlargs = {
                "num_workers": self._args.num_workers,
                "pin_memory": True,
                "batch_size": self._args.batch_size_cuda,
            }

        self.train_loader = DataLoader(train_dataset, **dlargs)
        self.val_loader = DataLoader(val_dataset, **dlargs)
