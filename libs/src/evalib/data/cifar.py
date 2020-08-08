import numpy as np
import torch
from albumentations import CoarseDropout
from albumentations import Compose
from albumentations import HorizontalFlip
from albumentations import HueSaturationValue
from albumentations import Normalize
from albumentations import Rotate
from albumentations.pytorch import ToTensor
from torch import cuda
from torch.utils.data import DataLoader
from torchvision import datasets


class CIFAR:
    def __init__(self, args):
        super(CIFAR, self).__init__()
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        self._args = args

        self._prepare_dataloader()

    def _train_mean(self):
        return (0.4914, 0.4822, 0.4465)

    def _train_std(self):
        return (0.247, 0.243, 0.261)

    def _test_mean(self):
        return (0.4914, 0.4822, 0.4465)

    def _test_std(self):
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
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transforms)

        # No Augumentation while testing
        test_transforms_album = Compose(
            [
                Normalize(mean=self._test_mean(), std=self._test_std()),
                ToTensor()
                # transforms.ToTensor(),
                # transforms.Normalize(self._test_mean(), self._test_std()),
            ]
        )

        def test_transforms(img):
            return test_transforms_album(image=np.array(img))["image"]

        # Pytorch default approach
        # test_transforms = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(self._test_mean(), self._test_std()),
        #     ]
        # )
        test_dataset = datasets.CIFAR10(root="./data", train=False, transform=test_transforms)

        return train_dataset, test_dataset

    def _prepare_dataloader(self):
        train_dataset, test_dataset = self._prepare_data()

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
        self.test_loader = DataLoader(test_dataset, **dlargs)
