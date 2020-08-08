"""
Author: Ramjee Ganti
Date: 2020-Apr-19
----
Original Author: Meng Lee, mnicnc404
Date: 2018/06/04
References:
    - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
"""
import glob
import os

from PIL import Image
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset

EXTENSION = "JPEG"
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = "wnids.txt"
VAL_ANNOTATION_FILE = "val_annotations.txt"


class TinyImageNet(VisionDataset):

    """ TinyImageNet data set available from
    `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Args:
        root (string): Root directory including `train`, `test` and `val`
            subdirectories.
        split (string, optional): Indicating which split to return as a data set.
            Valid option: [`train`, `test`, `val`]
        transform (callable, optional): torchvision.transforms A (series) of valid
            transformation(s).
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        in_memory: bool
            Set to True if there is enough memory (about 5G)
            and want to minimize disk IO overhead.

    Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    base_folder = "tiny-imagenet"
    file_list = [
        (
            "tiny-imagenet-200.zip",
            "90528d7ca1a48142e341f4ef8d21d0de",
            "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        )
    ]

    def __init__(
        self, root, split="train", transform=None, target_transform=None, download=False, in_memory=False, **kwargs
    ):

        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = os.path.expanduser(root)
        self.root_final = os.path.join(self.root, self.base_folder, "tiny-imagenet-200")

        self.split = verify_str_arg(
            split, "split", ("train", "val", "test")
        )  # training set , validation set or test set
        self.split_dir = os.path.join(self.root_final, self.split)

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")

        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, "**", "*.%s" % EXTENSION), recursive=True))

        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory

        self.classes = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root_final, CLASS_LIST_FILE), "r") as fp:
            self.classes_names = sorted([text.strip() for text in fp.readlines()])
        self.class_to_idx = {text: i for i, text in enumerate(self.classes_names)}

        if self.split == "train":
            for class_text, i in self.class_to_idx.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.classes["%s_%d.%s" % (class_text, cnt, EXTENSION)] = i
        elif self.split == "val":
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), "r") as fp:
                for line in fp.readlines():
                    terms = line.split("\t")
                    file_name, class_text = terms[0], terms[1]
                    self.classes[file_name] = self.class_to_idx[class_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def _check_integrity(self):
        dir_exists = os.path.isdir(self.root_final)

        if not dir_exists:
            for (filename, md5, _url) in self.file_list:
                fpath = os.path.join(self.root, self.base_folder, filename)
                _, ext = os.path.splitext(filename)
                # Allow original archive to be deleted (zip and 7z)
                # Only need the extracted images
                if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                    return False

        # Should check a hash of the images
        return dir_exists

    def _download(self):

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for (filename, md5, url) in self.file_list:
            download_and_extract_archive(
                url, os.path.join(self.root, self.base_folder), filename=filename, md5=md5, remove_finished=True,
            )

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == "test":
            return img
        else:
            # file_name = file_path.split('/')[-1]
            return img, self.classes[os.path.basename(file_path)]

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img


if __name__ == "__main__":

    tiny_train = TinyImageNet("~/Downloads", split="train", download=True)
    print(len(tiny_train))
    print(tiny_train.__getitem__(99))
    for fname, number in tiny_train.classes.items():
        if number == 19:
            print(fname, number)

    tiny_train = TinyImageNet("~/Downloads", split="val")
    print(tiny_train.__getitem__(99))

    # # in-memory test
    # tiny_val = TinyImageNet("~/Downloads", split="test", in_memory=True)
