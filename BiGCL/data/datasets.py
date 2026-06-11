"""Dataset loaders for standard image clustering benchmarks.

Supports: CIFAR-10, CIFAR-100, STL-10, ImageNet-10, ImageNet-Dogs,
Tiny-ImageNet, and custom image folder datasets.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms as T
from PIL import Image

from .augmentations import CLIPAugmentation


def _extract_targets(base_dataset):
    """Extract target labels from a torchvision-style dataset."""
    if hasattr(base_dataset, "targets"):
        return np.array(base_dataset.targets)
    elif hasattr(base_dataset, "labels"):
        return np.array(base_dataset.labels)
    return None


def _ensure_pil_rgb(img):
    """Convert a Tensor or non-RGB PIL image to an RGB PIL Image."""
    if isinstance(img, torch.Tensor):
        img = T.ToPILImage()(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


class ClusteringDataset(Dataset):
    """Wrapper dataset that applies dual-view augmentation for contrastive learning.

    Wraps any torchvision-style dataset and returns two augmented views
    plus the ground-truth label for evaluation.
    """

    def __init__(self, base_dataset, augmentation=None, return_index=True):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.return_index = return_index
        self.targets = _extract_targets(base_dataset)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        if self.augmentation is not None:
            img = _ensure_pil_rgb(img)
            v1, v2 = self.augmentation(img)
        else:
            v1 = img
            v2 = img

        out = {"view1": v1, "view2": v2, "label": label}
        if self.return_index:
            out["index"] = idx
        return out


class EvalDataset(Dataset):
    """Dataset wrapper for evaluation (single view, no augmentation)."""

    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.targets = _extract_targets(base_dataset)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        if self.transform is not None:
            img = _ensure_pil_rgb(img)
            img = self.transform(img)

        return {"image": img, "label": label, "index": idx}


def build_dataset(
    name: str,
    data_dir: str = "./datasets",
    split: str = "train",
    image_size: int = 224,
    strong_aug: bool = True,
    eval_mode: bool = False,
) -> Dataset:
    """Build dataset by name with appropriate augmentations.

    Args:
        name: dataset name (cifar10, cifar100, stl10, imagenet10, tinyimagenet)
        data_dir: root directory for datasets
        split: train or test
        image_size: target image size
        strong_aug: whether to use strong augmentation
        eval_mode: if True, return evaluation dataset
    Returns:
        Dataset instance
    """
    augmentation = CLIPAugmentation(image_size=image_size, strong=strong_aug)
    is_train = split == "train"

    if name == "fashionmnist":
        base = datasets.FashionMNIST(
            root=data_dir, train=is_train, download=True
        )
        n_clusters = 10
    elif name == "cifar10":
        base = datasets.CIFAR10(
            root=data_dir, train=is_train, download=True
        )
        n_clusters = 10
    elif name == "cifar100":
        base = datasets.CIFAR100(
            root=data_dir, train=is_train, download=True
        )
        # Use 20 superclass (coarse) labels instead of 100 fine-grained labels
        import pickle as _pkl
        _split_file = "train" if is_train else "test"
        _meta_path = os.path.join(data_dir, "cifar-100-python", _split_file)
        with open(_meta_path, "rb") as _f:
            _coarse = _pkl.load(_f, encoding="bytes")[b"coarse_labels"]
        base.targets = _coarse
        n_clusters = 20
    elif name == "cifar100_full":
        base = datasets.CIFAR100(
            root=data_dir, train=is_train, download=True
        )
        n_clusters = 100
    elif name == "stl10":
        stl_split = "train" if is_train else "test"
        base = datasets.STL10(
            root=data_dir, split=stl_split, download=True
        )
        n_clusters = 10
    elif name == "imagenet10":
        # Subset of ImageNet with 10 classes
        img_dir = os.path.join(data_dir, "imagenet10", split)
        base = datasets.ImageFolder(root=img_dir)
        n_clusters = 10
    elif name == "imagenet_dogs":
        img_dir = os.path.join(data_dir, "imagenet_dogs", split)
        base = datasets.ImageFolder(root=img_dir)
        n_clusters = 15
    elif name == "tinyimagenet":
        img_dir = os.path.join(data_dir, "tiny-imagenet-200", split)
        base = datasets.ImageFolder(root=img_dir)
        n_clusters = 200
    elif name == "imagenet_r":
        img_dir = os.path.join(data_dir, "imagenet-r")
        base = datasets.ImageFolder(root=img_dir)
        n_clusters = 200
    else:
        # Custom image folder
        img_dir = os.path.join(data_dir, name, split)
        base = datasets.ImageFolder(root=img_dir)
        n_classes = len(base.classes)
        n_clusters = n_classes

    if eval_mode:
        return EvalDataset(base, transform=augmentation.eval_transform), n_clusters
    else:
        return ClusteringDataset(base, augmentation=augmentation), n_clusters


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Build DataLoader with standard settings."""

    def collate_fn(batch):
        """Custom collate for dict-based datasets."""
        out = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                out[key] = torch.stack([b[key] for b in batch])
            elif isinstance(batch[0][key], (int, np.integer)):
                out[key] = torch.tensor([b[key] for b in batch])
            else:
                out[key] = [b[key] for b in batch]
        return out

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
