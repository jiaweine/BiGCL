"""Data augmentation strategies for BiGCL.

Provides CLIP-compatible augmentations and multi-crop strategies for
contrastive learning on image clustering benchmarks.
"""

import torch
from torchvision import transforms

# CLIP image normalization (OpenAI CLIP training stats)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
clip_normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)


class CLIPAugmentation:
    """Dual-view augmentation pipeline compatible with CLIP preprocessing.

    Returns two augmented views of the same image for contrastive learning,
    using asymmetric augmentation strengths (strong + weak).
    """

    def __init__(
        self,
        image_size: int = 224,
        strong: bool = True,
    ):
        self.image_size = image_size

        # Strong augmentation view
        self.strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.RandomSolarize(threshold=128, p=0.1),
            transforms.ToTensor(),
            clip_normalize,
        ])

        # Weak augmentation view
        self.weak_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            ], p=0.5),
            transforms.ToTensor(),
            clip_normalize,
        ])

        # Evaluation transform (no augmentation)
        self.eval_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            clip_normalize,
        ])

        self.strong = strong

    def __call__(self, image):
        """Return two augmented views of the input image."""
        if self.strong:
            v1 = self.strong_transform(image)
            v2 = self.weak_transform(image)
        else:
            v1 = self.weak_transform(image)
            v2 = self.weak_transform(image)
        return v1, v2


class MultiCropAugmentation:
    """Multi-crop augmentation following DINO/SwAV strategy.

    Generates a mix of global and local crops for multi-scale contrastive learning.
    """

    def __init__(
        self,
        image_size: int = 224,
        n_global: int = 2,
        n_local: int = 4,
        global_scale: tuple = (0.4, 1.0),
        local_scale: tuple = (0.05, 0.4),
        local_size: int = 96,
    ):
        self.n_global = n_global
        self.n_local = n_local

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            clip_normalize,
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            clip_normalize,
        ])

    def __call__(self, image):
        """Return list of global + local crop views."""
        views = []
        for _ in range(self.n_global):
            views.append(self.global_transform(image))
        for _ in range(self.n_local):
            views.append(self.local_transform(image))
        return views


def build_augmentation(cfg) -> CLIPAugmentation:
    """Factory function for building augmentation pipeline from config."""
    return CLIPAugmentation(
        image_size=cfg.get("image_size", 224),
        strong=cfg.get("strong_aug", True),
    )
