"""Unit tests for data/datasets.py."""

import torch
import numpy as np
import pytest
from unittest.mock import MagicMock
from PIL import Image

from data.datasets import ClusteringDataset, EvalDataset, build_dataloader
from data.augmentations import CLIPAugmentation


class FakeBaseDataset:
    """Minimal fake dataset for testing wrappers."""

    def __init__(self, n=20, n_classes=5):
        self.data = [
            Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            for _ in range(n)
        ]
        self.targets = list(np.random.randint(0, n_classes, size=n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TestClusteringDataset:
    @pytest.fixture
    def dataset(self):
        base = FakeBaseDataset(n=10)
        aug = CLIPAugmentation(image_size=32)
        return ClusteringDataset(base, augmentation=aug, return_index=True)

    def test_length(self, dataset):
        assert len(dataset) == 10

    def test_getitem_keys(self, dataset):
        item = dataset[0]
        assert "view1" in item
        assert "view2" in item
        assert "label" in item
        assert "index" in item

    def test_view_shapes(self, dataset):
        item = dataset[0]
        assert item["view1"].shape == (3, 32, 32)
        assert item["view2"].shape == (3, 32, 32)

    def test_index_matches(self, dataset):
        item = dataset[5]
        assert item["index"] == 5

    def test_no_index_mode(self):
        base = FakeBaseDataset(n=5)
        aug = CLIPAugmentation(image_size=32)
        dataset = ClusteringDataset(base, augmentation=aug, return_index=False)
        item = dataset[0]
        assert "index" not in item

    def test_no_augmentation(self):
        base = FakeBaseDataset(n=5)
        dataset = ClusteringDataset(base, augmentation=None)
        item = dataset[0]
        # Without augmentation, views are the raw PIL image
        assert "view1" in item

    def test_targets_extracted(self):
        base = FakeBaseDataset(n=10, n_classes=3)
        dataset = ClusteringDataset(base, augmentation=None)
        assert dataset.targets is not None
        assert len(dataset.targets) == 10


class TestEvalDataset:
    @pytest.fixture
    def dataset(self):
        base = FakeBaseDataset(n=8)
        aug = CLIPAugmentation(image_size=32)
        return EvalDataset(base, transform=aug.eval_transform)

    def test_length(self, dataset):
        assert len(dataset) == 8

    def test_getitem_keys(self, dataset):
        item = dataset[0]
        assert "image" in item
        assert "label" in item
        assert "index" in item

    def test_image_shape(self, dataset):
        item = dataset[0]
        assert item["image"].shape == (3, 32, 32)

    def test_index(self, dataset):
        item = dataset[3]
        assert item["index"] == 3

    def test_no_transform(self):
        base = FakeBaseDataset(n=5)
        dataset = EvalDataset(base, transform=None)
        item = dataset[0]
        assert "image" in item

    def test_targets_from_labels_attr(self):
        """Test extraction when base has 'labels' attribute instead of 'targets'."""
        base = FakeBaseDataset(n=5)
        base.labels = base.targets
        del base.targets
        dataset = EvalDataset(base, transform=None)
        assert dataset.targets is not None


class TestBuildDataloader:
    def test_returns_dataloader(self):
        base = FakeBaseDataset(n=16)
        aug = CLIPAugmentation(image_size=32)
        dataset = ClusteringDataset(base, augmentation=aug)
        loader = build_dataloader(dataset, batch_size=4, num_workers=0, drop_last=False)
        assert loader is not None

    def test_batch_shape(self):
        base = FakeBaseDataset(n=16)
        aug = CLIPAugmentation(image_size=32)
        dataset = ClusteringDataset(base, augmentation=aug)
        loader = build_dataloader(dataset, batch_size=4, num_workers=0, drop_last=False)
        batch = next(iter(loader))
        assert batch["view1"].shape == (4, 3, 32, 32)
        assert batch["view2"].shape == (4, 3, 32, 32)
        assert batch["label"].shape == (4,)

    def test_collate_stacks_tensors(self):
        base = FakeBaseDataset(n=8)
        aug = CLIPAugmentation(image_size=32)
        dataset = ClusteringDataset(base, augmentation=aug, return_index=True)
        loader = build_dataloader(dataset, batch_size=4, num_workers=0, drop_last=False)
        batch = next(iter(loader))
        assert isinstance(batch["view1"], torch.Tensor)
        assert isinstance(batch["label"], torch.Tensor)
        assert isinstance(batch["index"], torch.Tensor)

    def test_shuffle_option(self):
        base = FakeBaseDataset(n=16)
        aug = CLIPAugmentation(image_size=32)
        dataset = ClusteringDataset(base, augmentation=aug)
        loader = build_dataloader(dataset, batch_size=8, num_workers=0, shuffle=False, drop_last=False)
        batch = next(iter(loader))
        # First batch indices should be 0..7 when not shuffled
        assert batch["index"][0].item() == 0
