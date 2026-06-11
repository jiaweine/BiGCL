"""Unit tests for data/augmentations.py."""

import torch
import pytest
from PIL import Image

from data.augmentations import CLIPAugmentation, MultiCropAugmentation, build_augmentation


@pytest.fixture
def sample_image():
    """Create a random PIL image for testing."""
    return Image.fromarray(
        (torch.rand(64, 64, 3).numpy() * 255).astype("uint8"), mode="RGB"
    )


class TestCLIPAugmentation:
    def test_returns_two_views(self, sample_image):
        aug = CLIPAugmentation(image_size=32)
        v1, v2 = aug(sample_image)
        assert isinstance(v1, torch.Tensor)
        assert isinstance(v2, torch.Tensor)

    def test_output_shape(self, sample_image):
        aug = CLIPAugmentation(image_size=64)
        v1, v2 = aug(sample_image)
        assert v1.shape == (3, 64, 64)
        assert v2.shape == (3, 64, 64)

    def test_different_views(self, sample_image):
        """Two views should generally be different (stochastic augmentation)."""
        torch.manual_seed(42)
        aug = CLIPAugmentation(image_size=32, strong=True)
        v1, v2 = aug(sample_image)
        # Views should not be identical due to different augmentation strengths
        assert not torch.allclose(v1, v2)

    def test_weak_only_mode(self, sample_image):
        aug = CLIPAugmentation(image_size=32, strong=False)
        v1, v2 = aug(sample_image)
        assert v1.shape == (3, 32, 32)
        assert v2.shape == (3, 32, 32)

    def test_eval_transform(self, sample_image):
        aug = CLIPAugmentation(image_size=48)
        result = aug.eval_transform(sample_image)
        assert result.shape == (3, 48, 48)

    def test_normalized_output(self, sample_image):
        """Output should be roughly normalized (not in [0, 1] range)."""
        aug = CLIPAugmentation(image_size=32)
        v1, _ = aug(sample_image)
        # After CLIP normalization, values should not all be in [0, 1]
        assert v1.min() < 0 or v1.max() > 1


class TestMultiCropAugmentation:
    def test_returns_correct_number_of_views(self, sample_image):
        aug = MultiCropAugmentation(
            image_size=32, n_global=2, n_local=4, local_size=16
        )
        views = aug(sample_image)
        assert len(views) == 6  # 2 global + 4 local

    def test_global_crop_shape(self, sample_image):
        aug = MultiCropAugmentation(image_size=48, n_global=2, n_local=2, local_size=16)
        views = aug(sample_image)
        assert views[0].shape == (3, 48, 48)
        assert views[1].shape == (3, 48, 48)

    def test_local_crop_shape(self, sample_image):
        aug = MultiCropAugmentation(image_size=48, n_global=2, n_local=3, local_size=24)
        views = aug(sample_image)
        assert views[2].shape == (3, 24, 24)
        assert views[3].shape == (3, 24, 24)
        assert views[4].shape == (3, 24, 24)

    def test_all_tensors(self, sample_image):
        aug = MultiCropAugmentation(image_size=32, n_global=1, n_local=2, local_size=16)
        views = aug(sample_image)
        for v in views:
            assert isinstance(v, torch.Tensor)

    def test_zero_local_crops(self, sample_image):
        aug = MultiCropAugmentation(image_size=32, n_global=2, n_local=0)
        views = aug(sample_image)
        assert len(views) == 2


class TestBuildAugmentation:
    def test_default_config(self):
        cfg = {"image_size": 32, "strong_aug": True}
        aug = build_augmentation(cfg)
        assert isinstance(aug, CLIPAugmentation)
        assert aug.image_size == 32
        assert aug.strong is True

    def test_custom_config(self):
        cfg = {"image_size": 64, "strong_aug": False}
        aug = build_augmentation(cfg)
        assert aug.image_size == 64
        assert aug.strong is False

    def test_missing_keys_uses_defaults(self):
        cfg = {}
        aug = build_augmentation(cfg)
        assert aug.image_size == 224
        assert aug.strong is True
