"""Unit tests for models/losses.py."""

import torch
import torch.nn.functional as F
import pytest

from models.losses import BiGCLLoss


class TestBiGCLLoss:
    @pytest.fixture
    def loss_fn(self):
        return BiGCLLoss(temperature=0.5, w_instance=1.0, w_cluster=1.0, w_entropy=0.5)

    @pytest.fixture
    def batch_data(self):
        B, D, K = 16, 128, 10
        z_i1 = F.normalize(torch.randn(B, D, requires_grad=True), dim=-1)
        z_i2 = F.normalize(torch.randn(B, D, requires_grad=True), dim=-1)
        z_c1 = F.softmax(torch.randn(B, K, requires_grad=True), dim=-1)
        z_c2 = F.softmax(torch.randn(B, K, requires_grad=True), dim=-1)
        return z_i1, z_i2, z_c1, z_c2

    def test_output_keys(self, loss_fn, batch_data):
        result = loss_fn(*batch_data)
        assert "loss" in result
        assert "instance_loss" in result
        assert "cluster_loss" in result
        assert "entropy" in result

    def test_loss_is_scalar(self, loss_fn, batch_data):
        result = loss_fn(*batch_data)
        assert result["loss"].ndim == 0

    def test_loss_has_grad_fn(self, loss_fn, batch_data):
        result = loss_fn(*batch_data)
        assert result["loss"].grad_fn is not None

    def test_components_are_floats(self, loss_fn, batch_data):
        result = loss_fn(*batch_data)
        assert isinstance(result["instance_loss"], float)
        assert isinstance(result["cluster_loss"], float)
        assert isinstance(result["entropy"], float)

    def test_instance_loss_positive(self, loss_fn, batch_data):
        result = loss_fn(*batch_data)
        assert result["instance_loss"] > 0

    def test_cluster_loss_positive(self, loss_fn, batch_data):
        result = loss_fn(*batch_data)
        assert result["cluster_loss"] > 0

    def test_entropy_non_negative(self, loss_fn, batch_data):
        result = loss_fn(*batch_data)
        assert result["entropy"] >= 0

    def test_perfect_instance_alignment_low_loss(self):
        """When views are identical, instance loss should be lower."""
        loss_fn = BiGCLLoss(temperature=0.5)
        B, D, K = 8, 64, 5
        z_i = F.normalize(torch.randn(B, D), dim=-1)
        z_c = F.softmax(torch.randn(B, K), dim=-1)

        # Same instance features for both views
        result_same = loss_fn(z_i, z_i, z_c, z_c)
        # Different instance features
        z_i2 = F.normalize(torch.randn(B, D), dim=-1)
        result_diff = loss_fn(z_i, z_i2, z_c, z_c)

        assert result_same["instance_loss"] < result_diff["instance_loss"]

    def test_different_temperatures(self):
        B, D, K = 8, 64, 5
        z_i1 = F.normalize(torch.randn(B, D), dim=-1)
        z_i2 = F.normalize(torch.randn(B, D), dim=-1)
        z_c1 = F.softmax(torch.randn(B, K), dim=-1)
        z_c2 = F.softmax(torch.randn(B, K), dim=-1)

        loss_low_t = BiGCLLoss(temperature=0.1)
        loss_high_t = BiGCLLoss(temperature=1.0)

        r_low = loss_low_t(z_i1, z_i2, z_c1, z_c2)
        r_high = loss_high_t(z_i1, z_i2, z_c1, z_c2)

        # Both should produce valid losses
        assert r_low["loss"].isfinite()
        assert r_high["loss"].isfinite()

    def test_weight_scaling(self):
        B, D, K = 8, 64, 5
        z_i1 = F.normalize(torch.randn(B, D), dim=-1)
        z_i2 = F.normalize(torch.randn(B, D), dim=-1)
        z_c1 = F.softmax(torch.randn(B, K), dim=-1)
        z_c2 = F.softmax(torch.randn(B, K), dim=-1)

        loss_1x = BiGCLLoss(w_instance=1.0, w_cluster=1.0, w_entropy=0.0)
        loss_2x = BiGCLLoss(w_instance=2.0, w_cluster=2.0, w_entropy=0.0)

        r_1x = loss_1x(z_i1, z_i2, z_c1, z_c2)
        r_2x = loss_2x(z_i1, z_i2, z_c1, z_c2)

        assert r_2x["loss"].item() == pytest.approx(2 * r_1x["loss"].item(), rel=1e-4)

    def test_backward_pass(self, loss_fn):
        B, D, K = 8, 64, 5
        raw_i1 = torch.randn(B, D, requires_grad=True)
        raw_i2 = torch.randn(B, D, requires_grad=True)
        raw_c1 = torch.randn(B, K, requires_grad=True)
        raw_c2 = torch.randn(B, K, requires_grad=True)

        z_i1 = F.normalize(raw_i1, dim=-1)
        z_i2 = F.normalize(raw_i2, dim=-1)
        z_c1 = F.softmax(raw_c1, dim=-1)
        z_c2 = F.softmax(raw_c2, dim=-1)

        result = loss_fn(z_i1, z_i2, z_c1, z_c2)
        result["loss"].backward()
        # Gradients flow to leaf tensors
        assert raw_i1.grad is not None
        assert raw_c1.grad is not None
