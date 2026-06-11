"""Unit tests for models/clustering_head.py."""

import torch
import torch.nn.functional as F
import pytest

from models.clustering_head import ProgressiveSelfTrainer


class TestProgressiveSelfTrainer:
    @pytest.fixture
    def trainer(self):
        return ProgressiveSelfTrainer(
            initial_threshold=0.9,
            final_threshold=0.6,
            warmup_epochs=10,
            total_epochs=100,
        )

    def test_threshold_during_warmup(self, trainer):
        for epoch in range(10):
            assert trainer.get_threshold(epoch) == 0.9

    def test_threshold_after_warmup_decreases(self, trainer):
        t_start = trainer.get_threshold(10)
        t_mid = trainer.get_threshold(55)
        t_end = trainer.get_threshold(100)
        assert t_start >= t_mid >= t_end

    def test_threshold_final_value(self, trainer):
        t = trainer.get_threshold(100)
        assert t == pytest.approx(0.6)

    def test_threshold_at_warmup_boundary(self, trainer):
        assert trainer.get_threshold(10) == pytest.approx(0.9)

    def test_threshold_beyond_total_epochs(self, trainer):
        """Should clamp at final threshold."""
        t = trainer.get_threshold(200)
        assert t == pytest.approx(0.6)

    def test_update_pseudo_labels_shape(self, trainer):
        N, K = 32, 10
        assignments = F.softmax(torch.randn(N, K), dim=-1)
        result = trainer.update_pseudo_labels(assignments, epoch=50)
        assert result["pseudo_labels"].shape == (N,)
        assert result["mask"].shape == (N,)

    def test_update_pseudo_labels_keys(self, trainer):
        assignments = F.softmax(torch.randn(16, 5), dim=-1)
        result = trainer.update_pseudo_labels(assignments, epoch=0)
        assert "pseudo_labels" in result
        assert "mask" in result
        assert "threshold" in result
        assert "coverage" in result

    def test_coverage_range(self, trainer):
        assignments = F.softmax(torch.randn(100, 10), dim=-1)
        result = trainer.update_pseudo_labels(assignments, epoch=50)
        assert 0.0 <= result["coverage"] <= 1.0

    def test_high_confidence_passes_threshold(self, trainer):
        """When all assignments are very confident, coverage should be high."""
        N, K = 32, 5
        # Create nearly one-hot assignments
        logits = torch.zeros(N, K)
        logits[:, 0] = 10.0  # Very confident
        assignments = F.softmax(logits, dim=-1)
        result = trainer.update_pseudo_labels(assignments, epoch=50)
        assert result["coverage"] > 0.9

    def test_low_confidence_fails_threshold(self, trainer):
        """When assignments are uniform, coverage should be low."""
        N, K = 32, 10
        # Uniform assignments (max confidence ~= 1/K)
        assignments = torch.ones(N, K) / K
        result = trainer.update_pseudo_labels(assignments, epoch=0)
        assert result["coverage"] == 0.0


class TestSelfTrainingLoss:
    @pytest.fixture
    def trainer(self):
        return ProgressiveSelfTrainer()

    def test_zero_loss_when_no_confident(self, trainer):
        N, K = 16, 5
        logits2 = torch.randn(N, K)
        pseudo_labels = torch.zeros(N, dtype=torch.long)
        mask = torch.zeros(N, dtype=torch.bool)  # No confident samples
        loss = trainer.self_training_loss(logits2, pseudo_labels, mask)
        assert loss.item() == 0.0

    def test_nonzero_loss_with_confident(self, trainer):
        N, K = 16, 5
        logits2 = torch.randn(N, K)
        pseudo_labels = torch.randint(0, K, (N,))
        mask = torch.ones(N, dtype=torch.bool)
        loss = trainer.self_training_loss(logits2, pseudo_labels, mask)
        assert loss.item() > 0.0

    def test_soft_target_loss(self, trainer):
        N, K = 16, 5
        logits2 = torch.randn(N, K)
        pseudo_labels = torch.randint(0, K, (N,))
        mask = torch.ones(N, dtype=torch.bool)
        soft_targets = F.softmax(torch.randn(N, K), dim=-1)
        loss = trainer.self_training_loss(
            logits2, pseudo_labels, mask, soft_targets=soft_targets
        )
        assert loss.item() >= 0.0

    def test_sharpen_temperature_effect(self, trainer):
        """Lower sharpen_temp should produce sharper targets and potentially different loss."""
        N, K = 32, 5
        logits2 = torch.randn(N, K)
        pseudo_labels = torch.randint(0, K, (N,))
        mask = torch.ones(N, dtype=torch.bool)
        soft_targets = F.softmax(torch.randn(N, K), dim=-1)

        loss_sharp = trainer.self_training_loss(
            logits2, pseudo_labels, mask, soft_targets=soft_targets, sharpen_temp=0.1
        )
        loss_smooth = trainer.self_training_loss(
            logits2, pseudo_labels, mask, soft_targets=soft_targets, sharpen_temp=2.0
        )
        # Both should be valid
        assert loss_sharp.isfinite()
        assert loss_smooth.isfinite()

    def test_backward_pass(self, trainer):
        N, K = 16, 5
        logits2 = torch.randn(N, K, requires_grad=True)
        pseudo_labels = torch.randint(0, K, (N,))
        mask = torch.ones(N, dtype=torch.bool)
        loss = trainer.self_training_loss(logits2, pseudo_labels, mask)
        loss.backward()
        assert logits2.grad is not None
