"""Unit tests for models/bigcl.py (skip_backbone=True to avoid CLIP loading)."""

import torch
import torch.nn.functional as F
import pytest

from models.bigcl import BiGCL


@pytest.fixture
def model():
    """Create BiGCL model without loading CLIP backbone."""
    return BiGCL(
        n_clusters=10,
        proj_dim=128,
        clip_dim=512,
        skip_backbone=True,
        feat_drop_rate=0.3,
        w_ortho=0.5,
        w_entropy=3.0,
        use_prototype=True,
    )


@pytest.fixture
def model_no_proto():
    """Create BiGCL model without prototypes (CC-style linear head)."""
    return BiGCL(
        n_clusters=10,
        proj_dim=128,
        clip_dim=512,
        skip_backbone=True,
        use_prototype=False,
    )


class TestBiGCLForward:
    def test_forward_features_output_keys(self, model):
        model.train()
        x = torch.randn(8, 512)
        out = model.forward_features(x)
        assert "loss" in out
        assert "instance_loss" in out
        assert "cluster_loss" in out
        assert "entropy" in out
        assert "ortho_loss" in out
        assert "assignments" in out
        assert "hard_labels" in out
        assert "features" in out
        assert "logits" in out

    def test_forward_features_shapes(self, model):
        model.train()
        B = 16
        x = torch.randn(B, 512)
        out = model.forward_features(x)
        assert out["assignments"].shape == (B, 10)
        assert out["hard_labels"].shape == (B,)
        assert out["features"].shape == (B, 128)
        assert out["logits"].shape == (B, 10)

    def test_loss_is_finite(self, model):
        model.train()
        x = torch.randn(8, 512)
        out = model.forward_features(x)
        assert out["loss"].isfinite()

    def test_backward_pass(self, model):
        model.train()
        x = torch.randn(8, 512)
        out = model.forward_features(x)
        out["loss"].backward()
        # Check that trainable parameters have gradients
        for p in model.instance_head.parameters():
            assert p.grad is not None

    def test_assignments_are_probabilities(self, model):
        model.eval()
        x = torch.randn(8, 512)
        with torch.no_grad():
            out = model.forward_features(x)
        assignments = out["assignments"]
        assert (assignments >= 0).all()
        assert torch.allclose(assignments.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_hard_labels_in_range(self, model):
        model.eval()
        x = torch.randn(8, 512)
        with torch.no_grad():
            out = model.forward_features(x)
        assert (out["hard_labels"] >= 0).all()
        assert (out["hard_labels"] < 10).all()


class TestBiGCLNoPrototype:
    def test_forward_features_works(self, model_no_proto):
        model_no_proto.train()
        x = torch.randn(8, 512)
        out = model_no_proto.forward_features(x)
        assert "loss" in out
        assert out["ortho_loss"] == 0.0

    def test_assignments_shape(self, model_no_proto):
        model_no_proto.eval()
        x = torch.randn(8, 512)
        with torch.no_grad():
            out = model_no_proto.forward_features(x)
        assert out["assignments"].shape == (8, 10)


class TestOrthoLoss:
    def test_ortho_loss_range(self, model):
        loss = model._ortho_loss()
        assert loss.item() >= 0.0

    def test_ortho_loss_zero_for_orthogonal(self):
        """When prototypes are perfectly orthogonal, loss should be ~0."""
        model = BiGCL(
            n_clusters=5,
            proj_dim=128,
            clip_dim=512,
            skip_backbone=True,
            use_prototype=True,
            ortho_init=True,
        )
        loss = model._ortho_loss()
        # Orthogonal init should give near-zero ortho loss
        assert loss.item() < 0.1

    def test_ortho_loss_high_for_identical(self):
        """When prototypes are identical, loss should be high."""
        model = BiGCL(
            n_clusters=5,
            proj_dim=128,
            clip_dim=512,
            skip_backbone=True,
            use_prototype=True,
        )
        # Set all prototypes to be identical
        with torch.no_grad():
            model.prototypes.copy_(torch.ones(5, 128))
        loss = model._ortho_loss()
        assert loss.item() > 0.5


class TestExtractFromFeatures:
    def test_output_keys(self, model):
        model.eval()
        x = torch.randn(8, 512)
        out = model.extract_from_features(x)
        assert "features" in out
        assert "assignments" in out
        assert "hard_labels" in out

    def test_output_shapes(self, model):
        model.eval()
        B = 16
        x = torch.randn(B, 512)
        out = model.extract_from_features(x)
        assert out["features"].shape == (B, 128)
        assert out["assignments"].shape == (B, 10)
        assert out["hard_labels"].shape == (B,)

    def test_deterministic(self, model):
        model.eval()
        x = torch.randn(8, 512)
        out1 = model.extract_from_features(x)
        out2 = model.extract_from_features(x)
        assert torch.allclose(out1["assignments"], out2["assignments"])


class TestGetTrainableParams:
    def test_returns_list(self, model):
        groups = model.get_trainable_params()
        assert isinstance(groups, list)

    def test_groups_have_required_keys(self, model):
        groups = model.get_trainable_params()
        for g in groups:
            assert "params" in g
            assert "lr_scale" in g
            assert "name" in g

    def test_includes_expected_groups(self, model):
        groups = model.get_trainable_params()
        names = [g["name"] for g in groups]
        assert "instance_head" in names
        assert "cluster_proj" in names
        assert "prototypes" in names
        assert "log_tau" in names

    def test_shared_path_no_duplicate(self):
        model = BiGCL(
            n_clusters=5,
            proj_dim=64,
            clip_dim=512,
            skip_backbone=True,
            shared_path=True,
        )
        groups = model.get_trainable_params()
        names = [g["name"] for g in groups]
        assert "cluster_proj" not in names


class TestTauProperty:
    def test_tau_positive(self, model):
        assert model.tau.item() > 0

    def test_tau_clamped(self):
        model = BiGCL(
            n_clusters=5,
            proj_dim=64,
            clip_dim=512,
            skip_backbone=True,
        )
        # Set log_tau to very negative value
        with torch.no_grad():
            model.log_tau.fill_(-100.0)
        assert model.tau.item() >= 0.01 - 1e-6
