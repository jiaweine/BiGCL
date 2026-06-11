"""Unit tests for utils/graph_utils.py."""

import torch
import pytest

from utils.graph_utils import compute_affinity, build_bipartite_graph, augment_graph, knn_graph


class TestComputeAffinity:
    def test_output_shape(self):
        img_feat = torch.randn(16, 64)
        proto_feat = torch.randn(5, 64)
        aff = compute_affinity(img_feat, proto_feat, temperature=0.1, top_k=3)
        assert aff.shape == (16, 5)

    def test_non_negative(self):
        img_feat = torch.randn(10, 32)
        proto_feat = torch.randn(4, 32)
        aff = compute_affinity(img_feat, proto_feat)
        assert (aff >= 0).all()

    def test_sparsity(self):
        """Each row should have at most top_k nonzero entries."""
        img_feat = torch.randn(8, 64)
        proto_feat = torch.randn(10, 64)
        top_k = 3
        aff = compute_affinity(img_feat, proto_feat, top_k=top_k)
        nonzero_per_row = (aff > 0).sum(dim=-1)
        assert (nonzero_per_row <= top_k).all()

    def test_top_k_larger_than_K(self):
        """When top_k > K, all entries can be nonzero."""
        img_feat = torch.randn(8, 32)
        proto_feat = torch.randn(3, 32)
        aff = compute_affinity(img_feat, proto_feat, top_k=10)
        assert aff.shape == (8, 3)

    def test_rows_sum_to_one_approximately(self):
        """Nonzero rows should sum to approximately 1 (softmax over selected)."""
        img_feat = torch.randn(16, 64)
        proto_feat = torch.randn(8, 64)
        aff = compute_affinity(img_feat, proto_feat, top_k=4)
        row_sums = aff.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01)


class TestBuildBipartiteGraph:
    def test_output_keys(self):
        img_feat = torch.randn(10, 32)
        proto_feat = torch.randn(4, 32)
        result = build_bipartite_graph(img_feat, proto_feat)
        assert "adj" in result
        assert "edge_index" in result
        assert "edge_weight" in result
        assert "n_images" in result
        assert "n_protos" in result

    def test_dimensions(self):
        N, K, D = 12, 5, 64
        img_feat = torch.randn(N, D)
        proto_feat = torch.randn(K, D)
        result = build_bipartite_graph(img_feat, proto_feat)
        assert result["adj"].shape == (N, K)
        assert result["n_images"] == N
        assert result["n_protos"] == K
        assert result["edge_index"].shape[0] == 2

    def test_edge_weight_positive(self):
        img_feat = torch.randn(8, 32)
        proto_feat = torch.randn(4, 32)
        result = build_bipartite_graph(img_feat, proto_feat)
        assert (result["edge_weight"] > 0).all()

    def test_edge_index_bounds(self):
        N, K = 10, 5
        img_feat = torch.randn(N, 32)
        proto_feat = torch.randn(K, 32)
        result = build_bipartite_graph(img_feat, proto_feat)
        assert result["edge_index"][0].max() < N
        assert result["edge_index"][1].max() < K


class TestAugmentGraph:
    def test_no_augmentation(self):
        adj = torch.rand(8, 4)
        feat = torch.randn(8, 32)
        aug_adj, aug_feat = augment_graph(adj, feat, edge_drop_rate=0.0, feat_mask_rate=0.0)
        assert torch.allclose(aug_adj, adj)
        assert torch.allclose(aug_feat, feat)

    def test_edge_dropout_changes_adj(self):
        torch.manual_seed(0)
        adj = torch.ones(8, 4)
        feat = torch.randn(8, 32)
        aug_adj, _ = augment_graph(adj, feat, edge_drop_rate=0.5, feat_mask_rate=0.0)
        # Some edges should be dropped
        assert not torch.allclose(aug_adj, adj)

    def test_feat_mask_changes_feat(self):
        torch.manual_seed(0)
        adj = torch.rand(8, 4)
        feat = torch.ones(8, 32)
        _, aug_feat = augment_graph(adj, feat, edge_drop_rate=0.0, feat_mask_rate=0.5)
        # Some features should be masked (zeroed)
        assert (aug_feat == 0).any()

    def test_output_shapes(self):
        adj = torch.rand(10, 5)
        feat = torch.randn(10, 64)
        aug_adj, aug_feat = augment_graph(adj, feat, edge_drop_rate=0.2, feat_mask_rate=0.1)
        assert aug_adj.shape == adj.shape
        assert aug_feat.shape == feat.shape

    def test_edge_dropout_row_normalization(self):
        """After edge dropout, rows should sum to ~1 (re-normalized)."""
        torch.manual_seed(42)
        adj = torch.ones(8, 4) / 4  # Normalized
        feat = torch.randn(8, 32)
        aug_adj, _ = augment_graph(adj, feat, edge_drop_rate=0.3, feat_mask_rate=0.0)
        row_sums = aug_adj.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01)


class TestKnnGraph:
    def test_output_shape(self):
        features = torch.randn(20, 64)
        adj = knn_graph(features, k=5)
        assert adj.shape == (20, 20)

    def test_symmetric(self):
        features = torch.randn(15, 32)
        adj = knn_graph(features, k=3)
        assert torch.allclose(adj, adj.t())

    def test_no_self_loops(self):
        features = torch.randn(10, 16)
        adj = knn_graph(features, k=3)
        assert adj.diag().sum() == 0

    def test_binary_entries(self):
        features = torch.randn(12, 32)
        adj = knn_graph(features, k=4)
        assert ((adj == 0) | (adj == 1)).all()

    def test_min_connections(self):
        """Each node should have at least k connections (before symmetrization)."""
        features = torch.randn(20, 64)
        k = 5
        adj = knn_graph(features, k=k)
        # After symmetrization, each node has >= k connections
        assert (adj.sum(dim=-1) >= k).all()
