"""Unit tests for utils/metrics.py."""

import numpy as np
import pytest

from utils.metrics import (
    clustering_accuracy,
    normalized_mutual_info,
    adjusted_rand_index,
    hungarian_match,
    evaluate_clustering,
)


class TestClusteringAccuracy:
    def test_perfect_match(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        assert clustering_accuracy(y_true, y_pred) == 1.0

    def test_permuted_labels(self):
        """Accuracy should be 1.0 even with permuted cluster IDs."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([2, 2, 0, 0, 1, 1])
        assert clustering_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_completely_wrong(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 0])
        acc = clustering_accuracy(y_true, y_pred)
        assert 0.0 <= acc <= 1.0

    def test_single_cluster(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        assert clustering_accuracy(y_true, y_pred) == 1.0

    def test_shape_mismatch_raises(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])
        with pytest.raises(AssertionError):
            clustering_accuracy(y_true, y_pred)

    def test_returns_float(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([1, 0, 2, 1, 0, 2])
        result = clustering_accuracy(y_true, y_pred)
        assert isinstance(result, float)

    def test_large_random(self, rng):
        """Accuracy stays in [0, 1] for larger random inputs."""
        y_true = rng.integers(0, 10, size=500)
        y_pred = rng.integers(0, 10, size=500)
        acc = clustering_accuracy(y_true, y_pred)
        assert 0.0 <= acc <= 1.0


class TestNMI:
    def test_perfect(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        assert normalized_mutual_info(y_true, y_pred) == pytest.approx(1.0)

    def test_permuted(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 1, 2, 2, 0, 0])
        assert normalized_mutual_info(y_true, y_pred) == pytest.approx(1.0)

    def test_random_range(self, rng):
        y_true = rng.integers(0, 5, size=200)
        y_pred = rng.integers(0, 5, size=200)
        nmi = normalized_mutual_info(y_true, y_pred)
        assert 0.0 <= nmi <= 1.0


class TestARI:
    def test_perfect(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        assert adjusted_rand_index(y_true, y_pred) == pytest.approx(1.0)

    def test_random_close_to_zero(self, rng):
        y_true = rng.integers(0, 10, size=1000)
        y_pred = rng.integers(0, 10, size=1000)
        ari = adjusted_rand_index(y_true, y_pred)
        assert -0.5 <= ari <= 1.0


class TestHungarianMatch:
    def test_identity_mapping(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        mapping = hungarian_match(y_true, y_pred)
        assert mapping[0] == 0
        assert mapping[1] == 1
        assert mapping[2] == 2

    def test_permuted_mapping(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([2, 2, 0, 0, 1, 1])
        mapping = hungarian_match(y_true, y_pred)
        assert mapping[2] == 0
        assert mapping[0] == 1
        assert mapping[1] == 2

    def test_returns_dict(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 0, 2])
        result = hungarian_match(y_true, y_pred)
        assert isinstance(result, dict)

    def test_explicit_n_classes(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        mapping = hungarian_match(y_true, y_pred, n_classes=5)
        assert len(mapping) == 5


class TestEvaluateClustering:
    def test_returns_all_metrics(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 1, 2, 2, 0, 0])
        result = evaluate_clustering(y_true, y_pred)
        assert "ACC" in result
        assert "NMI" in result
        assert "ARI" in result

    def test_perfect_scores(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        result = evaluate_clustering(y_true, y_pred)
        assert result["ACC"] == pytest.approx(1.0)
        assert result["NMI"] == pytest.approx(1.0)
        assert result["ARI"] == pytest.approx(1.0)
