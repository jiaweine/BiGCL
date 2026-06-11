"""Clustering evaluation metrics: ACC, NMI, ARI.

Implements standard unsupervised clustering evaluation using the
Hungarian algorithm for optimal label matching.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def _build_cost_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Build the assignment cost matrix between true and predicted labels.

    Args:
        y_true: (N,) ground-truth labels (int64)
        y_pred: (N,) predicted cluster assignments (int64)
    Returns:
        (C, C) cost matrix where C = max(y_true.max(), y_pred.max()) + 1
    """
    n_classes = max(y_true.max(), y_pred.max()) + 1
    cost_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cost_matrix[t, p] += 1
    return cost_matrix


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute clustering accuracy using the Hungarian algorithm.

    Finds the optimal one-to-one mapping between predicted cluster IDs
    and ground-truth labels that maximizes accuracy.

    Args:
        y_true: (N,) ground-truth labels
        y_pred: (N,) predicted cluster assignments
    Returns:
        accuracy in [0, 1]
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape, "Shape mismatch"

    cost_matrix = _build_cost_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    accuracy = cost_matrix[row_ind, col_ind].sum() / y_true.shape[0]
    return float(accuracy)


def normalized_mutual_info(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Normalized Mutual Information (NMI)."""
    return float(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"))


def adjusted_rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Adjusted Rand Index (ARI)."""
    return float(adjusted_rand_score(y_true, y_pred))


def hungarian_match(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = None) -> dict:
    """Return optimal pred_label → true_label mapping via Hungarian algorithm.

    Args:
        y_true: (N,) ground-truth labels
        y_pred: (N,) predicted cluster assignments
        n_classes: number of classes (inferred if None)
    Returns:
        dict mapping pred_label (int) → true_label (int)
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cost = _build_cost_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cost)
    return {int(col): int(row) for row, col in zip(row_ind, col_ind)}


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all clustering metrics.

    Args:
        y_true: (N,) ground-truth labels
        y_pred: (N,) predicted cluster assignments
    Returns:
        dict with ACC, NMI, ARI
    """
    return {
        "ACC": clustering_accuracy(y_true, y_pred),
        "NMI": normalized_mutual_info(y_true, y_pred),
        "ARI": adjusted_rand_index(y_true, y_pred),
    }
