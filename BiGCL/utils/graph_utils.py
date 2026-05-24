"""Graph construction and augmentation utilities for BiGCL.

Provides functions for building bipartite graphs, computing affinity matrices,
and applying stochastic graph augmentations.
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_affinity(
    img_feat: torch.Tensor,
    proto_feat: torch.Tensor,
    temperature: float = 0.1,
    top_k: int = 5,
) -> torch.Tensor:
    """Compute sparse affinity matrix between images and prototypes.

    Args:
        img_feat: (N, D) L2-normalized image features
        proto_feat: (K, D) L2-normalized prototype features
        temperature: softmax temperature
        top_k: number of top connections per image
    Returns:
        (N, K) sparse soft affinity matrix
    """
    img_feat = F.normalize(img_feat, dim=-1)
    proto_feat = F.normalize(proto_feat, dim=-1)

    sim = torch.mm(img_feat, proto_feat.t())  # (N, K)
    K = proto_feat.shape[0]
    k = min(top_k, K)

    topk_vals, topk_idx = sim.topk(k, dim=-1)
    mask = torch.zeros_like(sim)
    mask.scatter_(-1, topk_idx, 1.0)

    sim_masked = sim * mask + (1 - mask) * (-1e9)
    affinity = F.softmax(sim_masked / temperature, dim=-1) * mask

    return affinity


def build_bipartite_graph(
    img_feat: torch.Tensor,
    proto_feat: torch.Tensor,
    temperature: float = 0.1,
    top_k: int = 5,
) -> dict:
    """Build bipartite graph structure from features.

    Args:
        img_feat: (N, D) image features
        proto_feat: (K, D) prototype features
        temperature: softmax temperature for edge weights
        top_k: sparsity level
    Returns:
        dict with adjacency, edge_index, edge_weight
    """
    affinity = compute_affinity(img_feat, proto_feat, temperature, top_k)

    # Convert to edge_index format (COO)
    nonzero = affinity.nonzero(as_tuple=False)  # (E, 2)
    edge_index = nonzero.t()  # (2, E)
    edge_weight = affinity[nonzero[:, 0], nonzero[:, 1]]

    return {
        "adj": affinity,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "n_images": img_feat.shape[0],
        "n_protos": proto_feat.shape[0],
    }


def augment_graph(
    adj: torch.Tensor,
    feat: torch.Tensor,
    edge_drop_rate: float = 0.1,
    feat_mask_rate: float = 0.1,
) -> tuple:
    """Apply stochastic augmentation to graph structure and features.

    Args:
        adj: (N, K) adjacency matrix
        feat: (N, D) node features
        edge_drop_rate: probability of dropping each edge
        feat_mask_rate: probability of masking each feature dimension
    Returns:
        (aug_adj, aug_feat) augmented adjacency and features
    """
    # Edge dropout
    if edge_drop_rate > 0:
        edge_mask = torch.bernoulli(
            torch.full_like(adj, 1.0 - edge_drop_rate)
        )
        aug_adj = adj * edge_mask
        row_sum = aug_adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        aug_adj = aug_adj / row_sum
    else:
        aug_adj = adj

    # Feature masking
    if feat_mask_rate > 0:
        feat_mask = torch.bernoulli(
            torch.full(feat.shape[-1:], 1.0 - feat_mask_rate, device=feat.device)
        )
        aug_feat = feat * feat_mask.unsqueeze(0) / (1.0 - feat_mask_rate + 1e-8)
    else:
        aug_feat = feat

    return aug_adj, aug_feat


def knn_graph(features: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Build k-nearest-neighbor graph from features.

    Args:
        features: (N, D) feature matrix
        k: number of nearest neighbors
    Returns:
        (N, N) adjacency matrix
    """
    features = F.normalize(features, dim=-1)
    sim = torch.mm(features, features.t())

    # Zero out self-connections
    sim.fill_diagonal_(float("-inf"))

    topk_vals, topk_idx = sim.topk(k, dim=-1)
    adj = torch.zeros_like(sim)
    adj.scatter_(-1, topk_idx, 1.0)

    # Symmetrize
    adj = (adj + adj.t()).clamp(max=1.0)
    return adj
