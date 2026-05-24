"""Losses for BiGCL.

Three standard objectives applied to GNN-refined features:
1. Instance InfoNCE: forces diverse representations (feature-level, O(B²·D))
2. Cluster contrastive: forces meaningful clusters (column-wise, O(K²·B))
3. Entropy maximization: balanced cluster sizes (O(K))

Matches the CC (Contrastive Clustering, Li et al. AAAI 2021) loss exactly.
The novelty is in what FEATURES the losses operate on: GNN-refined
context-dependent features vs. CC's context-free MLP features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGCLLoss(nn.Module):
    """Combined loss for BiGCL, matching CC's exact formulation.

    Three complementary objectives:
    - InfoNCE: instance discrimination → diverse features (O(B²·D))
    - Cluster contrastive: cluster discrimination → meaningful groups (O(K²·B))
    - Marginal entropy: balanced cluster sizes (O(K))

    Applied to GNN-refined features (context-dependent) rather than
    raw MLP features (context-free, as in CC). This is the distinction.
    """

    def __init__(
        self,
        temperature: float = 0.5,
        w_instance: float = 1.0,
        w_cluster: float = 1.0,
        w_entropy: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.temperature = temperature
        self.w_instance = w_instance
        self.w_cluster = w_cluster
        self.w_entropy = w_entropy

    def forward(
        self,
        z_i1: torch.Tensor,
        z_i2: torch.Tensor,
        z_c1: torch.Tensor,
        z_c2: torch.Tensor,
    ) -> dict:
        """
        Args:
            z_i1, z_i2: (N, D) L2-normalized instance features from two views
            z_c1, z_c2: (N, K) softmax cluster assignments from two views
        Returns:
            dict with total loss and individual components
        """
        B = z_i1.shape[0]

        # --- Instance InfoNCE (same as CC) ---
        z = torch.cat([z_i1, z_i2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)
        pos_idx = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        l_instance = F.cross_entropy(sim, pos_idx)

        # --- Cluster contrastive (same as CC: asymmetric) ---
        c1 = F.normalize(z_c1.t(), dim=-1)  # (K, B)
        c2 = F.normalize(z_c2.t(), dim=-1)
        K = c1.shape[0]
        sim_c = torch.mm(c1, c2.t()) / self.temperature  # (K, K)
        l_cluster = F.cross_entropy(sim_c, torch.arange(K, device=sim_c.device))

        # --- Marginal entropy maximization (same as CC: only marginal) ---
        avg_c = (z_c1.mean(0) + z_c2.mean(0)) / 2
        avg_c = torch.clamp(avg_c, min=1e-8)
        l_entropy = -(avg_c * torch.log(avg_c)).sum()

        # --- Total: inst + cluster - w_entropy * entropy ---
        total = (self.w_instance * l_instance
                 + self.w_cluster * l_cluster
                 - self.w_entropy * l_entropy)

        return {
            "loss": total,
            "instance_loss": l_instance.item(),
            "cluster_loss": l_cluster.item(),
            "entropy": l_entropy.item(),
        }
