"""BiGCL: Bipartite Graph Contrastive Learning for Image Clustering.

Architecture (prototype-anchored contrastive clustering):
    CLIP features (frozen, 512-dim)
        ↓  dropout (two views)
        ├── Instance head (512→D→D) → L2-norm → InfoNCE
        └── Cluster proj (512→D→D) → L2-norm ──┐
                                                  ├→ cos_sim / τ → softmax
                          Prototypes (K × D) ────┘
                          + Orthogonality regularization

Loss = CC losses (InfoNCE + cluster contrastive + entropy) + prototype ortho.

Single innovation: explicit, orthogonality-regularized prototypes replace the
MLP cluster head. Cluster assignment = cosine similarity to prototypes on the
unit hypersphere, giving assignments direct geometric meaning and forcing
maximum cluster separation. No cross-attention, no extra MLP head.

Complexity: O(B²·D + B·K·D) — dominated by InfoNCE (B²·D).
Parameters: ~397K vs ~730K (old BiGCL) for D=256, K=10. 45% reduction.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_backbone import CLIPBackbone
from .losses import BiGCLLoss


class BiGCL(nn.Module):
    """Bipartite Graph Contrastive Learning for Image Clustering.

    Dual-path architecture with prototype-anchored cluster assignment:
    - Instance path: MLP → L2-norm → InfoNCE (CC-identical)
    - Cluster path: MLP → L2-norm → cosine similarity with orthogonal prototypes

    The prototypes are K learnable vectors on the unit hypersphere, each
    representing a cluster center. Cluster assignment is computed as
    softmax(cos_sim / τ), giving direct geometric meaning. An orthogonality
    loss forces maximum angular separation between prototypes.

    Training: Adam optimizer, constant LR, dropout augmentation on raw features.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        n_clusters: int = 10,
        proj_dim: int = 256,
        feat_drop_rate: float = 0.3,
        w_entropy: float = 5.0,
        w_ortho: float = 1.0,
        w_instance: float = 1.0,
        w_cluster: float = 1.0,
        shared_path: bool = False,
        use_prototype: bool = True,
        clip_dim: int = 512,
        skip_backbone: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.proj_dim = proj_dim
        self.feat_drop_rate = feat_drop_rate
        self.w_ortho = w_ortho

        if not skip_backbone:
            # CLIP backbone (frozen — only used for raw feature extraction)
            # Save RNG state: CLIPBackbone loading consumes random state,
            # which would make trainable layer init seed-dependent on CLIP internals.
            _rng_cpu = torch.get_rng_state()
            _rng_gpu = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            self.backbone = CLIPBackbone(
                model_name=clip_model_name,
                pretrained=clip_pretrained,
                proj_dim=proj_dim,
                freeze_visual=True,
                freeze_text=True,
            )
            # Restore RNG state so trainable layers get deterministic init
            torch.set_rng_state(_rng_cpu)
            if _rng_gpu is not None:
                torch.cuda.set_rng_state(_rng_gpu)

            clip_dim = self.backbone.clip_dim  # 512

            # Freeze CLIPBackbone's built-in projections (we don't use them)
            for p in self.backbone.visual_proj.parameters():
                p.requires_grad = False
            for p in self.backbone.text_proj.parameters():
                p.requires_grad = False
        else:
            self.backbone = None

        # === Instance path (CC-identical) ===
        self.instance_head = nn.Sequential(
            nn.Linear(clip_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # === Cluster path: projection + prototype cosine similarity ===
        if shared_path:
            # Ablation: shared MLP for both instance and cluster
            self.cluster_proj = self.instance_head
        else:
            self.cluster_proj = nn.Sequential(
                nn.Linear(clip_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            )

        # Learnable cluster prototypes (orthogonal initialization)
        self.use_prototype = use_prototype
        self.ortho_init = kwargs.get("ortho_init", True)
        if use_prototype:
            proto_init = torch.empty(n_clusters, proj_dim)
            if self.ortho_init:
                nn.init.orthogonal_(proto_init)
            else:
                nn.init.normal_(proto_init, std=0.02)
            self.prototypes = nn.Parameter(proto_init)
        else:
            # CC-style linear cluster head (ablation: no prototype)
            self.cluster_linear = nn.Linear(proj_dim, n_clusters)

        # Learnable temperature for prototype similarity
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.1)))

        # Loss: CC's exact formulation (InfoNCE + cluster contrastive + entropy)
        self.criterion = BiGCLLoss(w_entropy=w_entropy, w_instance=w_instance, w_cluster=w_cluster)

    @property
    def tau(self):
        """Temperature for prototype cosine similarity (always positive)."""
        return self.log_tau.exp().clamp(min=0.01)

    def _ortho_loss(self) -> torch.Tensor:
        """Prototype orthogonality regularization.

        Penalizes off-diagonal entries of the prototype Gram matrix,
        forcing maximum angular separation on the unit hypersphere.

        Returns:
            scalar loss ∈ [0, 1]
        """
        p = F.normalize(self.prototypes, dim=-1)            # (K, D)
        gram = p @ p.t()                                     # (K, K)
        I = torch.eye(self.n_clusters, device=gram.device)
        off_diag = gram - I
        return (off_diag ** 2).sum() / (self.n_clusters * (self.n_clusters - 1))

    def forward(self, images: torch.Tensor) -> dict:
        """Full forward pass from images."""
        img_feat_raw = self.backbone.encode_image_raw(images)
        return self.forward_features(img_feat_raw)

    def forward_features(self, img_feat_raw: torch.Tensor) -> dict:
        """Forward pass using pre-extracted CLIP features.

        Two views via dropout on raw CLIP features (same as CC).
        Instance path and cluster path process each view independently.

        Args:
            img_feat_raw: (B, clip_dim) raw CLIP features
        Returns:
            dict with loss, assignments, features, etc.
        """
        # Two views via dropout on raw features
        x1 = F.dropout(img_feat_raw, p=self.feat_drop_rate, training=self.training)
        x2 = F.dropout(img_feat_raw, p=self.feat_drop_rate, training=self.training)

        # Instance path: CC-identical
        z_i1 = F.normalize(self.instance_head(x1), dim=-1)
        z_i2 = F.normalize(self.instance_head(x2), dim=-1)

        # Cluster path: prototype-anchored assignment
        h1 = F.normalize(self.cluster_proj(x1), dim=-1)     # (B, D)
        h2 = F.normalize(self.cluster_proj(x2), dim=-1)     # (B, D)

        if self.use_prototype:
            p = F.normalize(self.prototypes, dim=-1)             # (K, D)
            tau = self.tau
            logits1 = h1 @ p.t() / tau                          # (B, K)
            logits2 = h2 @ p.t() / tau                          # (B, K)
        else:
            # CC-style: linear head on normalized embeddings
            logits1 = self.cluster_linear(h1)
            logits2 = self.cluster_linear(h2)

        z_c1 = F.softmax(logits1, dim=-1)
        z_c2 = F.softmax(logits2, dim=-1)

        # CC losses + prototype orthogonality
        loss_dict = self.criterion(z_i1, z_i2, z_c1, z_c2)
        if self.use_prototype:
            l_ortho = self._ortho_loss()
            loss_dict["loss"] = loss_dict["loss"] + self.w_ortho * l_ortho
            loss_dict["ortho_loss"] = l_ortho.item()
        else:
            loss_dict["ortho_loss"] = 0.0

        # Outputs for evaluation
        loss_dict["logits"] = logits1.detach()
        loss_dict["logits2"] = logits2  # keep grad for cross-view ST
        loss_dict["h2"] = h2.detach()  # normalized cluster embeddings (view2) for EMA ST
        loss_dict["assignments"] = z_c1.detach()
        loss_dict["hard_labels"] = z_c1.argmax(dim=-1)
        loss_dict["features"] = h1.detach()

        return loss_dict

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> dict:
        """Extract features and cluster assignments for evaluation."""
        self.eval()
        img_feat_raw = self.backbone.encode_image_raw(images)
        return self.extract_from_features(img_feat_raw)

    @torch.no_grad()
    def extract_from_features(self, img_feat_raw: torch.Tensor) -> dict:
        """Extract cluster assignments from pre-cached CLIP features."""
        h = F.normalize(self.cluster_proj(img_feat_raw), dim=-1)
        if self.use_prototype:
            p = F.normalize(self.prototypes, dim=-1)
            logits = h @ p.t() / self.tau
        else:
            logits = self.cluster_linear(h)
        z_c = F.softmax(logits, dim=-1)
        return {
            "features": h,
            "assignments": z_c,
            "hard_labels": z_c.argmax(dim=-1),
        }

    def get_trainable_params(self) -> list:
        """Get parameter groups with per-group learning rate scales."""
        groups = [
            {"params": list(self.instance_head.parameters()), "lr_scale": 1.0, "name": "instance_head"},
        ]
        # When shared_path=True, cluster_proj IS instance_head — don't register twice
        if self.cluster_proj is not self.instance_head:
            groups.append({"params": list(self.cluster_proj.parameters()), "lr_scale": 1.0, "name": "cluster_proj"})
        if self.use_prototype:
            groups.append({"params": [self.prototypes], "lr_scale": 1.0, "name": "prototypes"})
        else:
            groups.append({"params": list(self.cluster_linear.parameters()), "lr_scale": 1.0, "name": "cluster_linear"})
        groups.append({"params": [self.log_tau], "lr_scale": 1.0, "name": "log_tau"})
        return groups
