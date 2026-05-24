"""Progressive self-training for cluster refinement.

Implements confidence-based pseudo-labeling that gradually includes
more samples as training progresses and assignments stabilize.
"""

import torch
import torch.nn.functional as F


class ProgressiveSelfTrainer:
    """Progressive self-training with confidence-based pseudo-labeling.

    Gradually increases the confidence threshold over training epochs
    to include more samples in the self-training objective.
    """

    def __init__(
        self,
        initial_threshold: float = 0.9,
        final_threshold: float = 0.6,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        momentum: float = 0.999,
    ):
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.momentum = momentum
        self.pseudo_labels = None
        self.label_confidence = None

    def get_threshold(self, epoch: int) -> float:
        """Compute current confidence threshold with linear schedule."""
        if epoch < self.warmup_epochs:
            return self.initial_threshold
        progress = (epoch - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1
        )
        progress = min(progress, 1.0)
        return self.initial_threshold - progress * (
            self.initial_threshold - self.final_threshold
        )

    @torch.no_grad()
    def update_pseudo_labels(
        self, assignments: torch.Tensor, epoch: int
    ) -> dict:
        """Generate pseudo-labels from current batch's confident assignments.

        Args:
            assignments: (N, K) soft cluster assignments
            epoch: current training epoch
        Returns:
            dict with pseudo_labels, mask, threshold, coverage
        """
        confidence, labels = assignments.max(dim=-1)
        threshold = self.get_threshold(epoch)
        mask = confidence >= threshold

        return {
            "pseudo_labels": labels,
            "mask": mask,
            "threshold": threshold,
            "coverage": mask.float().mean().item(),
        }

    def self_training_loss(
        self,
        logits2: torch.Tensor,
        pseudo_labels: torch.Tensor,
        mask: torch.Tensor,
        soft_targets: torch.Tensor = None,
        sharpen_temp: float = 0.5,
    ) -> torch.Tensor:
        """Sharpened cross-view KL divergence on confident samples.

        Uses the full soft assignment distribution from view 1 (sharpened)
        as target for view 2 logits, providing gradient to all K classes.

        Args:
            logits2: (N, K) assignment logits from view 2
            pseudo_labels: (N,) pseudo cluster labels (unused, kept for compat)
            mask: (N,) boolean mask for confident samples
            soft_targets: (N, K) soft assignments from view 1 (detached)
            sharpen_temp: temperature for sharpening targets (< 1 = sharper)
        Returns:
            scalar loss (0 if no confident samples)
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits2.device)

        if soft_targets is not None:
            # Sharpen soft targets: re-normalize with lower temperature
            sharp = soft_targets[mask].pow(1.0 / sharpen_temp)
            sharp = sharp / sharp.sum(dim=-1, keepdim=True)
            # KL(sharp_target || softmax(logits2))
            log_pred = F.log_softmax(logits2[mask], dim=-1)
            return F.kl_div(log_pred, sharp, reduction="batchmean")
        else:
            # Fallback: hard CE (legacy)
            return F.cross_entropy(logits2[mask], pseudo_labels[mask])
