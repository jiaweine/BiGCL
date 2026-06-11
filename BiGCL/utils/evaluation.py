"""Shared evaluation utilities for feature extraction and metric computation."""

import numpy as np
import torch
from tqdm import tqdm

from .metrics import evaluate_clustering


@torch.no_grad()
def extract_all_features(model, data_loader, device, desc="Extracting features"):
    """Extract features, predictions, assignments, and labels from a data loader.

    Args:
        model: BiGCL model (switched to eval mode internally)
        data_loader: DataLoader yielding dicts with 'image' and 'label' keys
        device: torch device
        desc: progress bar description
    Returns:
        dict with 'features', 'predictions', 'assignments', 'labels' (all numpy)
    """
    model.eval()
    all_features = []
    all_preds = []
    all_labels = []
    all_assignments = []

    for batch in tqdm(data_loader, desc=desc, leave=False):
        images = batch["image"].to(device)
        labels = batch["label"]

        out = model.extract_features(images)

        all_features.append(out["features"].cpu().numpy())
        all_preds.append(out["hard_labels"].cpu().numpy())
        all_assignments.append(out["assignments"].cpu().numpy())
        all_labels.append(labels.numpy())

    return {
        "features": np.concatenate(all_features),
        "predictions": np.concatenate(all_preds),
        "assignments": np.concatenate(all_assignments),
        "labels": np.concatenate(all_labels),
    }


@torch.no_grad()
def evaluate_model(model, eval_loader, device, logger=None):
    """Run evaluation: extract features and compute clustering metrics.

    Args:
        model: BiGCL model
        eval_loader: evaluation DataLoader
        device: torch device
        logger: optional logger for printing results
    Returns:
        (metrics_dict, features_array)
    """
    results = extract_all_features(model, eval_loader, device, desc="Evaluating")
    metrics = evaluate_clustering(results["labels"], results["predictions"])

    if logger is not None:
        logger.info(
            f"  ACC={metrics['ACC']:.4f}  NMI={metrics['NMI']:.4f}  ARI={metrics['ARI']:.4f}"
        )

    return metrics, results["features"]
