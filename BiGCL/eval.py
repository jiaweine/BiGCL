"""Evaluation script for BiGCL.

Loads a trained checkpoint and evaluates clustering performance.
Optionally saves features and visualizations.

Usage:
    python eval.py --checkpoint output/cifar10_k10_s42/best.pth
    python eval.py --checkpoint output/cifar10_k10_s42/best.pth --save_features
"""

import os
import argparse
import numpy as np
import torch

from models.factory import build_model_from_config
from data.datasets import build_dataset, build_dataloader
from utils.evaluation import extract_all_features
from utils.metrics import evaluate_clustering, hungarian_match
from utils.misc import set_seed, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="BiGCL Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_features", action="store_true", help="Save extracted features")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    return parser.parse_args()


def compute_per_class_accuracy(y_true, y_pred, n_classes):
    """Compute per-class clustering accuracy after Hungarian matching."""
    mapping = hungarian_match(y_true, y_pred)

    per_class = {}
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        mapped_preds = np.array([mapping.get(p, -1) for p in y_pred[mask]])
        per_class[c] = (mapped_preds == c).mean()

    return per_class


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    save_dir = args.save_dir or os.path.dirname(args.checkpoint)
    logger = get_logger("BiGCL-Eval", log_file=os.path.join(save_dir, "eval.log"))

    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    logger.info(f"Config: dataset={cfg['dataset']['name']}, n_clusters={cfg['model']['n_clusters']}")
    if "epoch" in ckpt:
        logger.info(f"Checkpoint epoch: {ckpt['epoch']}")
    if "best_acc" in ckpt:
        logger.info(f"Checkpoint best ACC: {ckpt['best_acc']:.4f}")

    set_seed(cfg["seed"])

    # Build model
    model = build_model_from_config(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Model loaded successfully")

    # Build evaluation dataset
    eval_dataset, n_clusters = build_dataset(
        name=cfg["dataset"]["name"],
        data_dir=cfg["dataset"]["data_dir"],
        split=args.split,
        image_size=cfg["dataset"]["image_size"],
        eval_mode=True,
    )
    eval_loader = build_dataloader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=cfg["dataset"]["num_workers"],
        shuffle=False,
        drop_last=False,
    )
    logger.info(f"Evaluation set ({args.split}): {len(eval_dataset)} samples")

    # Extract features
    results = extract_all_features(model, eval_loader, device)

    # Compute metrics
    metrics = evaluate_clustering(results["labels"], results["predictions"])
    logger.info("=" * 50)
    logger.info(f"Clustering Results on {cfg['dataset']['name']} ({args.split}):")
    logger.info(f"  ACC = {metrics['ACC']:.4f}")
    logger.info(f"  NMI = {metrics['NMI']:.4f}")
    logger.info(f"  ARI = {metrics['ARI']:.4f}")
    logger.info("=" * 50)

    # Per-class accuracy
    per_class = compute_per_class_accuracy(
        results["labels"], results["predictions"], n_clusters
    )
    logger.info("Per-class accuracy:")
    for c, acc in sorted(per_class.items()):
        logger.info(f"  Class {c}: {acc:.4f}")

    # Cluster distribution
    unique, counts = np.unique(results["predictions"], return_counts=True)
    logger.info("Cluster distribution:")
    for u, c in zip(unique, counts):
        logger.info(f"  Cluster {u}: {c} samples ({c/len(results['predictions'])*100:.1f}%)")

    # Save features if requested
    if args.save_features:
        feat_path = os.path.join(save_dir, f"features_{args.split}.npz")
        np.savez(
            feat_path,
            features=results["features"],
            predictions=results["predictions"],
            assignments=results["assignments"],
            labels=results["labels"],
        )
        logger.info(f"Features saved to {feat_path}")

    # Save metrics
    import json
    metrics_path = os.path.join(save_dir, f"metrics_{args.split}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
