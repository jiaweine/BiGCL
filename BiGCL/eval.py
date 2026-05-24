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
import yaml
from tqdm import tqdm

from models.bigcl import BiGCL
from data.datasets import build_dataset, build_dataloader
from utils.metrics import evaluate_clustering
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


def build_model_from_config(cfg, device):
    """Build model from saved config dict."""
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]

    model = BiGCL(
        clip_model_name=model_cfg["clip_model_name"],
        clip_pretrained=model_cfg["clip_pretrained"],
        n_clusters=model_cfg["n_clusters"],
        proj_dim=model_cfg["proj_dim"],
        n_ctx=model_cfg["n_ctx"],
        gnn_layers=model_cfg["gnn_layers"],
        gnn_heads=model_cfg["gnn_heads"],
        gnn_top_k=model_cfg["gnn_top_k"],
        gnn_temperature=model_cfg["gnn_temperature"],
        feat_drop_rate=model_cfg["feat_drop_rate"],
        edge_drop_rate=model_cfg["edge_drop_rate"],
        sinkhorn_iters=model_cfg["sinkhorn_iters"],
        node_temperature=loss_cfg["node_temperature"],
        cross_temperature=loss_cfg["cross_temperature"],
        graph_temperature=loss_cfg["graph_temperature"],
        w_node=loss_cfg["w_node"],
        w_cross=loss_cfg["w_cross"],
        w_graph=loss_cfg["w_graph"],
        w_uniform=loss_cfg["w_uniform"],
        w_entropy=loss_cfg["w_entropy"],
        w_self_train=loss_cfg["w_self_train"],
        use_momentum=model_cfg["use_momentum"],
        base_momentum=model_cfg["base_momentum"],
        dropout=model_cfg["dropout"],
    )
    return model.to(device)


@torch.no_grad()
def extract_all_features(model, data_loader, device):
    """Extract features, predictions, and labels from the entire dataset."""
    model.eval()
    all_features = []
    all_preds = []
    all_labels = []
    all_assignments = []

    for batch in tqdm(data_loader, desc="Extracting features"):
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


def compute_per_class_accuracy(y_true, y_pred, n_classes):
    """Compute per-class clustering accuracy after Hungarian matching."""
    from scipy.optimize import linear_sum_assignment

    n_labels = max(y_true.max(), y_pred.max()) + 1
    cost_matrix = np.zeros((n_labels, n_labels), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cost_matrix[t, p] += 1

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    mapping = dict(zip(col_ind, row_ind))

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
