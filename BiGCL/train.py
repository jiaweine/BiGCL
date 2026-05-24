"""Training script for BiGCL.

Usage:
    python train.py --config configs/cifar10.yaml
    python train.py --config configs/cifar10.yaml --gpu 0 --batch_size 128
"""

import os
import sys
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from tqdm import tqdm

from models.bigcl import BiGCL
from models.clustering_head import ProgressiveSelfTrainer
from data.datasets import build_dataset, build_dataloader
from utils.metrics import evaluate_clustering
from utils.misc import (
    set_seed,
    AverageMeter,
    save_checkpoint,
    cosine_scheduler,
    get_logger,
)


def parse_args():
    parser = argparse.ArgumentParser(description="BiGCL Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    # Override common hyperparameters
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--clip_model", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


def load_config(args):
    """Load YAML config and apply CLI overrides."""
    # Load default config
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Load dataset-specific config and merge
    if args.config != "configs/default.yaml":
        with open(args.config, "r") as f:
            override = yaml.safe_load(f)
        for section, values in override.items():
            if isinstance(values, dict) and section in cfg:
                cfg[section].update(values)
            else:
                cfg[section] = values

    # CLI overrides
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.n_clusters is not None:
        cfg["model"]["n_clusters"] = args.n_clusters
    if args.dataset is not None:
        cfg["dataset"]["name"] = args.dataset
    if args.clip_model is not None:
        cfg["model"]["clip_model_name"] = args.clip_model
    if args.output_dir is not None:
        cfg["logging"]["output_dir"] = args.output_dir
    if args.use_wandb:
        cfg["logging"]["use_wandb"] = True

    return cfg


def build_model(cfg, device):
    """Instantiate BiGCL model from config."""
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


def build_optimizer(model, cfg):
    """Build optimizer with parameter-group-specific learning rates."""
    train_cfg = cfg["training"]
    base_lr = train_cfg["lr"]
    wd = train_cfg["weight_decay"]

    param_groups = model.get_trainable_params()
    opt_groups = []
    for pg in param_groups:
        opt_groups.append({
            "params": pg["params"],
            "lr": base_lr * pg["lr_scale"],
            "weight_decay": wd,
            "name": pg["name"],
        })

    if train_cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(opt_groups)
    else:
        optimizer = torch.optim.SGD(opt_groups, momentum=0.9)

    return optimizer


@torch.no_grad()
def evaluate(model, eval_loader, device, logger):
    """Run evaluation and compute clustering metrics."""
    model.eval()
    all_labels = []
    all_preds = []
    all_features = []

    for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"]

        out = model.extract_features(images)
        all_preds.append(out["hard_labels"].cpu().numpy())
        all_labels.append(labels.numpy())
        all_features.append(out["features"].cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_features = np.concatenate(all_features)

    metrics = evaluate_clustering(all_labels, all_preds)
    logger.info(
        f"  ACC={metrics['ACC']:.4f}  NMI={metrics['NMI']:.4f}  ARI={metrics['ARI']:.4f}"
    )
    return metrics, all_features


def train_one_epoch(
    model, train_loader, optimizer, lr_schedule, epoch, total_epochs,
    self_trainer, device, logger, cfg, global_step
):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter("loss")
    node_meter = AverageMeter("node")
    cross_meter = AverageMeter("cross")
    graph_meter = AverageMeter("graph")

    log_freq = cfg["logging"]["log_freq"]

    for batch_idx, batch in enumerate(train_loader):
        # Update learning rate
        step = global_step + batch_idx
        for i, pg in enumerate(optimizer.param_groups):
            base_scale = model.get_trainable_params()[i]["lr_scale"]
            pg["lr"] = lr_schedule[min(step, len(lr_schedule) - 1)] * base_scale

        images_v1 = batch["view1"].to(device)
        images_v2 = batch["view2"].to(device)

        # Forward pass with view1 (augmentation is already applied in dataset)
        out = model(images_v1)

        loss = out["loss"]

        # Self-training loss
        if (
            self_trainer is not None
            and cfg["self_training"]["enable"]
            and epoch >= cfg["self_training"]["warmup_epochs"]
        ):
            st_out = self_trainer.update_pseudo_labels(
                out["assignments"].detach(), epoch
            )
            st_loss = self_trainer.self_training_loss(
                out["logits"], st_out["pseudo_labels"].to(device), st_out["mask"].to(device)
            )
            loss = loss + cfg["loss"]["w_self_train"] * st_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if cfg["training"]["grad_clip"] > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["grad_clip"]
            )
        optimizer.step()

        # Update momentum teacher
        model.update_teacher(epoch, total_epochs)

        # Logging
        loss_meter.update(out["loss"].item())
        node_meter.update(out.get("node_loss", 0))
        cross_meter.update(out.get("cross_loss", 0))
        graph_meter.update(out.get("graph_loss", 0))

        if (batch_idx + 1) % log_freq == 0:
            lr_current = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch [{epoch}/{total_epochs}] Step [{batch_idx+1}/{len(train_loader)}] "
                f"Loss={loss_meter.avg:.4f} Node={node_meter.avg:.4f} "
                f"Cross={cross_meter.avg:.4f} Graph={graph_meter.avg:.4f} "
                f"LR={lr_current:.2e}"
            )

    return loss_meter.avg, global_step + len(train_loader)


def main():
    args = parse_args()
    cfg = load_config(args)

    # Setup
    set_seed(cfg["seed"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    output_dir = cfg["logging"]["output_dir"]
    dataset_name = cfg["dataset"]["name"]
    exp_name = f"{dataset_name}_k{cfg['model']['n_clusters']}_s{cfg['seed']}"
    save_dir = os.path.join(output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = get_logger("BiGCL", log_file=os.path.join(save_dir, "train.log"))
    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")
    logger.info(f"Save directory: {save_dir}")

    # Save config
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # Build dataset
    train_dataset, n_clusters = build_dataset(
        name=cfg["dataset"]["name"],
        data_dir=cfg["dataset"]["data_dir"],
        split="train",
        image_size=cfg["dataset"]["image_size"],
        strong_aug=cfg["dataset"]["strong_aug"],
        eval_mode=False,
    )
    eval_dataset, _ = build_dataset(
        name=cfg["dataset"]["name"],
        data_dir=cfg["dataset"]["data_dir"],
        split="test" if cfg["dataset"]["name"] != "stl10" else "test",
        image_size=cfg["dataset"]["image_size"],
        eval_mode=True,
    )

    # Override n_clusters if auto-detected
    if cfg["model"]["n_clusters"] != n_clusters:
        logger.info(f"Overriding n_clusters: {cfg['model']['n_clusters']} -> {n_clusters}")
        cfg["model"]["n_clusters"] = n_clusters

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
        shuffle=True,
        drop_last=True,
    )
    eval_loader = build_dataloader(
        eval_dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
        shuffle=False,
        drop_last=False,
    )

    logger.info(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")
    logger.info(f"Number of clusters: {n_clusters}")

    # Build model
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # Build optimizer
    optimizer = build_optimizer(model, cfg)

    # Learning rate schedule
    total_epochs = cfg["training"]["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    lr_schedule = cosine_scheduler(
        base_value=cfg["training"]["lr"],
        final_value=cfg["training"]["min_lr"],
        epochs=total_steps,
        warmup_epochs=cfg["training"]["warmup_epochs"] * steps_per_epoch,
    )

    # Self-trainer
    self_trainer = None
    if cfg["self_training"]["enable"]:
        self_trainer = ProgressiveSelfTrainer(
            initial_threshold=cfg["self_training"]["initial_threshold"],
            final_threshold=cfg["self_training"]["final_threshold"],
            warmup_epochs=cfg["self_training"]["warmup_epochs"],
            total_epochs=total_epochs,
        )

    # Resume
    start_epoch = 0
    best_acc = 0.0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        global_step = ckpt.get("global_step", 0)
        logger.info(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    # Eval only
    if args.eval_only:
        metrics, _ = evaluate(model, eval_loader, device, logger)
        logger.info(f"Evaluation: {metrics}")
        return

    # Optional wandb
    if cfg["logging"]["use_wandb"]:
        try:
            import wandb
            wandb.init(project=cfg["logging"]["wandb_project"], name=exp_name, config=cfg)
        except ImportError:
            logger.warning("wandb not installed, skipping")
            cfg["logging"]["use_wandb"] = False

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()

        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, lr_schedule, epoch, total_epochs,
            self_trainer, device, logger, cfg, global_step,
        )

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{total_epochs} done in {elapsed:.1f}s, avg_loss={avg_loss:.4f}"
        )

        # Evaluation
        if (epoch + 1) % cfg["eval"]["eval_freq"] == 0 or epoch == total_epochs - 1:
            metrics, features = evaluate(model, eval_loader, device, logger)

            is_best = metrics["ACC"] > best_acc
            if is_best:
                best_acc = metrics["ACC"]
                logger.info(f"  ** New best ACC: {best_acc:.4f} **")

            if cfg["logging"]["use_wandb"]:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "loss": avg_loss,
                    "ACC": metrics["ACC"],
                    "NMI": metrics["NMI"],
                    "ARI": metrics["ARI"],
                    "best_ACC": best_acc,
                })

            # Save checkpoint
            if (epoch + 1) % cfg["eval"]["save_freq"] == 0 or is_best:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "metrics": metrics,
                        "global_step": global_step,
                        "config": cfg,
                    },
                    save_dir=save_dir,
                    filename=f"checkpoint_epoch{epoch}.pth",
                    is_best=is_best,
                )

    logger.info("=" * 60)
    logger.info(f"Training complete. Best ACC: {best_acc:.4f}")
    logger.info("=" * 60)

    if cfg["logging"]["use_wandb"]:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
