"""Miscellaneous utilities for training and evaluation."""

import os
import random
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    state: dict,
    save_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
):
    """Save training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(save_dir, "best.pth")
        torch.save(state, best_path)
    logger.info(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    strict: bool = True,
) -> dict:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    warmup_epochs: int = 0,
    warmup_start_value: float = 0.0,
) -> np.ndarray:
    """Cosine annealing schedule with linear warmup.

    Args:
        base_value: peak learning rate
        final_value: minimum learning rate
        epochs: total number of epochs
        warmup_epochs: linear warmup duration
        warmup_start_value: initial warmup value
    Returns:
        (epochs,) array of scheduled values
    """
    warmup = np.linspace(warmup_start_value, base_value, warmup_epochs)
    cosine_epochs = epochs - warmup_epochs
    cosine = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * np.arange(cosine_epochs) / cosine_epochs)
    )
    return np.concatenate([warmup, cosine])


def get_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Create logger with console and optional file handler."""
    log = logging.getLogger(name)
    log.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        log.addHandler(fh)

    return log
