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
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'"
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except (RuntimeError, EOFError) as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{checkpoint_path}' "
            f"(file may be corrupted): {exc}"
        ) from exc
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' is missing 'model_state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"State dict mismatch when loading '{checkpoint_path}': {exc}"
        ) from exc
    if optimizer is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            logger.warning(
                "Checkpoint missing 'optimizer_state_dict'; "
                "optimizer state will not be restored."
            )
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
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        log.addHandler(fh)

    return log
