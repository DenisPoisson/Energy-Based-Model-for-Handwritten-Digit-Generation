"""Helper functions for the FashionMNIST EBM project.

Pure utilities only -- no training logic lives here:
    - reproducibility (seeds)
    - checkpoint save/load
    - metric logging (csv + json)
    - plotting (loss / energy / time curves)
    - saving generated-image grids
"""
from __future__ import annotations

import csv
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # headless (server has no display) -> render straight to file
import matplotlib.pyplot as plt
import torchvision


# --------------------------------------------------------------------------- reproducibility
def set_seed(seed: int) -> None:
    """Seed every RNG so a run is reproducible: Python, NumPy, Torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic cuDNN (slightly slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, loss: float, config: Dict[str, Any]) -> None:
    """Save a full checkpoint: weights + optimizer state + epoch + loss + config.

    Storing the optimizer state and epoch is what makes training RESUMABLE.
    Storing the config makes the checkpoint self-describing (reproducibility).
    """
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load a checkpoint into `model` (and `optimizer` if given). Returns the raw dict
    so the caller can read epoch/loss/config for resuming."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# --------------------------------------------------------------------------- metrics
def save_metrics(history: List[Dict[str, float]], out_dir: str) -> None:
    """Write the per-epoch metric list to BOTH training_metrics.csv and .json.

    `history` is a list of dicts, one per epoch, e.g.
        {"epoch": 1, "loss": 0.8, "e_real": -1.2, "e_fake": 0.5, "time_sec": 12.4}
    """
    os.makedirs(out_dir, exist_ok=True)
    fields = ["epoch", "loss", "e_real", "e_fake", "time_sec"]

    with open(os.path.join(out_dir, "training_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    with open(os.path.join(out_dir, "training_metrics.json"), "w") as f:
        json.dump(history, f, indent=2)


# --------------------------------------------------------------------------- plots
def plot_curves(history: List[Dict[str, float]], out_dir: str) -> None:
    """Generate the 3 required plots as PNGs: loss, energy separation, epoch time."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    # 1. Loss vs epoch
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [h["loss"] for h in history], marker="o", ms=3)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training loss")
    plt.axhline(0, color="gray", lw=0.5); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=120); plt.close()

    # 2. Energy separation: E_real vs E_fake (the KEY health plot for an EBM)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [h["e_real"] for h in history], marker="o", ms=3, label="E_real")
    plt.plot(epochs, [h["e_fake"] for h in history], marker="o", ms=3, label="E_fake")
    plt.xlabel("epoch"); plt.ylabel("energy")
    plt.title("Energy separation (real should sit BELOW fake)")
    plt.axhline(0, color="gray", lw=0.5); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "energy_curve.png"), dpi=120); plt.close()

    # 3. Epoch time vs epoch
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [h["time_sec"] for h in history], marker="o", ms=3)
    plt.xlabel("epoch"); plt.ylabel("seconds"); plt.title("Epoch training time")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_curve.png"), dpi=120); plt.close()


# --------------------------------------------------------------------------- sample grids
def save_sample_grid(images: torch.Tensor, path: str, nrow: int = 8) -> None:
    """Save a batch of generated images (B,1,28,28) as a single PNG grid.

    normalize + value_range map the model's [-1, 1] pixels back to a viewable range.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = torchvision.utils.make_grid(
        images.cpu(), nrow=nrow, normalize=True, value_range=(-1, 1)
    )
    plt.figure(figsize=(9, 9))
    plt.imshow(grid.permute(1, 2, 0))  # (C,H,W) -> (H,W,C) for matplotlib
    plt.axis("off"); plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
