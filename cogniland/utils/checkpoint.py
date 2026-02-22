"""Checkpoint save/load utilities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer | None,
    step: int,
    path: str = "checkpoints/checkpoint.pt",
    extra: dict | None = None,
) -> str:
    """Save model + optimizer + RNG state to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "np_rng_state": np.random.get_state(),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        ckpt.update(extra)

    torch.save(ckpt, path)
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict:
    """Load checkpoint and restore model/optimizer/RNG state.

    Returns the full checkpoint dict (for accessing step, etc.).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if "np_rng_state" in ckpt:
        np.random.set_state(ckpt["np_rng_state"])
    return ckpt
