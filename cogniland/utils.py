"""Checkpoint and reproducibility utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_reproducibility(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds and (optionally) enable deterministic mode."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

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
    """Load checkpoint and restore model/optimizer/RNG state."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if "np_rng_state" in ckpt:
        np.random.set_state(ckpt["np_rng_state"])
    return ckpt


# ---------------------------------------------------------------------------
# Trajectory rendering (shared by all model eval loops)
# ---------------------------------------------------------------------------

def render_trajectory(world_map, positions, target, reached_target, env_idx,
                      compiled, observed_mask=None):
    """Render agent trajectory on top of the island map.

    Args:
        world_map: [H, W] heightmap tensor.
        positions: list of (row, col) tuples — the agent's path.
        target: [2] tensor — target position.
        reached_target: bool — whether the agent reached its goal.
        env_idx: int — episode index (for title).
        compiled: CompiledTerrainData instance.
        observed_mask: optional [H, W] bool array — cells seen during episode.
            Unseen cells are rendered darker (fog of war halo).

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wm = world_map.cpu().numpy()
    thresholds = compiled.thresholds.cpu().numpy()
    terrain_map = np.searchsorted(thresholds, wm).clip(0, compiled.num_terrains - 1)

    color_lut = compiled.color_lut.float().cpu().numpy() / 255.0
    rgb = color_lut[terrain_map]

    # Fog-of-war: darken cells the agent never observed
    if observed_mask is not None:
        fog = np.where(observed_mask[:, :, None], 1.0, 0.70).astype(np.float32)
        rgb = rgb * fog

    fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
    ax.imshow(rgb, origin="upper", interpolation="nearest")

    # Per-segment visit count: colour each step red→black by revisit count
    visit_counts = np.zeros(wm.shape, dtype=np.float32)
    pos = np.array(positions)
    seg_counts = []
    for r, c in positions:
        visit_counts[r, c] += 1
        seg_counts.append(visit_counts[r, c])

    max_count = 10.0  # cap: 1 visit = red, ≥10 = black
    for i in range(len(pos) - 1):
        t = min(seg_counts[i + 1], max_count) / max_count  # 0→1
        color = (1.0 - t, 0.0, 0.0)  # red → black
        ax.plot(pos[i:i+2, 1], pos[i:i+2, 0], color=color,
                linewidth=0.8, alpha=0.9, solid_capstyle="round")

    ax.scatter(pos[0, 1], pos[0, 0], c="lime", s=120, marker="o",
               edgecolors="k", linewidth=1.5, zorder=5, label="Start")
    ax.scatter(pos[-1, 1], pos[-1, 0], c="red", s=120, marker="X",
               edgecolors="k", linewidth=1.5, zorder=5, label="End", alpha=0.5)
    tgt = target.cpu().numpy()
    ax.scatter(tgt[1], tgt[0], c="gold", s=160, marker="*",
               edgecolors="k", linewidth=1.5, zorder=5, label="Target")

    status = "SUCCESS" if reached_target else "FAILED"
    ax.set_title(f"{status} ({len(positions)} moves)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_axis_off()
    fig.tight_layout()
    return fig
