#!/usr/bin/env python3
"""Pull the fork_wall PPO+GRU training run from W&B and plot return/success curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-path", required=True, help="e.g. crusoe/bridge_tunnel/<run_id>")
    p.add_argument("--out", type=Path, default=Path("outputs/bridge_tunnel_forkwall/training_curve.png"))
    args = p.parse_args()

    api = wandb.Api()
    run = api.run(args.run_path)
    hist = run.history(samples=5000)
    hist = hist.sort_values("_step")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(hist["_step"], hist["return/mean"], lw=0.7, alpha=0.35, color="#1f77b4")
    if hist["return/rolling100"].notna().any():
        ax.plot(hist["_step"], hist["return/rolling100"], lw=1.8, color="#1f77b4", label="return/rolling100")
    ax.set_xlabel("env steps"); ax.set_ylabel("episode return")
    ax.set_title("training return"); ax.legend(loc="lower right"); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(hist["_step"], hist["success/mean"], lw=0.7, alpha=0.35, color="#2ca02c")
    if hist["success/rolling100"].notna().any():
        ax.plot(hist["_step"], hist["success/rolling100"], lw=1.8, color="#2ca02c", label="success/rolling100")
    for cat, color in [("lakes", "#d62728"), ("rocky", "#9467bd"), ("balanced", "#ff7f0e")]:
        col = f"success/{cat}"
        if col in hist.columns and hist[col].notna().any():
            s = hist[["_step", col]].dropna()
            roll = s[col].rolling(20, min_periods=1).mean()
            ax.plot(s["_step"], roll, lw=1.2, color=color, alpha=0.8, label=f"{cat} (roll20)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("env steps"); ax.set_ylabel("success rate (correct door)")
    ax.set_title("training success — fork_wall task"); ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle(f"PPO+GRU bridge_tunnel fork_wall — {run.name} ({run.state})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")

    final = hist.dropna(subset=["success/mean"]).tail(20)
    print("\nlast ~20 logged iterations:")
    print(final[["_step", "return/mean", "success/mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
