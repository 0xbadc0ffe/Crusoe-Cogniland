#!/usr/bin/env python3
"""Reward + success training curves for PPO vs DreamerV3 on the SAME bridge_tunnel
env (natural, 3-cell centre door, edge forests) with the SAME categorical
observation — pulled from W&B, plotted on shared axes for a fair comparison.

    python scripts/figures/plot_training_curves.py \\
        --run ppo+gru=epykvjql --run dreamerv3=be4qhuvf \\
        --out paper/figures/bridge_tunnel/training_curves.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


def _series(run, ykey, xkey="_step"):
    h = run.history(keys=[ykey, xkey], pandas=True)
    h = h.dropna(subset=[ykey, xkey]).sort_values(xkey)
    return h[xkey].to_numpy(), h[ykey].to_numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True,
                   help="label=wandb_run_id (repeatable)")
    p.add_argument("--project", default="crusoe/bridge_tunnel")
    p.add_argument("--out", type=Path, default=Path("paper/figures/bridge_tunnel/training_curves.png"))
    args = p.parse_args()

    api = wandb.Api()
    runs = [(lbl, api.run(f"{args.project}/{rid}"))
            for lbl, rid in (s.split("=", 1) for s in args.run)]

    fig, (axs, axr) = plt.subplots(1, 2, figsize=(11, 4.0))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (lbl, run), col in zip(runs, colors):
        sx, sy = _series(run, "success/rolling100")
        if len(sx) == 0:
            sx, sy = _series(run, "success/mean")
        rx, ry = _series(run, "return/rolling100")
        if len(rx) == 0:
            rx, ry = _series(run, "return/mean")
        axs.plot(sx / 1e6, sy, color=col, label=lbl, lw=2)
        axr.plot(rx / 1e6, ry, color=col, label=lbl, lw=2)

    axs.set_title("Success rate (rolling)")
    axs.set_xlabel("env steps (M)"); axs.set_ylabel("reach rate")
    axs.set_ylim(-0.02, 1.02); axs.grid(alpha=0.3); axs.legend(loc="lower right", fontsize=9)
    axr.set_title("Episode return (rolling)")
    axr.set_xlabel("env steps (M)"); axr.set_ylabel("return")
    axr.grid(alpha=0.3); axr.legend(loc="lower right", fontsize=9)
    fig.suptitle("PPO+GRU vs DreamerV3 — same env, same categorical observation", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")
    for lbl, run in runs:
        print(f"  {lbl}: success/rolling100={run.summary.get('success/rolling100')}  "
              f"return/rolling100={run.summary.get('return/rolling100')}  state={run.state}")


if __name__ == "__main__":
    main()
