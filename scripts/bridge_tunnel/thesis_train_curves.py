#!/usr/bin/env python3
"""§3 figure: fork_wall training curves, aux-belief vs no-aux.

Reads metrics.jsonl from both checkpoint dirs. Panels:
  (A) mean episode return          (B) success (correct door, all episodes)
  (C) split: % terminated (either door) & % correct-door AMONG terminated
  (D) aux-belief head accuracy (aux run only)

  python scripts/bridge_tunnel/thesis_train_curves.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "aux belief (s2)": Path("outputs/ppo_checkpoints/forkwall_aux/forkwall_aux/metrics.jsonl"),
    "aux belief (s3)": Path("outputs/ppo_checkpoints/forkwall_aux_s3/forkwall_aux_s3/metrics.jsonl"),
    "no aux (s2)":     Path("outputs/ppo_checkpoints/forkwall_noaux/forkwall_noaux/metrics.jsonl"),
}
COL = {"aux belief (s2)": "#8e44ad", "aux belief (s3)": "#c39bd3", "no aux (s2)": "#1b9e77"}
OUT = Path("outputs/thesis_forkwall")


def _smooth(x, k=9):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    pad = np.concatenate([x[:1].repeat(k // 2), x, x[-1:].repeat(k // 2)])
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def main():
    data = {}
    for name, p in RUNS.items():
        rows = [json.loads(l) for l in open(p)]
        rows = [r for r in rows if "return/mean" in r]
        data[name] = dict(
            step=np.array([r["step"] for r in rows]) / 1e6,
            ret=np.array([r["return/mean"] for r in rows]),
            succ=np.array([r["success/mean"] for r in rows]),
            term=np.array([r["success/terminated"] for r in rows]),
            door=np.array([r["success/door_given_terminated"] for r in rows]),
            bel=np.array([r.get("belief/acc", np.nan) for r in rows]),
        )
        print(f"{name}: {len(rows)} iters, final succ "
              f"{np.mean(data[name]['succ'][-10:]):.3f} term "
              f"{np.mean(data[name]['term'][-10:]):.3f} door|term "
              f"{np.mean(data[name]['door'][-10:]):.3f}")

    fig, axs = plt.subplots(1, 4, figsize=(17.5, 3.9))
    for name, d in data.items():
        c = COL[name]
        axs[0].plot(d["step"], _smooth(d["ret"]), c=c, lw=2, label=name)
        axs[1].plot(d["step"], _smooth(d["succ"]), c=c, lw=2, label=name)
        axs[2].plot(d["step"], _smooth(d["term"]), c=c, lw=2, label=f"{name}: terminated")
        axs[2].plot(d["step"], _smooth(d["door"]), c=c, lw=2, ls="--",
                    label=f"{name}: door|term")
        if np.isfinite(d["bel"]).any():
            axs[3].plot(d["step"], _smooth(d["bel"]), c=c, lw=2, label=name)
    axs[0].set_ylabel("mean episode return"); axs[0].set_title("(A) return")
    axs[1].set_ylabel("success (correct door)"); axs[1].set_ylim(-0.02, 1.02)
    axs[1].set_title("(B) success rate")
    axs[2].set_ylim(-0.02, 1.02)
    axs[2].set_title("(C) split: terminated / correct door among terminated")
    axs[3].set_ylim(0.3, 1.02); axs[3].axhline(1 / 3, ls=":", c="#999")
    axs[3].set_title("(D) aux belief-head accuracy")
    for ax in axs:
        ax.set_xlabel("environment steps (M)"); ax.legend(fontsize=7.5)
    fig.suptitle("fork_wall PPO+GRU training: auxiliary belief loss (2 seeds) vs none "
                 "(identical hyperparameters)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig3_training.png", dpi=140, bbox_inches="tight")
    print(f"wrote {OUT/'fig3_training.png'}")


if __name__ == "__main__":
    main()
