#!/usr/bin/env python3
"""Training curves for the IAB appendix: recurrent (GRU) PPO against the
feed-forward control, mean +/- std over seeds, from outputs/ppo_noaux/*/metrics.jsonl.

  python scripts/figures/paper/iab_training_curves.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "outputs/ppo_noaux"
OUT = [REPO / "paper/iab2026/paper/figures/training_curves.png",
       REPO / "paper/figures/iab2026/training_curves.png"]
GROUPS = {"GRU (recurrent)": [f"noaux_ent15_s{i}" for i in range(1, 6)],
          "feed-forward control": [f"ff6m_control_s{i}" for i in range(1, 6)]}
COL = {"GRU (recurrent)": "#2a78d6", "feed-forward control": "#eb6834"}
INK, INK2, MUTE = "#0b0b0b", "#52514e", "#a8a7a1"


def load(run):
    rows = [json.loads(l) for l in (RUNS / run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "return/rolling100" in r]
    step = np.array([r["step"] for r in rows], float)
    ret = np.array([r["return/rolling100"] for r in rows], float)
    dec = np.array([np.mean([r.get("success/lakes", np.nan), r.get("success/rocky", np.nan)]) for r in rows], float)
    succ = np.array([r["success/rolling100"] for r in rows], float)
    return step, ret, succ, dec


def smooth(y, k=25):
    if len(y) < k: return y
    ker = np.ones(k) / k
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, ker, mode="valid")


def main():
    grid = np.linspace(0, 6.0e6, 300)
    RC = {"figure.dpi": 200, "savefig.dpi": 200, "font.size": 9,
          "axes.spines.top": False, "axes.spines.right": False,
          "axes.edgecolor": MUTE, "xtick.color": INK2, "ytick.color": INK2,
          "axes.labelcolor": INK2, "figure.facecolor": "white", "axes.facecolor": "white"}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
        for name, runs in GROUPS.items():
            R, S, D = [], [], []
            for run in runs:
                step, ret, succ, dec = load(run)
                lim = step.max()
                g = grid[grid <= lim]
                R.append(np.interp(g, step, smooth(ret), right=np.nan))
                S.append(np.interp(g, step, smooth(succ), right=np.nan))
                D.append(np.interp(g, step, smooth(dec), right=np.nan))
            n = min(len(r) for r in R)
            g = grid[:n]
            for ax, arr, lab in zip(axes, (R, S, D), ("return (rolling 100 episodes)",
                                                       "success, all categories",
                                                       "success, lakes and rocky only")):
                a = np.array([x[:n] for x in arr])
                m, s = np.nanmean(a, 0), np.nanstd(a, 0)
                ax.plot(g / 1e6, m, color=COL[name], lw=2, label=f"{name}, {len(runs)} seeds", zorder=3)
                ax.fill_between(g / 1e6, m - s, m + s, color=COL[name], alpha=.18, lw=0, zorder=2)
                ax.set_title(lab, loc="left", color=INK, fontsize=9.5)
        for ax in axes[1:]:
            ax.set_ylim(0, 1.02)
            ax.set_yticks([0, .25, .5, .75, 1.0]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        axes[2].axhline(.5, color=MUTE, lw=.8, ls=":", zorder=1)
        axes[2].text(5.9, .52, "chance", ha="right", va="bottom", fontsize=8, color=INK2)
        axes[1].axhline(2 / 3, color=MUTE, lw=.8, ls=":", zorder=1)
        axes[1].text(5.9, .68, "constant flag", ha="right", va="bottom", fontsize=8, color=INK2)
        axes[0].set_ylim(bottom=0)
        for ax in axes:
            ax.set_xlim(0, 6.0); ax.set_xlabel("environment steps (millions)")
            ax.grid(axis="y", color="#e8e7e3", lw=.6, zorder=0)
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        for o in OUT:
            o.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(o, bbox_inches="tight")
        plt.close(fig)
    print("wrote", [str(o) for o in OUT])


if __name__ == "__main__":
    main()
