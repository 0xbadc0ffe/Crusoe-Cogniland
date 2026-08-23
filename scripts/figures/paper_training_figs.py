#!/usr/bin/env python3
"""Training-curve figures for the fork_wall paper (reads training_data.json).

  fig_ppo_training.png      PPO: 4-arm x 3-seed exploration sweep (return, success,
                            episode length, belief-probe accuracy, KL, entropy schedule)
  fig_dreamer_training.png  Dreamer: capacity x context sweep (eval success/score,
                            world-model losses, actor entropy, throughput)
  fig_storm_training.png    STORM: reward/success, world-model + AC losses
  fig_compare.png           all three on one axis: held-out success vs env frames
                            + sample-efficiency and wall-clock summary bars

Usage: python scripts/figures/paper_training_figs.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PLT_RC = {
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": .25, "grid.linewidth": .6,
}
C = {"ppo": "#d97706", "dreamer": "#2563eb", "storm": "#16a34a"}
ARM_C = ["#d97706", "#7c3aed", "#0891b2", "#be123c"]


def xy(series, key, xscale=1.0):
    a = np.asarray(series.get(key, []), dtype=float)
    if a.size == 0:
        return np.array([]), np.array([])
    return a[:, 0] * xscale, a[:, 1]


def smooth(y, k=9):
    """Centred moving average with shrinking windows at the edges.

    (A plain `np.convolve(..., "same")` divides the truncated edge windows by
    the full kernel width and fabricates a collapse in the last points.)
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3 or k < 2:
        return y
    k = min(k, len(y))
    c = np.concatenate([[0.0], np.cumsum(y)])
    n = len(y)
    idx = np.arange(n)
    lo = np.maximum(idx - k // 2, 0)
    hi = np.minimum(idx + k // 2 + 1, n)
    return (c[hi] - c[lo]) / (hi - lo)


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def band_by_arm(runs, key, arm_filter):
    """Aggregate seeds of one arm -> (x, mean, min, max) on a common grid."""
    xs, ys = [], []
    for name, blob in runs.items():
        if not name.startswith(arm_filter):
            continue
        x, y = xy(blob["series"], key)
        if x.size:
            xs.append(x); ys.append(y)
    if not xs:
        return None
    grid = np.linspace(max(x[0] for x in xs), min(x[-1] for x in xs), 200)
    ip = np.stack([np.interp(grid, x, y) for x, y in zip(xs, ys)])
    return grid, ip.mean(0), ip.min(0), ip.max(0), len(xs)


# ── PPO ──────────────────────────────────────────────────────────────────

def fig_ppo(data, out):
    runs = data["ppo"]
    arms = sorted({b["meta"]["arm"] for b in runs.values()})
    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.0))

        panels = [
            ("return", "episode return", "(a) training return"),
            ("success", "success rate", "(b) training success (reward>0 proxy)"),
            ("ep_length", "steps", "(c) episode length"),
            ("belief_acc", "accuracy", "(d) belief-probe accuracy (aux head)"),
            ("kl", "KL", "(e) policy KL per update"),
        ]
        for ax, (key, ylab, title) in zip(axes.flat, panels):
            for ci, arm in enumerate(arms):
                agg = band_by_arm(runs, key, arm)
                if agg is None:
                    continue
                g, m, lo, hi, n = agg
                star = "★" in arm
                ax.plot(g / 1e6, smooth(m), color=ARM_C[ci % 4], lw=2.0 if star else 1.2,
                        label=f"{arm} (n={n})", zorder=5 if star else 3)
                ax.fill_between(g / 1e6, smooth(lo), smooth(hi), color=ARM_C[ci % 4],
                                alpha=.13, lw=0)
            ax.set_xlabel("environment frames (M)"); ax.set_ylabel(ylab)
            ax.set_title(title, loc="left")
        axes.flat[1].axhline(2 / 3, color="#6b7280", ls="--", lw=1.0)
        axes.flat[1].annotate("constant-door ceiling (⅔)", xy=(0.35, 2 / 3),
                              fontsize=7, color="#6b7280", va="bottom")
        axes.flat[3].axhline(1 / 3, color="#6b7280", ls="--", lw=1.0)

        ax = axes.flat[5]
        for ci, arm in enumerate(arms):
            finals = [xy(b["series"], "success")[1][-20:].mean()
                      for n, b in runs.items() if n.startswith(arm)
                      and xy(b["series"], "success")[1].size]
            ax.scatter([ci] * len(finals), finals, color=ARM_C[ci % 4], s=34, zorder=4)
            ax.scatter([ci], [np.mean(finals)], marker="_", s=420,
                       color=ARM_C[ci % 4], zorder=5)
        ax.axhline(2 / 3, color="#6b7280", ls="--", lw=1.0)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a.replace(" + ", "\n+ ").replace("  ★released", "\n★released")
                            for a in arms], fontsize=6.5)
        ax.set_ylabel("final success (last 20 iters)")
        ax.set_title("(f) per-seed outcome: the basin is a seed lottery", loc="left")

        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(.5, -.02))
        fig.suptitle("PPO+GRU — plain-reward exploration sweep (4 arms × 3 seeds, 4M frames each)",
                     y=1.0)
        fig.tight_layout(rect=[0, .035, 1, .975])
        fig.savefig(out / "fig_ppo_training.png", bbox_inches="tight")
        plt.close(fig)


# ── Dreamer ──────────────────────────────────────────────────────────────

def fig_dreamer(data, out):
    runs = data["dreamer"]
    order = sorted(runs, key=lambda k: (runs[k]["meta"]["size"],
                                        runs[k]["meta"]["batch_length"]))
    cols = {"12M, batch_length 64": "#93c5fd", "12M, batch_length 128": "#2563eb",
            "25M, batch_length 64": "#fca5a5", "25M, batch_length 128": "#b91c1c"}
    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.0))
        panels = [
            ("episode/eval_success", "eval success", "(a) held-out success (eval episodes)"),
            ("episode/score", "return", "(b) training return"),
            ("episode/eval_length", "steps", "(c) eval episode length"),
            ("train/loss/dyn", "KL", "(d) dynamics loss (prior↔posterior KL)"),
            ("train/loss/rew", "nats", "(e) reward-head loss"),
            ("train/action_entropy", "nats", "(f) actor entropy"),
        ]
        for ax, (key, ylab, title) in zip(axes.flat, panels):
            for name in order:
                x, y = xy(runs[name]["series"], key)
                if x.size == 0:
                    continue
                ax.plot(x / 1e6, smooth(y, 15), color=cols.get(name, "#333"),
                        lw=1.9 if "25M, batch_length 64" == name else 1.2,
                        label=name)
            ax.set_xlabel("environment frames (M)"); ax.set_ylabel(ylab)
            ax.set_title(title, loc="left")
        axes.flat[0].axhline(2 / 3, color="#6b7280", ls="--", lw=1.0)
        axes.flat[0].annotate("constant-door ceiling (⅔)", xy=(0.5, 2 / 3), fontsize=7,
                              color="#6b7280", va="bottom")
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(.5, -.02))
        fig.suptitle("DreamerV3 (R2) — capacity × training-context sweep", y=1.0)
        fig.tight_layout(rect=[0, .035, 1, .975])
        fig.savefig(out / "fig_dreamer_training.png", bbox_inches="tight")
        plt.close(fig)


# ── STORM ────────────────────────────────────────────────────────────────

def fig_storm(data, out, heldout_csv=None):
    runs = data.get("storm", {})
    if not runs:
        print("  [storm] no telemetry yet — skipping fig_storm_training.png")
        return
    name = sorted(runs)[-1]
    s = runs[name]["series"]
    pfx = "train/BridgeTunnel/forkwall/"
    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.0))
        panels = [
            (pfx + "moving_avg_reward", "return", "(a) training return (moving avg)"),
            (pfx + "moving_avg_success_rate", "success", "(b) training success (proxy)"),
            (pfx + "moving_avg_length", "steps", "(c) episode length"),
            (pfx + "loss/rec", "nats", "(d) reconstruction loss"),
            (pfx + "loss/rew", "nats", "(e) reward-head loss"),
            (pfx + "loss/dyn", "KL", "(f) dynamics KL"),
        ]
        for ax, (key, ylab, title) in zip(axes.flat, panels):
            x, y = xy(s, key)
            if x.size == 0:
                ax.text(.5, .5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="#9ca3af")
            else:
                ax.plot(x, smooth(y, 9), color=C["storm"], lw=1.4)
            ax.set_xlabel("training episode"); ax.set_ylabel(ylab)
            ax.set_title(title, loc="left")
        fig.suptitle("STORM (storm2) — winning recipe: entropy 0.01, batch_length 128, "
                     "context 128", y=1.0)
        fig.tight_layout(rect=[0, 0, 1, .975])
        fig.savefig(out / "fig_storm_training.png", bbox_inches="tight")
        plt.close(fig)


# ── cross-agent comparison ───────────────────────────────────────────────

def fig_compare(data, out, heldout):
    """Training-time learning curves (each agent's own telemetry) + final held-out."""
    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.3),
                                 gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
        ax = axes[0]
        # PPO: released arm, mean over 3 seeds
        agg = band_by_arm(data["ppo"], "success", "ent 0.15 + anneal  ★released")
        if agg:
            g, m, lo, hi, n = agg
            ax.plot(g / 1e6, smooth(m, 9), color=C["ppo"], lw=1.8, label="PPO+GRU")
            ax.fill_between(g / 1e6, smooth(lo, 9), smooth(hi, 9), color=C["ppo"],
                            alpha=.15, lw=0)
        # Dreamer: released config (25M, bl64)
        dr = data["dreamer"].get("25M, batch_length 64")
        if dr:
            x, y = xy(dr["series"], "episode/eval_success")
            ax.plot(x / 1e6, smooth(y, 15), color=C["dreamer"], lw=1.8,
                    label="DreamerV3 25M")
        # STORM: released recipe
        st = data.get("storm", {})
        if st:
            s = st[sorted(st)[-1]]["series"]
            k = "train/BridgeTunnel/forkwall/moving_avg_success_rate"
            xs, ys = xy(s, k)
            kf = "train/BridgeTunnel/forkwall/frame"
            xf, yf = xy(s, kf)
            if xs.size and yf.size:                 # episode index -> env frames
                frames = np.interp(xs, xf, yf)
                ax.plot(frames / 1e6, smooth(ys, 9), color=C["storm"], lw=1.8,
                        label="STORM")
        ax.axhline(2 / 3, color="#6b7280", ls="--", lw=1.0)
        ax.annotate("constant-door ceiling (⅔)", xy=(0.05, .60), fontsize=7,
                    color="#6b7280", va="top")
        ax.set_xlabel("environment frames (M)")
        ax.set_ylabel("training success (proxy)")
        ax.set_title("(a) training-time learning curves", loc="left")
        ax.set_ylim(0, 1.05); ax.legend(frameon=False, loc="lower right")

        # (b) unified held-out evaluation with Wilson 95% CIs
        ax = axes[1]
        ev = json.loads((out / "eval_all.json").read_text())
        names, vals, los, his = [], [], [], []
        for a in ("ppo", "dreamer", "storm"):
            r = ev[a]
            n, p = r["episodes"], r["success"]
            lo, hi = wilson(round(p * n), n)
            names.append(a.upper()); vals.append(p); los.append(p - lo); his.append(hi - p)
        ax.bar(names, vals, color=[C["ppo"], C["dreamer"], C["storm"]],
               yerr=[los, his], capsize=4, error_kw=dict(lw=1.1, ecolor="#374151"))
        for i, v in enumerate(vals):
            ax.text(i, v + his[i] + .003, f"{v*100:.1f}%", ha="center", fontsize=8)
        ax.set_ylim(.94, 1.005); ax.set_ylabel("held-out success (TRUE metric)")
        ax.set_title("(b) unified eval, 900 episodes each\n(differences n.s., all p>0.25)",
                     loc="left", fontsize=8.5)

        # (c) outcome decomposition of the residual error
        ax = axes[2]
        w = 0.6
        bottoms = np.zeros(3)
        for key, col, lab in (("wrong_door", "#ef4444", "wrong door"),
                              ("timeout", "#f59e0b", "timeout")):
            vs = np.array([ev[a][key] * 100 for a in ("ppo", "dreamer", "storm")])
            ax.bar(["PPO", "DREAMER", "STORM"], vs, w, bottom=bottoms, color=col,
                   label=lab)
            bottoms += vs
        for i, v in enumerate(bottoms):
            ax.text(i, v + .04, f"{v:.1f}%", ha="center", fontsize=8)
        ax.set_ylabel("share of episodes (%)")
        ax.legend(frameon=False, fontsize=7.5)
        ax.set_title("(c) how the residual error is spent", loc="left")

        fig.suptitle("Three agents, one task: all clear the memoryless ⅔ ceiling", y=1.03)
        fig.tight_layout()
        fig.savefig(out / "fig_compare.png", bbox_inches="tight")
        plt.close(fig)


# held-out evaluations measured during development (TRUE door metric, sampled
# actions for PPO/STORM, deterministic for Dreamer). PPO/Dreamer points are
# single end-of-training measurements; STORM's are per-checkpoint sweeps.
HELDOUT = {
    "storm": [[583_000, .622], [962_000, .775], [1_330_000, .840],
              [2_080_000, .887], [2_440_000, .904], [2_500_000, .993]],
    "ppo":   [[4_000_000, .995]],
    "dreamer": [[3_000_000, .980]],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "paper/figures/forkwall_paper/training_data.json"))
    p.add_argument("--out", default=str(REPO / "paper/figures/forkwall_paper"))
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.data).read_text())

    fig_ppo(data, out)
    fig_dreamer(data, out)
    fig_storm(data, out)
    fig_compare(data, out, HELDOUT)
    print("wrote figures ->", out)


if __name__ == "__main__":
    main()
