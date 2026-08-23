#!/usr/bin/env python3
"""Reward and dataset figures for the Cogniland paper.

Figures 1 and 2 live in paper_task_figs.py; this file must never write those
filenames (two writers = whichever ran last wins, silently).

Generates (into --out, default paper/figures/forkwall_paper/):
  fig_observation.png       agent view (kept for reference; unused by the report)
  fig_reward.png            reward decomposition along a real episode
  fig_dataset.png           per-type water/rock coverage

Usage:
    PYTHONPATH=src python scripts/figures/paper_env_figs.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cogniland.bridge_tunnel.env import BridgeTunnelEnv  # noqa: E402
from cogniland.bridge_tunnel.map_pool import MapPool  # noqa: E402
from cogniland.bridge_tunnel.tiles import (  # noqa: E402
    OOB, TILE_COLORS, TILE_NAMES,
)

# the canonical fork_wall task (identical for PPO / Dreamer / STORM)
FORKWALL_KWARGS = dict(
    variant="btc", commit=False, fork_wall=True,
    categories=("balanced", "lakes", "rocky"),
    passage_half=1, wall_margin=1, mem_gap=16, shaping_gamma=1.0,
    size=32, width=64, view_size=21, max_steps=800,
    orientation="natural", tree_frac=0.03, goal_half=0,
    slack_penalty=-0.01, shaping_coef=0.015, reach_bonus=3.0,
    build_cost=0.0, commit_cost=0.05, illegal_penalty=0.02,
    gamma=0.99,
)
CATS = ("lakes", "balanced", "rocky")
CAT_DOOR = {"lakes": "bottom", "rocky": "top", "balanced": "either"}
PLT_RC = {
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
}


def rgb(terrain):
    return TILE_COLORS[terrain]


def load_records(maps_path, n_per_cat=40):
    with open(maps_path, "rb") as f:
        records = pickle.load(f)
    by_cat = {c: [] for c in CATS}
    for r in records:
        if r.category in by_cat and len(by_cat[r.category]) < n_per_cat:
            by_cat[r.category].append(r)
    return records, by_cat


def annotate_map(ax, rec, show_labels=True):
    """Draw doors (green=rewarded, red=decoy), spawn, wall column."""
    corr = rec.correct_target
    for cells, name in ((rec.top_goal_cells, "top"), (rec.bottom_goal_cells, "bottom")):
        good = corr == "either" or corr == name
        color = "#22c55e" if good else "#ef4444"
        for (r, c) in cells:
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                   edgecolor=color, lw=2.0, zorder=6))
    sr, sc = rec.spawn
    ax.plot(sc, sr, "o", color="white", mec="black", ms=6, zorder=7)
    if show_labels and rec.wall_col is not None:
        ax.axvline(rec.wall_col, color="white", ls=":", lw=1.0, alpha=.75, zorder=5)


def fig_observation(by_cat, out):
    """What the agent actually sees: egocentric 21x21 crop + 5 scalars."""
    rec = by_cat["lakes"][0]
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset()
    # walk right a few steps so the crop is interesting
    for _ in range(8):
        obs, *_ = env.step(3)
    pos = env._pos
    V = env.view_size

    with plt.rc_context(PLT_RC):
        fig = plt.figure(figsize=(10.5, 3.9))
        gs = fig.add_gridspec(1, 3, width_ratios=[2.35, 1.0, 1.05], wspace=0.18)

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(rgb(rec.terrain), interpolation="nearest")
        half = V // 2
        ax0.add_patch(Rectangle((pos[1] - half - .5, pos[0] - half - .5), V, V,
                                fill=False, edgecolor="white", lw=1.8, zorder=6))
        ax0.plot(pos[1], pos[0], "o", color="white", mec="black", ms=6, zorder=7)
        ax0.set_title("(a) world state (privileged; never observed)", loc="left")
        ax0.set_xticks([]); ax0.set_yticks([])

        ax1 = fig.add_subplot(gs[1])
        crop = np.asarray(obs["minimap"])
        ax1.imshow(rgb(crop), interpolation="nearest")
        ax1.plot(half, half, "o", color="white", mec="black", ms=6)
        ax1.set_title(f"(b) observation: {V}×{V} egocentric crop", loc="left")
        ax1.set_xticks([]); ax1.set_yticks([])
        oob_frac = float((crop == OOB).mean())
        ax1.set_xlabel(f"out-of-bounds padding: {oob_frac:.0%} of cells")

        ax2 = fig.add_subplot(gs[2])
        sc = np.asarray(obs["scalars"], dtype=float)
        names = ["facing↑", "facing↓", "facing←", "facing→", "step/max"]
        ax2.barh(range(len(sc)), sc, color="#6366f1")
        ax2.set_yticks(range(len(sc))); ax2.set_yticklabels(names)
        ax2.invert_yaxis(); ax2.set_xlim(0, 1.05)
        ax2.set_title("(c) observation: 5 scalars", loc="left")
        ax2.set_xlabel("value")

        fig.suptitle("The agent is a POMDP observer: a symbolic local crop plus "
                     "heading and elapsed time — the category is never given directly",
                     y=1.02, fontsize=9.5)
        fig.savefig(out / "fig_observation.png", bbox_inches="tight")
        plt.close(fig)
    return dict(vec_dim=V * V * 9 + len(sc), view=V, n_scalars=len(sc))


def fig_reward(by_cat, out, ppo_ckpt):
    """Reward decomposition along a real trajectory of the released PPO agent."""
    import torch
    from cogniland.bridge_tunnel.policy import PPOGRUPolicy

    rec = by_cat["rocky"][2]
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset()
    ckpt = torch.load(ppo_ckpt, map_location="cpu", weights_only=False)
    policy = PPOGRUPolicy.from_checkpoint(ckpt, env.observation_space)
    h = torch.zeros(1, 1, policy.gru_hidden)

    rewards, shaped, slack, steps = [], [], [], []
    for t in range(FORKWALL_KWARGS["max_steps"]):
        with torch.no_grad():
            tobs = {k: torch.as_tensor(np.asarray(v))[None] for k, v in obs.items()}
            a, _, _, _, h = policy.get_action_and_value(tobs, h, torch.zeros(1))
        obs, r, term, trunc, _ = env.step(int(a.item()))
        rewards.append(r)
        slack.append(FORKWALL_KWARGS["slack_penalty"])
        shaped.append(r - FORKWALL_KWARGS["slack_penalty"])
        steps.append(t + 1)
        if term or trunc:
            break

    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4),
                                 gridspec_kw=dict(width_ratios=[1.35, 1]))
        ax = axes[0]
        ax.plot(steps, np.cumsum(rewards), color="#111827", lw=1.8, label="cumulative return")
        ax.plot(steps, np.cumsum(slack), color="#ef4444", lw=1.2, ls="--",
                label="cumulative slack ($-0.01\\,t$)")
        ax.plot(steps, np.cumsum(shaped), color="#2563eb", lw=1.2, ls="-.",
                label="cumulative shaping + bonus")
        ax.axhline(0, color="#9ca3af", lw=.6)
        ax.set_xlabel("environment step"); ax.set_ylabel("reward")
        ax.set_title("(a) return decomposition, near-optimal episode", loc="left")
        ax.legend(frameon=False, fontsize=8)

        ax = axes[1]
        ax.plot(steps, rewards, color="#2563eb", lw=.9)
        ax.axhline(FORKWALL_KWARGS["slack_penalty"], color="#ef4444", lw=1.0, ls="--")
        ax.set_xlabel("environment step"); ax.set_ylabel("per-step reward")
        ax.set_title("(b) per-step reward (spike = +3 door bonus)", loc="left")
        fig.tight_layout()
        fig.savefig(out / "fig_reward.png", bbox_inches="tight")
        plt.close(fig)

    return dict(steps=len(steps), total=float(np.sum(rewards)),
                slack_total=float(np.sum(slack)), shaped_total=float(np.sum(shaped)))


def fig_dataset(records, out, meta_path):
    """Per-category coverage: one panel each, ordered rocky -> balanced -> lakes."""
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    order = ("rocky", "balanced", "lakes")
    water = {c: [] for c in order}
    rock = {c: [] for c in order}
    for r in records:
        if r.category not in water:
            continue
        water[r.category].append(float((r.terrain == 1).mean()) * 100)
        rock[r.category].append(float((r.terrain == 2).mean()) * 100)

    w_col = np.array(TILE_COLORS[1]) / 255
    r_col = np.array(TILE_COLORS[2]) / 255
    bins = np.linspace(0, 28, 45)
    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.0), sharey=True, sharex=True)
        for ax, cat in zip(axes, order):
            ax.hist(water[cat], bins=bins, color=w_col, alpha=.80, label="water")
            ax.hist(rock[cat], bins=bins, color=r_col, alpha=.80, label="rock")
            ax.axvline(np.mean(water[cat]), color=w_col, lw=1.4, ls="--")
            ax.axvline(np.mean(rock[cat]), color=r_col, lw=1.4, ls="--")
            ax.set_title(f"{cat} coverage", loc="left")
            ax.set_xlabel("% of map cells")
        axes[0].set_ylabel("maps")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Each map type is a different pair of terrain fractions — that "
                     "difference is the only signal the agent can use", y=1.04)
        fig.tight_layout()
        fig.savefig(out / "fig_dataset.png", bbox_inches="tight")
        plt.close(fig)

    return dict(meta=meta,
                counts={c: len(water[c]) for c in order},
                water={c: (float(np.mean(water[c])), float(np.std(water[c]))) for c in order},
                rock={c: (float(np.mean(rock[c])), float(np.std(rock[c]))) for c in order})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default="data/bridge_tunnel/forkwall6k/train.pkl")
    p.add_argument("--meta", default="data/bridge_tunnel/forkwall6k/meta.pkl")
    p.add_argument("--out", default="paper/figures/forkwall_paper")
    p.add_argument("--ppo-ckpt", default="final_models/ppo/ppo_plain.pt")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    records, by_cat = load_records(args.maps)
    # NB: figures 1 & 2 are owned by paper_task_figs.py -- do not write them here,
    # or whichever script runs last silently wins.
    obs_info = fig_observation(by_cat, out)
    rew_info = fig_reward(by_cat, out, args.ppo_ckpt)
    ds_info = fig_dataset(records, out, args.meta)

    print("obs:", obs_info)
    print("reward:", rew_info)
    print("dataset:", ds_info["counts"])
    print("water:", ds_info["water"])
    print("rock: ", ds_info["rock"])
    print("wrote ->", out)


if __name__ == "__main__":
    main()
