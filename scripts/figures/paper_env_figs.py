#!/usr/bin/env python3
"""Environment figures for the fork_wall paper.

Generates (into --out, default paper/figures/forkwall_paper/):
  fig_task_categories.png   one example map per category, doors + spawn annotated
  fig_task_anatomy.png      a single map with the corridor / wall / doors labelled
  fig_observation.png       agent view: full map + egocentric 21x21 crop + scalars
  fig_reward.png            reward decomposition along an optimal trajectory
  fig_dataset.png           dataset composition + per-category geometry stats

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


def fig_task_categories(by_cat, out):
    """One example map per category."""
    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.0))
        for ax, cat in zip(axes, CATS):
            rec = by_cat[cat][0]
            ax.imshow(rgb(rec.terrain), interpolation="nearest")
            annotate_map(ax, rec)
            ax.set_title(f"category = {cat}   →   rewarded door: {CAT_DOOR[cat]}",
                         loc="left")
            ax.set_xticks([]); ax.set_yticks([])
        handles = [
            Patch(facecolor=np.array(TILE_COLORS[1]) / 255, label="water (bridgeable)"),
            Patch(facecolor=np.array(TILE_COLORS[2]) / 255, label="rock (mineable)"),
            Patch(facecolor=np.array(TILE_COLORS[6]) / 255, label="tree (impassable)"),
            Patch(facecolor=np.array(TILE_COLORS[4]) / 255, label="door"),
            Line2D([], [], color="#22c55e", lw=2, label="rewarded door"),
            Line2D([], [], color="#ef4444", lw=2, label="decoy door"),
            Line2D([], [], marker="o", color="white", mec="black", ls="",
                   label="spawn"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
                   bbox_to_anchor=(0.5, -0.005), fontsize=8)
        fig.suptitle("fork_wall: terrain statistics identify the map category; "
                     "the category selects which door pays", y=0.995)
        fig.tight_layout(rect=[0, 0.035, 1, 0.97])
        fig.savefig(out / "fig_task_categories.png", bbox_inches="tight")
        plt.close(fig)


def fig_task_anatomy(by_cat, out):
    """A single map with the phases of an episode annotated + a zoom of the fork."""
    rec = by_cat["rocky"][1]
    H, W = rec.terrain.shape
    wall = rec.wall_col
    with plt.rc_context(PLT_RC):
        fig = plt.figure(figsize=(11.0, 4.3))
        gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.08)
        ax = fig.add_subplot(gs[0])
        ax.imshow(rgb(rec.terrain), interpolation="nearest")
        annotate_map(ax, rec, show_labels=False)
        mem_lo = max(0, wall - 16)
        ax.add_patch(Rectangle((mem_lo - .5, -.5), wall - mem_lo, H, facecolor="black",
                               alpha=.20, zorder=4))
        ax.axvline(wall, color="white", ls=":", lw=1.2, zorder=5)

        def note(axis, x, y, text):
            axis.annotate(text, xy=(x, y), fontsize=8.5, color="white", ha="center",
                          va="center", zorder=8,
                          bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=.7,
                                    ec="none"))
        note(ax, mem_lo / 2, 2.0, "1. evidence phase\n(terrain reveals category)")
        note(ax, (mem_lo + wall) / 2, 2.0, "2. memory corridor\n(16 cols, no information)")
        # zoom box around the fork region
        zx0 = wall - 3
        ax.add_patch(Rectangle((zx0 - .5, -.5), W - zx0, H, fill=False,
                               edgecolor="#facc15", lw=1.6, zorder=9))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("(a) full map: the category is only visible before the corridor",
                     loc="left")

        axz = fig.add_subplot(gs[1])
        axz.imshow(rgb(rec.terrain[:, zx0:]), interpolation="nearest", aspect="auto")
        for cells, name in ((rec.top_goal_cells, "top"),
                            (rec.bottom_goal_cells, "bottom")):
            good = rec.correct_target in ("either", name)
            for (r, c) in cells:
                axz.add_patch(Rectangle((c - zx0 - .5, r - .5), 1, 1, fill=False,
                                        edgecolor="#22c55e" if good else "#ef4444",
                                        lw=2.4, zorder=6))

        def arrow(axis, xy, text, dx, dy, color="white"):
            axis.annotate(text, xy=xy, xytext=(xy[0] + dx, xy[1] + dy),
                          fontsize=8, color=color, ha="center", va="center", zorder=9,
                          arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                          bbox=dict(boxstyle="round,pad=0.22", fc="black", alpha=.75,
                                    ec="none"))
        if rec.passage_cells:
            pr = [c[0] for c in rec.passage_cells]
            pc = rec.passage_cells[0][1] - zx0
            axz.add_patch(Rectangle((pc - .5, min(pr) - .5), 1, len(pr), fill=False,
                                    edgecolor="#38bdf8", lw=2.0, zorder=6))
            arrow(axz, (pc, float(np.mean(pr))), "3. passage\n(only way through)",
                  -2.6, 0, color="#38bdf8")
        top_r = rec.top_goal_cells[0][0] if rec.top_goal_cells else 4
        bot_r = rec.bottom_goal_cells[0][0] if rec.bottom_goal_cells else H - 4
        tc = rec.top_goal_cells[0][1] - zx0 if rec.top_goal_cells else 5
        bc = rec.bottom_goal_cells[0][1] - zx0 if rec.bottom_goal_cells else 5
        arrow(axz, (tc, top_r), "4a. top door\n(rewarded: rocky)", -2.4, -4.5,
              color="#22c55e")
        arrow(axz, (bc, bot_r), "4b. bottom door\n(decoy)", -2.4, 4.5, color="#ef4444")
        axz.set_xticks([]); axz.set_yticks([])
        for s in axz.spines.values():
            s.set_edgecolor("#facc15"); s.set_linewidth(1.6); s.set_visible(True)
        axz.set_title("(b) the fork (zoom)", loc="left")

        fig.suptitle("Anatomy of an episode — the agent must carry the category "
                     "across an information-free corridor", y=1.02, fontsize=10)
        fig.savefig(out / "fig_task_anatomy.png", bbox_inches="tight")
        plt.close(fig)


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
    fig_task_categories(by_cat, out)
    fig_task_anatomy(by_cat, out)
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
