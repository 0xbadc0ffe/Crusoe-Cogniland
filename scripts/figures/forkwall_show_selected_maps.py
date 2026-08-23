#!/usr/bin/env python3
"""Render the curated balanced fork_wall maps with baseline rollouts overlaid.

One column per selected map: the terrain, the doors coloured by correctness
(on balanced maps BOTH pay, so both are green), many stochastic baseline
trajectories, and the belief trace underneath. Titles carry the screening
statistics so the door habit and the terrain that drives it sit together.

    python scripts/figures/forkwall_show_selected_maps.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "bridge_tunnel"))
sys.path.insert(0, str(REPO / "scripts" / "mechinterp"))

from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from cogniland.bridge_tunnel.mapgen import generate_commit_map  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy  # noqa: E402
from eval_bridge_tunnel_forkwall_steered import batched_rollout_steered  # noqa: E402
from cogniland.bridge_tunnel.steering import BELIEF2I  # noqa: E402
from train_belief_probe import load_belief_probe  # noqa: E402

CATS = ["balanced", "lakes", "rocky"]
CAT_COLOR = {"balanced": "#5C6B57", "lakes": "#1E6FA6", "rocky": "#A3572A"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", type=Path,
                   default=REPO / "data/bridge_tunnel/forkwall_balanced_clean.json")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO / "outputs/ppo_checkpoints/ppo_gru_forkwall_noaux_seed1/final.pt")
    p.add_argument("--traj", type=int, default=40, help="rollouts drawn per map")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--out", type=Path,
                   default=REPO / "paper/figures/forkwall_balanced_selected.png")
    p.add_argument("--cols", type=int, default=0,
                   help="wrap into a grid of this many columns (0 = one wide strip). "
                        "Belief panels are dropped in grid mode to keep the file small")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    sel = json.load(open(args.selection))["selected"]
    print(f"rendering {len(sel)} maps from {args.selection.name}")

    policy, cargs, view_size, env_size, env_width = _load_policy(args.checkpoint, device)
    if getattr(policy, "belief", None) is None:
        lin, pmeta = load_belief_probe(args.checkpoint.parent / "belief_probe.pt", device)
        policy.belief = lin
        print(f"attached belief probe (balanced acc {pmeta['balanced_accuracy']:.3f})")
    commit = False if cargs.get("no_commit", False) else None
    gh = cargs.get("goal_half", 0)
    gh = gh if (gh is not None and gh >= 0) else None
    torch.manual_seed(0)

    n = len(sel)
    grid = args.cols > 0
    if grid:
        cols = args.cols
        rows = int(np.ceil(n / cols))
        fig, axgrid = plt.subplots(rows, cols, figsize=(3.1 * cols, 1.95 * rows))
        axgrid = np.asarray(axgrid).reshape(rows, cols)
        for k in range(n, rows * cols):
            axgrid[k // cols, k % cols].axis("off")
        axes = None
    else:
        fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 6.4),
                                 gridspec_kw={"height_ratios": [1.5, 1]})
        axes = np.asarray(axes).reshape(2, n)

    for k, meta in enumerate(sel):
        seed = meta["seed"]
        rec = generate_commit_map(size=env_size, width=env_width, seed=seed,
                                  category="balanced", tree_frac=cargs.get("tree_frac", 0.03),
                                  goal_half=gh, fork_wall=True,
                                  passage_half=cargs.get("passage_half", 1),
                                  wall_margin=cargs.get("wall_margin", 1))
        o = batched_rollout_steered(policy, rec, args.traj, view_size, args.max_steps,
                                    device, commit=commit, steer_fn=None)

        ax = axgrid[k // args.cols, k % args.cols] if grid else axes[0, k]
        ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
        # trajectories, coloured by the door each one ended at
        for i, tr in enumerate(o["trajs"]):
            pos = np.asarray(tr, dtype=float)
            if len(pos) < 2:
                continue
            jit = (np.random.rand(*pos.shape) - 0.5) * 0.7
            xy = np.stack([(pos + jit)[:, 1], (pos + jit)[:, 0]], axis=1)
            segs = np.stack([xy[:-1], xy[1:]], axis=1)
            col = {"top": "#c0392b", "bottom": "#1E6FA6"}.get(o["doors"][i], "#888888")
            ax.add_collection(LineCollection(segs, colors=col, linewidths=0.7, alpha=0.30))
        # on balanced maps EITHER door is correct -> both green
        for cells in (rec.top_goal_cells, rec.bottom_goal_cells):
            if cells:
                ys = [r for r, _ in cells]; xs = [c for _, c in cells]
                ax.scatter(xs, ys, c="lime", s=34, marker="s", edgecolors="k", zorder=6)
        ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=28, marker="o",
                   edgecolors="k", zorder=6)
        if rec.wall_col is not None:
            ax.axvline(rec.wall_col, color="white", lw=0.8, alpha=0.55)
        ax.set_xticks([]); ax.set_yticks([])
        lean = "TOP" if meta["top_frac"] >= 0.5 else "BOTTOM"
        if grid:
            ax.set_title(f"{seed} · {lean} {meta['top_frac']:.0%}t · "
                         f"r−w {meta['rock_minus_water']:+d}", fontsize=6.5)
        else:
            ax.set_title(f"seed {seed} — {lean}-leaning\n"
                         f"top {meta['top_frac']:.0%} / bottom {meta['bottom_frac']:.0%} · "
                         f"belief-flip {meta['flip_rate']:.0%} · succ {meta['success']:.0%}\n"
                         f"water {meta['n_water']} rock {meta['n_rock']} "
                         f"(rock−water {meta['rock_minus_water']:+d})", fontsize=8.5)

        if grid:
            continue
        ax = axes[1, k]
        bp = o["belief_probs"]
        valid = np.isfinite(bp[..., 0])
        alive = valid.sum(axis=0)
        keep = np.where(alive >= max(3, 0.1 * bp.shape[0]))[0]
        tmax = int(keep[-1]) + 1 if len(keep) else bp.shape[1]
        with np.errstate(invalid="ignore"):
            for ci, c in enumerate(CATS):
                mu = np.nanmean(bp[:, :tmax, ci], axis=0)
                lo = np.nanpercentile(bp[:, :tmax, ci], 25, axis=0)
                hi = np.nanpercentile(bp[:, :tmax, ci], 75, axis=0)
                ts = np.arange(tmax)
                ax.plot(ts, mu, color=CAT_COLOR[c], lw=1.6, label=c)
                ax.fill_between(ts, lo, hi, color=CAT_COLOR[c], alpha=0.15, linewidth=0)
        ax.axhline(1/3, color="gray", lw=0.7, ls=":")
        ax.set_ylim(0, 1.02); ax.set_xlabel("timestep")
        if k == 0:
            ax.set_ylabel("probe P(category | $h_t$)"); ax.legend(fontsize=8, loc="center right")

    fig.suptitle(
        f"Curated balanced fork_wall maps — {args.checkpoint.parent.name} · "
        f"{args.traj} baseline rollouts/map · trajectory colour = door reached "
        f"(red = top, blue = bottom) · both doors pay on balanced terrain",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
