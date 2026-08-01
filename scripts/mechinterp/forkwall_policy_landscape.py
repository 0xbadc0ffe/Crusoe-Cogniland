#!/usr/bin/env python3
"""Fig-2 analogue: how the policy landscape sharpens over training.

The pendulum paper sweeps a 2-D observable state (theta, theta_dot) and colours
by torque. Our observation is a 21x21 one-hot crop plus scalars, which cannot be
enumerated — but the agent's POSITION on a fixed map is a genuine 2-D state, and
the fork_wall decision that matters is vertical (up -> top door, down -> bottom).

So: place the agent at every walkable cell of one fixed map, run the encoder and
one GRU step from a held hidden state, and colour by pi(up) - pi(down). Repeat
across checkpoints to watch the decision boundary form.

``--hidden zero`` (default) reads out the *reflexive* policy — what the network
does with no memory — which is the closest analogue to the paper's memoryless
state sweep. ``--hidden rollout`` instead uses, at each cell, the hidden state
the agent actually carries when it reaches that column, which folds the belief
back in and shows the memory-dependent landscape.

    python scripts/mechinterp/forkwall_policy_landscape.py --seed 1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "bridge_tunnel"))

from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv  # noqa: E402
from cogniland.bridge_tunnel.mapgen import generate_commit_map  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy  # noqa: E402

A_UP, A_DOWN = 0, 1
WALKABLE = (T.GRASS, T.WOOD, T.TARGET, T.SAND, T.DIRT)


@torch.no_grad()
def landscape(policy, rec, view_size, device, commit, hidden="zero", batch=256):
    """pi(up) - pi(down) at every walkable cell (NaN elsewhere)."""
    Hh, Ww = rec.terrain.shape
    env = BridgeTunnelCommitEnv(map_record=rec, size=Hh, width=Ww, view_size=view_size,
                                max_steps=400, commit=commit)
    env.reset()
    cells, obs = [], []
    for r in range(Hh):
        for c in range(Ww):
            if rec.terrain[r, c] not in WALKABLE:
                continue
            env._pos = [r, c]
            env._facing = 3
            o = env._make_obs()
            cells.append((r, c)); obs.append(o)
    if not cells:
        return np.full((Hh, Ww), np.nan)

    out = np.full((Hh, Ww), np.nan, dtype=np.float32)
    for i in range(0, len(cells), batch):
        chunk = obs[i:i + batch]
        mm = torch.from_numpy(np.stack([o["minimap"] for o in chunk]))[None].to(device)
        sc = torch.from_numpy(np.stack([o["scalars"] for o in chunk]))[None].to(device)
        h = torch.zeros(1, len(chunk), policy.gru_hidden, device=device)
        done = torch.zeros(len(chunk), device=device)
        _, h = policy._gru_forward({"minimap": mm, "scalars": sc}, done[None], h)
        logits, _ = policy._heads(h.squeeze(0))
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        for k, (r, c) in enumerate(cells[i:i + batch]):
            out[r, c] = p[k, A_UP] - p[k, A_DOWN]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-prefix", default="ppo_gru_forkwall_noaux_dense_seed")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--map-seed", type=int, default=200000)
    p.add_argument("--category", default="balanced")
    p.add_argument("--n-panels", type=int, default=6)
    p.add_argument("--ckpt-root", type=Path, default=REPO / "outputs/ppo_checkpoints")
    p.add_argument("--out", type=Path,
                   default=REPO / "paper/figures/forkwall_policy_landscape.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    d = args.ckpt_root / f"{args.run_prefix}{args.seed}"
    cks = sorted(d.glob("iter*.pt"),
                 key=lambda q: int(re.search(r"iter(\d+)", q.name).group(1)))
    cks.append(d / "final.pt")
    # log-spaced subset so the early, fast-moving phase is well covered
    pick = sorted({int(round(v)) for v in
                   np.geomspace(1, len(cks), num=args.n_panels)})
    cks = [cks[i - 1] for i in pick]
    print(f"seed {args.seed}: panels at {[c.name for c in cks]}")

    fig, axes = plt.subplots(2, len(cks), figsize=(3.05 * len(cks), 5.0),
                             gridspec_kw={"height_ratios": [1, 1]})
    axes = np.asarray(axes).reshape(2, len(cks))

    for k, ck in enumerate(cks):
        policy, cargs, view_size, env_size, env_width = _load_policy(ck, device)
        commit = False if cargs.get("no_commit", False) else None
        gh = cargs.get("goal_half", 0)
        gh = gh if (gh is not None and gh >= 0) else None
        rec = generate_commit_map(size=env_size, width=env_width, seed=args.map_seed,
                                  category=args.category,
                                  tree_frac=cargs.get("tree_frac", 0.03), goal_half=gh,
                                  fork_wall=True,
                                  passage_half=cargs.get("passage_half", 1),
                                  wall_margin=cargs.get("wall_margin", 1))
        L = landscape(policy, rec, view_size, device, commit)
        it = 0 if ck.name == "iter0.pt" else (
            "final" if ck.name == "final.pt"
            else int(re.search(r"iter(\d+)", ck.name).group(1)))

        ax = axes[0, k]
        ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
        for cells, ok in ((rec.top_goal_cells, True), (rec.bottom_goal_cells, True)):
            if cells:
                ys = [r for r, _ in cells]; xs = [c for _, c in cells]
                ax.scatter(xs, ys, c="lime", s=20, marker="s", edgecolors="k", zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{it} gradient steps", fontsize=9)

        ax = axes[1, k]
        im = ax.imshow(L, cmap="RdBu_r", vmin=-0.6, vmax=0.6, interpolation="nearest")
        if rec.wall_col is not None:
            ax.axvline(rec.wall_col, color="k", lw=0.8, alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_ylabel(r"$\pi(\rm up) - \pi(\rm down)$", fontsize=9)

    fig.colorbar(im, ax=list(axes[1]), fraction=0.02, pad=0.01,
                 label=r"$\pi(\rm up) - \pi(\rm down)$")
    fig.suptitle(f"Policy landscape over training — seed {args.seed}, "
                 f"{args.category} map (seed {args.map_seed}), memoryless (h = 0)\n"
                 f"red = pulls toward the TOP door, blue = toward the BOTTOM door",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.97, 0.90))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
