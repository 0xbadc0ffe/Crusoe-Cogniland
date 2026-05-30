#!/usr/bin/env python3
"""Trajectory-glow grid: rows = biome, cols = PPO model (by trained map size).

For every (biome, model) cell we:
  1. generate ONE validated map at the model's trained size + biome (fixed
     seed), so spawn/target are identical across all 100 runs in the cell;
  2. roll the *stochastic* policy (sampled moves) 100 times in parallel,
     batched through the GRU, on that single fixed map;
  3. darken the terrain and overlay the 100 paths as thin lines coloured by
     step-progress (plasma) at low alpha — overlapping paths accumulate into
     a glowing bundle.

Each model is evaluated only on the map size it was trained on (32/64/96/128).
"""
from __future__ import annotations

import glob
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cogniland.nav import CognilandNavEnv  # noqa: E402
from cogniland.nav.mapgen import generate_map  # noqa: E402
from cogniland.nav.tiles import TILE_COLORS  # noqa: E402

# import PPOGRUPolicy from the trainer script without packaging it
_spec = importlib.util.spec_from_file_location(
    "train_ppo_gru", str(ROOT / "scripts" / "train_ppo_gru.py")
)
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)
PPOGRUPolicy = _tp.PPOGRUPolicy

FIGDIR = Path(__file__).resolve().parent / "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZES = [32, 64, 96, 128]
BIOMES = ["lake", "rocky", "balanced"]
N_RUNS = 100
MAP_SEED = 7  # same seed used for the static size×biome grid


def _find_ckpt(size: int) -> Path:
    hits = glob.glob(str(ROOT / "checkpoints" / f"ppo_gru_size{size}_*_final.pt"))
    if not hits:
        raise FileNotFoundError(f"no checkpoint for size {size}")
    return Path(sorted(hits)[0])


def _load_policy(size: int):
    ckpt = torch.load(_find_ckpt(size), map_location=DEVICE, weights_only=False)
    a = dict(ckpt.get("args", {}))
    probe = CognilandNavEnv(size=size, map_type="lake",
                            view_size=a.get("view_size", 21),
                            obs_mode=a.get("obs_mode", "symbolic"),
                            max_steps=a.get("max_steps", 1000), seed=0)
    policy = PPOGRUPolicy(probe.observation_space,
                          num_move_actions=probe.action_space["move"].n,
                          gru_hidden=a.get("gru_hidden", 128),
                          embed_dim=a.get("embed_dim", 256)).to(DEVICE)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    probe.close()
    return policy, a


@torch.no_grad()
def _batched_rollouts(policy, args, rec, size, n=N_RUNS):
    """Run ``n`` stochastic rollouts on the fixed map ``rec``; return paths."""
    max_steps = int(args.get("max_steps", 1000))
    envs = [CognilandNavEnv(size=size, view_size=args.get("view_size", 21),
                            obs_mode=args.get("obs_mode", "symbolic"),
                            max_steps=max_steps, seed=i, map_record=rec)
            for i in range(n)]
    obs = [e.reset()[0] for e in envs]
    paths = [[tuple(e._pos)] for e in envs]
    done = [False] * n

    hidden = torch.zeros(1, n, policy.gru_hidden, device=DEVICE)
    done_t = torch.zeros(n, device=DEVICE)

    for _ in range(max_steps):
        sem = torch.as_tensor(np.stack([o["semantic"] for o in obs]), device=DEVICE)
        ska = torch.as_tensor(np.stack([o["skill_active"] for o in obs]), device=DEVICE)
        obs_t = {"semantic": sem, "skill_active": ska}
        action, belief, _, _, _, hidden = policy.get_action_and_value(obs_t, hidden, done_t)
        a_np = action.cpu().numpy()
        b_np = belief.cpu().numpy()
        for i in range(n):
            if done[i]:
                continue
            step = {"move": int(a_np[i]),
                    "build_scalar": np.array([float(b_np[i])], dtype=np.float32)}
            o, _, term, trunc, _ = envs[i].step(step)
            obs[i] = o
            paths[i].append(tuple(envs[i]._pos))
            if term or trunc:
                done[i] = True
                done_t[i] = 1.0
        if all(done):
            break
    for e in envs:
        e.close()
    return paths


def _draw_cell(ax, rec, paths):
    rgb = TILE_COLORS[rec.terrain].astype(np.float32)
    ax.imshow((rgb * 0.40).astype(np.uint8), interpolation="nearest")

    for p in paths:
        p = np.asarray(p, dtype=np.float32)
        if len(p) < 2:
            continue
        # tiny jitter so coincident segments don't perfectly overplot
        p = p + np.random.uniform(-0.18, 0.18, size=p.shape)
        xy = p[:, ::-1]  # (col, row) for plotting
        segs = np.concatenate([xy[:-1, None, :], xy[1:, None, :]], axis=1)
        t = np.linspace(0.0, 1.0, len(segs))
        lc = LineCollection(segs, cmap="plasma", array=t, linewidths=0.6,
                            alpha=0.10, capstyle="round")
        ax.add_collection(lc)

    sr, sc = rec.spawn
    tr, tc = rec.target
    ax.scatter([sc], [sr], marker="o", s=40, facecolor="#39ff14",
               edgecolor="black", linewidth=0.7, zorder=6)
    ax.scatter([tc], [tr], marker="*", s=110, facecolor="white",
               edgecolor="black", linewidth=0.7, zorder=6)
    ax.set_xlim(-0.5, rec.terrain.shape[1] - 0.5)
    ax.set_ylim(rec.terrain.shape[0] - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    policies = {sz: _load_policy(sz) for sz in SIZES}
    fig, axes = plt.subplots(len(BIOMES), len(SIZES),
                             figsize=(3.0 * len(SIZES), 3.0 * len(BIOMES)))
    for j, sz in enumerate(SIZES):
        policy, args = policies[sz]
        for i, bio in enumerate(BIOMES):
            rec = generate_map(size=sz, map_type=bio, seed=MAP_SEED)
            paths = _batched_rollouts(policy, args, rec, sz)
            ax = axes[i, j]
            _draw_cell(ax, rec, paths)
            reached = sum(1 for p in paths
                          if tuple(p[-1]) == (int(rec.target[0]), int(rec.target[1])))
            if i == 0:
                ax.set_title(f"PPO-GRU · {sz}x{sz}", fontsize=12)
            if j == 0:
                ax.set_ylabel(bio, fontsize=13)
            ax.text(0.02, 0.02, f"{reached}/{N_RUNS} reach",
                    transform=ax.transAxes, fontsize=8, color="white",
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.5))
            print(f"size {sz:3d} {bio:8s}: {reached}/{N_RUNS} reached")
    fig.tight_layout()
    p = FIGDIR / "grid_trajectories.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
