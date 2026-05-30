#!/usr/bin/env python3
"""Per-run trajectory-glow grids across 5 map seeds.

For each of the 4 trained checkpoints (ppo_{diverse,efficient}_{aux,noaux}),
render ONE grid: rows = biome (balanced/rocky/lake), cols = 5 map seeds. Each
cell fixes one composed map (shared spawn/target) at the model's trained size
and overlays ``N_RUNS`` stochastic rollouts as thin lines coloured by the
agent's active skill (no skill = dark, raft = orange, harness = yellow). The
inset shows successes/N.

Uses the *final* checkpoint of each run. Run after all 4 trainings finish:
    python paper/gen_trajectory_grids_runs.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cogniland.nav import CognilandNavEnv  # noqa: E402
from cogniland.nav.mapgen import generate_map  # noqa: E402
from cogniland.nav.tiles import TILE_COLORS  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_ppo_gru", str(ROOT / "scripts" / "train_ppo_gru.py"))
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)
PPOGRUPolicy = _tp.PPOGRUPolicy

OUTDIR = ROOT / "paper" / "figures" / "trajectory_grids" / "config_aux_ablation"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUNS = ["ppo_diverse_aux", "ppo_diverse_noaux",
        "ppo_efficient_aux", "ppo_efficient_noaux"]
BIOMES = ["balanced", "rocky", "lake"]
SEEDS = [7, 13, 21, 42, 77]
N_RUNS = 100

_PLASMA = mpl.colormaps["plasma"]
_TRAJ_ALPHA = 0.10
_SKILL_RGBA = {0: _PLASMA(0.00), 1: _PLASMA(0.65), 2: _PLASMA(0.92)}  # none/raft/harness
_LEGEND = [("no skill", 0), ("raft", 1), ("harness", 2)]


def _load_policy(run: str):
    ckpt = torch.load(ROOT / "checkpoints" / run / "final.pt",
                      map_location=DEVICE, weights_only=False)
    a = dict(ckpt.get("args", {}))
    probe = CognilandNavEnv(size=a.get("env_size", 64), map_type="lake",
                            view_size=a.get("view_size", 21),
                            obs_mode=a.get("obs_mode", "symbolic"),
                            max_steps=a.get("max_steps", 1000), seed=0)
    policy = PPOGRUPolicy(probe.observation_space,
                          num_move_actions=probe.action_space.n,   # Discrete(6)
                          gru_hidden=a.get("gru_hidden", 128),
                          embed_dim=a.get("embed_dim", 256)).to(DEVICE)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    probe.close()
    return policy, a


@torch.no_grad()
def _rollouts(policy, args, rec, size, n=N_RUNS):
    max_steps = int(args.get("max_steps", 1000))
    envs = [CognilandNavEnv(size=size, view_size=args.get("view_size", 21),
                            obs_mode=args.get("obs_mode", "symbolic"),
                            max_steps=max_steps, seed=i, map_record=rec)
            for i in range(n)]
    obs = [e.reset()[0] for e in envs]
    paths = [[tuple(e._pos)] for e in envs]
    objs = [[int(e._active_object)] for e in envs]
    done = [False] * n
    reached = [False] * n
    hidden = torch.zeros(1, n, policy.gru_hidden, device=DEVICE)
    done_t = torch.zeros(n, device=DEVICE)
    for _ in range(max_steps):
        sem = torch.as_tensor(np.stack([o["semantic"] for o in obs]), device=DEVICE)
        ska = torch.as_tensor(np.stack([o["skill_active"] for o in obs]), device=DEVICE)
        action, _, _, _, _, hidden = policy.get_action_and_value(
            {"semantic": sem, "skill_active": ska}, hidden, done_t)
        a_np = action.cpu().numpy()
        for i in range(n):
            if done[i]:
                continue
            o, _, term, trunc, info = envs[i].step(int(a_np[i]))  # Discrete action
            obs[i] = o
            paths[i].append(tuple(envs[i]._pos))
            objs[i].append(int(envs[i]._active_object))
            if term or trunc:
                done[i] = True
                done_t[i] = 1.0
                reached[i] = bool(info.get("reached_target", False))
        if all(done):
            break
    for e in envs:
        e.close()
    return paths, objs, int(sum(reached))


def _draw_cell(ax, rec, paths, objs, rng):
    rgb = TILE_COLORS[rec.terrain].astype(np.float32)
    ax.imshow((rgb * 0.40).astype(np.uint8), interpolation="nearest")
    for p, ob in zip(paths, objs):
        p = np.asarray(p, dtype=np.float32)
        if len(p) < 2:
            continue
        p = p + rng.uniform(-0.18, 0.18, size=p.shape)
        xy = p[:, ::-1]
        segs = np.concatenate([xy[:-1, None, :], xy[1:, None, :]], axis=1)
        cols = np.array([_SKILL_RGBA[ob[k + 1]] for k in range(len(segs))])
        cols[:, 3] = _TRAJ_ALPHA
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.6, capstyle="round"))
    sr, sc = rec.spawn
    tr, tc = rec.target
    ax.scatter([sc], [sr], marker="o", s=34, facecolor="#39ff14", edgecolor="black", lw=0.6, zorder=6)
    ax.scatter([tc], [tr], marker="*", s=95, facecolor="white", edgecolor="black", lw=0.6, zorder=6)
    ax.set_xlim(-0.5, rec.terrain.shape[1] - 0.5)
    ax.set_ylim(rec.terrain.shape[0] - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])


def make_grid(run: str, policy, args):
    size = int(args.get("env_size", 64))
    nrow, ncol = len(BIOMES), len(SEEDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.5 * nrow))
    axes = np.atleast_2d(axes)
    for i, biome in enumerate(BIOMES):
        for j, seed in enumerate(SEEDS):
            ax = axes[i, j]
            rec = generate_map(size=size, map_type=biome, seed=seed, max_retries=400)
            paths, objs, succ = _rollouts(policy, args, rec, size)
            _draw_cell(ax, rec, paths, objs, np.random.default_rng(1000 + seed))
            ax.text(0.03, 0.97, f"{succ}/{N_RUNS}", transform=ax.transAxes,
                    fontsize=8, va="top", ha="left", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
            if i == 0:
                ax.set_title(f"seed {seed}", fontsize=10)
            if j == 0:
                ax.set_ylabel(biome, fontsize=11)
    handles = [Line2D([0], [0], color=_SKILL_RGBA[k], lw=3, label=lbl) for lbl, k in _LEGEND]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=9, framealpha=0.9)
    fig.suptitle(f"{run}  ·  final checkpoint  ·  {N_RUNS} stochastic rollouts/cell",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / f"{run}_seeds.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def main():
    for run in RUNS:
        ckpt = ROOT / "checkpoints" / run / "final.pt"
        if not ckpt.exists():
            print(f"[skip] {run}: {ckpt} not found")
            continue
        print(f"[gen ] {run} …", flush=True)
        policy, args = _load_policy(run)
        make_grid(run, policy, args)
    print("done")


if __name__ == "__main__":
    main()
