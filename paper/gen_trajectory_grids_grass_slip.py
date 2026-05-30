#!/usr/bin/env python3
"""Per-run trajectory-glow grids for the grass-slip sweep.

For each grass-slip run (``ppo_grass{00,05,...,30}`` — efficient.yaml agents
that differ only in the bare-handed no-skill grass slip probability), render
ONE grid: rows = biome (balanced/rocky/lake), cols = 5 map seeds. Each cell
fixes one map (shared spawn/target) and overlays ``N_RUNS=200`` stochastic
rollouts as thin lines coloured by the agent's active skill (no skill = dark,
raft = mid, harness = bright). The inset shows successes/N.

The agents train on ``simplex`` maps; pass ``--generator`` to pick which map
distribution to roll out on (``simplex`` = in-distribution / training,
``composed`` / ``components`` = held-out test set). Uses the *final*
checkpoint of each run under ``checkpoints/<exp>/``.

    # in-distribution (training) maps:
    python paper/gen_trajectory_grids_grass_slip.py --generator simplex
    # held-out test maps:
    python paper/gen_trajectory_grids_grass_slip.py --generator composed
"""
from __future__ import annotations

import argparse
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

EXP = "grass_slip_hardland_mixtrain"  # 2026-05-28: hard-land slip + simplex+components mixed training
CKPT_ROOT = ROOT / "checkpoints" / EXP
FIG_ROOT = ROOT / "paper" / "figures" / "trajectory_grids" / EXP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUNS = [f"ppo_grass{pct:02d}" for pct in (0, 5, 10, 15, 20, 25, 30)]
BIOMES = ["balanced", "rocky", "lake"]
SEEDS = [7, 13, 21, 42, 77]
N_RUNS = 200

_PLASMA = mpl.colormaps["plasma"]
_TRAJ_ALPHA = 0.06   # lower than the 100-run figure since we draw 2x as many
_SKILL_RGBA = {0: _PLASMA(0.00), 1: _PLASMA(0.65), 2: _PLASMA(0.92)}  # none/raft/harness
_LEGEND = [("no skill", 0), ("raft", 1), ("harness", 2)]


def _load_policy(run: str):
    ckpt = torch.load(CKPT_ROOT / run / "final.pt",
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


def make_grid(run: str, policy, args, generator: str, outdir: Path, split: str):
    size = int(args.get("env_size", 64))
    grass_slip = args.get("grass_slip_noskill", None)
    nrow, ncol = len(BIOMES), len(SEEDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.5 * nrow))
    axes = np.atleast_2d(axes)
    for i, biome in enumerate(BIOMES):
        for j, seed in enumerate(SEEDS):
            ax = axes[i, j]
            rec = generate_map(size=size, map_type=biome, seed=seed,
                               generator=generator, max_retries=400)
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
    gs_str = f"no-skill grass slip = {grass_slip:.0%}" if grass_slip is not None else ""
    fig.suptitle(f"{run}  ·  {gs_str}  ·  {split} maps ({generator})  ·  "
                 f"{N_RUNS} stochastic rollouts/cell", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{run}_seeds.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="composed",
                    choices=("simplex", "composed", "components"),
                    help="map distribution to roll out on (simplex = training, "
                         "composed/components = held-out test set)")
    args_cli = ap.parse_args()
    gen = args_cli.generator
    # In the mixtrain experiment, training = {simplex, composed}, test = components.
    TRAIN_GENS = {"simplex", "composed"}
    split = "train" if gen in TRAIN_GENS else "test"
    # Per-experiment, per-split subfolder: trajectory_grids/<EXP>/<split>_<gen>/
    outdir = FIG_ROOT / f"{split}_{gen}"
    print(f"generator={gen}  split={split}  outdir={outdir}")
    for run in RUNS:
        ckpt = CKPT_ROOT / run / "final.pt"
        if not ckpt.exists():
            print(f"[skip] {run}: {ckpt} not found")
            continue
        print(f"[gen ] {run} …", flush=True)
        policy, args = _load_policy(run)
        make_grid(run, policy, args, gen, outdir, split)
    print("done")


if __name__ == "__main__":
    main()
