#!/usr/bin/env python3
"""Build-scalar-over-time grid, matching traj_slip75_seed21.png.

For each (biome, model-size) cell of the slip=0.75 / seed=21 grid we run the
policy 100 times on the fixed map and record the deterministic belief head
output (``build_scalar`` in [-1,1]) at every step. Each subplot shows the 100
per-trial traces (transparent) plus the bold step-wise mean (wandb style).

Layout: rows = biome (balanced/rocky/lake), cols = size (32/64/96).
Output: paper/figures/build_scalar_grid_slip75_seed21.png
"""
from __future__ import annotations

import glob
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cogniland.nav.skills as sk  # noqa: E402
from cogniland.nav import CognilandNavEnv  # noqa: E402
from cogniland.nav.mapgen import generate_map  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_ppo_gru", str(ROOT / "scripts" / "train_ppo_gru.py"))
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)
PPOGRUPolicy = _tp.PPOGRUPolicy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGDIR = Path(__file__).resolve().parent / "figures"

FOLDER, SLIP, SEED = "ppo_slip_75", 0.75, 21
SIZES = [32, 64, 96]
BIOMES = ["balanced", "rocky", "lake"]
N_RUNS = 100
# build_scalar sign convention (env): >=0 -> raft, <0 -> harness
SKILL_REF = {"lake": (+1.0, "raft (+1)"), "rocky": (-1.0, "harness (-1)"),
             "balanced": (0.0, "no skill (0)")}


def _load_policy(size):
    f = sorted(glob.glob(str(ROOT / "checkpoints" / FOLDER / f"*{size}*.pt")))[0]
    ckpt = torch.load(f, map_location=DEVICE, weights_only=False)
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
def _belief_traces(policy, args, rec, size, n=N_RUNS):
    """Return list of per-trial build_scalar sequences (one per step)."""
    max_steps = int(args.get("max_steps", 1000))
    envs = [CognilandNavEnv(size=size, view_size=args.get("view_size", 21),
                            obs_mode=args.get("obs_mode", "symbolic"),
                            max_steps=max_steps, seed=i, map_record=rec)
            for i in range(n)]
    obs = [e.reset()[0] for e in envs]
    traces = [[] for _ in range(n)]
    done = [False] * n
    hidden = torch.zeros(1, n, policy.gru_hidden, device=DEVICE)
    done_t = torch.zeros(n, device=DEVICE)
    for _ in range(max_steps):
        sem = torch.as_tensor(np.stack([o["semantic"] for o in obs]), device=DEVICE)
        ska = torch.as_tensor(np.stack([o["skill_active"] for o in obs]), device=DEVICE)
        action, belief, _, _, _, hidden = policy.get_action_and_value(
            {"semantic": sem, "skill_active": ska}, hidden, done_t)
        a_np, b_np = action.cpu().numpy(), belief.cpu().numpy()
        for i in range(n):
            if done[i]:
                continue
            traces[i].append(float(b_np[i]))   # belief that drove this step
            o, _, term, trunc, _ = envs[i].step(
                {"move": int(a_np[i]),
                 "build_scalar": np.array([float(b_np[i])], dtype=np.float32)})
            obs[i] = o
            if term or trunc:
                done[i] = True
                done_t[i] = 1.0
        if all(done):
            break
    for e in envs:
        e.close()
    return traces


def _draw(ax, traces, biome):
    lengths = [len(t) for t in traces if t]
    if not lengths:
        return
    xmax = int(np.percentile(lengths, 97))
    xmax = max(xmax, 1)
    # ragged -> NaN-padded matrix for the step-wise mean
    L = max(lengths)
    M = np.full((len(traces), L), np.nan)
    for i, t in enumerate(traces):
        if t:
            M[i, :len(t)] = t
    for t in traces:
        if t:
            ax.plot(range(len(t)), t, color="#1f77b4", lw=0.5, alpha=0.06)
    mean = np.nanmean(M, axis=0)
    ax.plot(range(L), mean, color="crimson", lw=2.2, label="mean", zorder=5)
    # reference line for the correct commitment on this biome
    ref, lab = SKILL_REF[biome]
    ax.axhline(ref, color="black", ls="--", lw=0.9, alpha=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, xmax)
    ax.axhline(0.0, color="gray", lw=0.5, alpha=0.4)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    sk.SLIP_PROB_DEFAULT = SLIP
    policies = {sz: _load_policy(sz) for sz in SIZES}
    fig, axes = plt.subplots(len(BIOMES), len(SIZES),
                             figsize=(3.3 * len(SIZES), 2.7 * len(BIOMES)),
                             sharey=True)
    for j, sz in enumerate(SIZES):
        policy, args = policies[sz]
        for i, bio in enumerate(BIOMES):
            rec = generate_map(size=sz, map_type=bio, seed=SEED, max_retries=400)
            traces = _belief_traces(policy, args, rec, sz)
            ax = axes[i, j]
            _draw(ax, traces, bio)
            if i == 0:
                ax.set_title(f"PPO-GRU · {sz}x{sz}", fontsize=12)
            if i == len(BIOMES) - 1:
                ax.set_xlabel("step", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{bio}\nbuild_scalar", fontsize=11)
            ref, lab = SKILL_REF[bio]
            ax.text(0.97, 0.04 if ref < 0 else 0.96, lab, transform=ax.transAxes,
                    fontsize=8, ha="right", va="bottom" if ref < 0 else "top",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
            print(f"size {sz:3d} {bio:8s}: mean_final_belief="
                  f"{np.nanmean([t[-1] for t in traces if t]):+.2f}")
    fig.suptitle(f"build_scalar over steps  ·  {FOLDER}  ·  slip={SLIP}  ·  seed {SEED}"
                 f"   (100 trials, bold = mean)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = FIGDIR / "build_scalar_grid_slip75_seed21.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
