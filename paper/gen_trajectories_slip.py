#!/usr/bin/env python3
"""Per-slip-regime trajectory glow grids.

For each checkpoint folder we render one 3x3 trajectory grid per map seed:
  rows = biome (lake / rocky / balanced),
  cols = PPO-GRU model by trained map size (32 / 64 / 96; 128 ignored).

The env's slip probability is patched to the folder's regime (0.90 / 0.75)
*before* map generation and rollout, so both the validated maps and the
runtime slip match how each model was trained. Each cell fixes one map
(shared spawn/target) and overlays 100 stochastic rollouts as thin
progress-coloured lines on a darkened map.

Output: paper/figures/trajectory_grids/slip_regime/traj_slip{90,75}_seed{seed}.png  (10 total)
"""
from __future__ import annotations

import glob
import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cogniland.nav.skills as sk  # noqa: E402  (patched per-folder)
from cogniland.nav import CognilandNavEnv  # noqa: E402
from cogniland.nav.mapgen import generate_map  # noqa: E402
from cogniland.nav.tiles import TILE_COLORS  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_ppo_gru", str(ROOT / "scripts" / "train_ppo_gru.py")
)
_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tp)
PPOGRUPolicy = _tp.PPOGRUPolicy

OUTDIR = Path(__file__).resolve().parent / "figures" / "trajectory_grids" / "slip_regime"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIZES = [32, 64, 96]                 # 128 ignored, per request
BIOMES = ["balanced", "rocky", "lake"]   # display row order (top -> bottom)
SEEDS = [7, 13, 21, 42, 77]          # 5 map seeds -> 5 grids per folder
N_RUNS = 100

# folder -> (slip probability, tag for filenames)
FOLDERS = {
    "ppo_slip_90": (0.90, "90"),
    "ppo_slip_75": (0.75, "75"),
}


def _find_ckpt(folder: str, size: int) -> Path:
    base = ROOT / "checkpoints" / folder
    for pat in (f"ppo_gru_size{size}_*_final.pt", f"ppo_map{size}.pt",
                f"*size{size}*.pt", f"*map{size}*.pt"):
        hits = sorted(glob.glob(str(base / pat)))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"no checkpoint for size {size} in {folder}")


def _load_policy(folder: str, size: int):
    ckpt = torch.load(_find_ckpt(folder, size), map_location=DEVICE, weights_only=False)
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
    max_steps = int(args.get("max_steps", 1000))
    envs = [CognilandNavEnv(size=size, view_size=args.get("view_size", 21),
                            obs_mode=args.get("obs_mode", "symbolic"),
                            max_steps=max_steps, seed=i, map_record=rec)
            for i in range(n)]
    obs = [e.reset()[0] for e in envs]
    paths = [[tuple(e._pos)] for e in envs]
    # per-step active object (0=none,1=raft,2=harness), aligned with paths
    objs = [[int(e._active_object)] for e in envs]
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
            o, _, term, trunc, _ = envs[i].step(
                {"move": int(a_np[i]),
                 "build_scalar": np.array([float(b_np[i])], dtype=np.float32)})
            obs[i] = o
            paths[i].append(tuple(envs[i]._pos))
            objs[i].append(int(envs[i]._active_object))
            if term or trunc:
                done[i] = True
                done_t[i] = 1.0
        if all(done):
            break
    # committed skill per run: 0=none, 1=raft, 2=harness (sk.NONE/RAFT/HARNESS)
    built = np.array([e._active_object for e in envs], dtype=int)
    for e in envs:
        e.close()
    return paths, objs, built


# trajectory colour by the agent's ACTIVE skill at each segment, sampled from
# plasma: no skill yet -> dark blue (low), raft -> orange, harness -> yellow.
_PLASMA = mpl.colormaps["plasma"]
_TRAJ_ALPHA = 0.10
_SKILL_RGBA = {
    0: _PLASMA(0.00),   # none  -> dark blue / purple (lowest plasma)
    1: _PLASMA(0.65),   # raft  -> orange
    2: _PLASMA(0.92),   # harness -> yellow
}
_SKILL_LEGEND = [("no skill", 0), ("raft", 1), ("harness", 2)]


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
        # segment k (path[k]->path[k+1]) coloured by the skill active after it
        cols = np.array([_SKILL_RGBA[ob[k + 1]] for k in range(len(segs))])
        cols[:, 3] = _TRAJ_ALPHA
        lc = LineCollection(segs, colors=cols, linewidths=0.6, capstyle="round")
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


# env active-object id (0=none,1=raft,2=harness) -> column (noskill,harness,raft)
_OBJ_TO_COL = {0: 0, 2: 1, 1: 2}
_SKILL_COLS = ["noskill", "harness", "raft"]
_MAT_ROWS = ["balanced", "rocky", "lake"]   # display row order (top->bottom)


def make_grid(folder: str, slip: float, tag: str, seed: int, policies):
    rng = np.random.default_rng(1000 + seed)
    nrows = len(BIOMES) + 1
    fig, axes = plt.subplots(nrows, len(SIZES),
                             figsize=(3.0 * len(SIZES), 3.0 * nrows))
    for j, sz in enumerate(SIZES):
        policy, args = policies[sz]
        skill_mat = np.zeros((len(BIOMES), 3), dtype=int)  # rows=biome, cols=skill
        for i, bio in enumerate(BIOMES):
            rec = generate_map(size=sz, map_type=bio, seed=seed, max_retries=400)
            paths, objs, built = _batched_rollouts(policy, args, rec, sz)
            for b in built:
                skill_mat[i, _OBJ_TO_COL[int(b)]] += 1
            ax = axes[i, j]
            _draw_cell(ax, rec, paths, objs, rng)
            tgt = (int(rec.target[0]), int(rec.target[1]))
            reached = sum(1 for p in paths if tuple(p[-1]) == tgt)
            if i == 0:
                ax.set_title(f"PPO-GRU · {sz}x{sz}", fontsize=12)
            if j == 0:
                ax.set_ylabel(bio, fontsize=13)
            ax.text(0.02, 0.02, f"{reached}/{N_RUNS}", transform=ax.transAxes,
                    fontsize=8, color="white", va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.5))

        # final row: skill-choice matrix, rows reordered to balanced/rocky/lake,
        # entries normalised to a row-fraction in [0,1]; raw trial counts in-cell.
        order = [BIOMES.index(b) for b in _MAT_ROWS]
        counts = skill_mat[order]                       # (3,3) raw counts
        totals = counts.sum(axis=1, keepdims=True)
        frac = counts / np.maximum(totals, 1)
        axm = axes[len(BIOMES), j]
        im = axm.imshow(frac, cmap="viridis", vmin=0, vmax=1, aspect="equal")
        axm.set_xticks(range(3))
        axm.set_xticklabels(_SKILL_COLS, fontsize=8, rotation=30, ha="right")
        axm.set_yticks(range(len(_MAT_ROWS)))
        axm.set_yticklabels(_MAT_ROWS if j == 0 else [""] * len(_MAT_ROWS), fontsize=8)
        for r in range(len(_MAT_ROWS)):
            for c in range(3):
                col = "white" if frac[r, c] < 0.55 else "black"
                axm.text(c, r - 0.13, f"{frac[r, c]:.2f}", ha="center", va="center",
                         fontsize=11, fontweight="bold", color=col)
                axm.text(c, r + 0.22, f"n={int(counts[r, c])}", ha="center", va="center",
                         fontsize=7, color=col)
        if j == 0:
            axm.set_ylabel("built skill\n(frac. per row, n/100)", fontsize=12)
        # colourbar to the right of the rightmost matrix
        if j == len(SIZES) - 1:
            cax = make_axes_locatable(axm).append_axes("right", size="7%", pad=0.08)
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("fraction of trials", fontsize=9)
            cb.ax.tick_params(labelsize=8)

    # legend for the trajectory skill colours (shared, below the suptitle)
    handles = [Line2D([0], [0], color=_SKILL_RGBA[k][:3], lw=3, label=lab)
               for lab, k in _SKILL_LEGEND]
    fig.legend(handles=handles, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.965), fontsize=10, frameon=False,
               title="trajectory colour = active skill")
    fig.suptitle(f"{folder}  ·  slip={slip:.2f}  ·  map seed {seed}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUTDIR / f"traj_slip{tag}_seed{seed}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for folder, (slip, tag) in FOLDERS.items():
        print(f"=== {folder}  (slip={slip}) ===")
        sk.SLIP_PROB_DEFAULT = slip          # patch map-gen + env runtime slip
        policies = {sz: _load_policy(folder, sz) for sz in SIZES}
        for seed in SEEDS:
            make_grid(folder, slip, tag, seed, policies)


if __name__ == "__main__":
    main()
