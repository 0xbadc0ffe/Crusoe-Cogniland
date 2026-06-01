#!/usr/bin/env python3
"""Evaluate a trained bridge_tunnel_commit PPO agent: the 3×3 belief→skill commit
matrix + a per-category trajectory grid.

The **commit matrix** is the headline interp artifact: rows = map category
(balanced / lakes / rocky = the *belief* the agent must infer), columns = the
skill it irreversibly committed to (none / commit_build / commit_mine). Each
entry is the fraction of evaluated episodes in that category that ended in that
commitment, so **each row sums to 1**. A competent, belief-reading agent commits
BUILD on lakes maps and MINE on rocky maps; balanced is a freer choice.

The **trajectory grid** overlays many stochastic rollouts per map (rows =
category, cols = seeds): path = dark blue, MINE cells = yellow, BUILD/bridge
cells = red, the commit step = a magenta ★. Titles show success + the map's
commit split.

    python scripts/eval_bridge_tunnel_commit_ppo.py \\
        --checkpoint checkpoints/ppo_commit_onehot_<ts>/final.pt \\
        --out-prefix paper/figures/bridge_tunnel_commit/ppo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.bridge_tunnel_commit import generate_commit_map, tiles as T  # noqa: E402
from cogniland.bridge_tunnel_commit.env import BridgeTunnelCommitEnv  # noqa: E402
from cogniland.bridge_tunnel_commit.mapgen import CATEGORIES  # noqa: E402
from train_ppo_bridge_tunnel_commit import PPOGRUPolicy  # noqa: E402

_FACE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
COMMIT_NAMES = ["none", "commit_build", "commit_mine"]


def _load_policy(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cargs = ckpt["args"]
    env_size = cargs.get("env_size", 32)
    env_width = cargs.get("env_width") or 64
    view_size = cargs.get("view_size", 21)
    dummy = BridgeTunnelCommitEnv(size=env_size, width=env_width, view_size=view_size)
    dummy.reset()
    sd = ckpt["policy"]
    obs_enc = cargs.get("obs_encoding", "embed")
    if "tile_embed.weight" in sd:
        n_tiles = int(sd["tile_embed.weight"].shape[0])
    else:
        n_tiles = int(sd["cnn.0.weight"].shape[1]) - 2
        obs_enc = "onehot"
    n_act = int(sd["actor.weight"].shape[0])
    policy = PPOGRUPolicy(dummy.observation_space, num_actions=n_act,
                          gru_hidden=cargs.get("gru_hidden", 128),
                          embed_dim=cargs.get("embed_dim", 256),
                          num_tile_classes=n_tiles, obs_encoding=obs_enc).to(device)
    policy.load_state_dict(sd)
    policy.eval()
    return policy, cargs, view_size, env_size, env_width


@torch.no_grad()
def batched_rollout(policy, rec, n_traj, view_size, max_steps, device):
    """``n_traj`` stochastic rollouts on one fixed map in lockstep. Returns
    (trajs, reached[bool], final_commit[int], commit_pts[list], mine_pts, bridge_pts)."""
    H, W = rec.terrain.shape
    envs = [BridgeTunnelCommitEnv(map_record=rec, size=H, width=W, view_size=view_size,
                                  max_steps=max_steps) for _ in range(n_traj)]
    obs = [e.reset()[0] for e in envs]
    h = torch.zeros(1, n_traj, policy.gru_hidden, device=device)
    done = torch.zeros(n_traj, device=device)
    active = np.ones(n_traj, dtype=bool)
    reached = np.zeros(n_traj, dtype=bool)
    final_commit = np.zeros(n_traj, dtype=np.int64)
    trajs = [[tuple(e._pos)] for e in envs]
    commit_pts, mine_pts, bridge_pts = [], [], []

    for _ in range(max_steps):
        mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
        sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
        gru_out, h = policy._gru_forward({"minimap": mm, "scalars": sc}, done[None], h)
        logits, _ = policy._heads(gru_out.squeeze(0))
        acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            o, r, term, trunc, info = e.step(int(acts[i]))
            obs[i] = o
            trajs[i].append(tuple(e._pos))
            final_commit[i] = info["commit"]
            if info["committed_now"]:
                commit_pts.append(tuple(e._pos))
            if info["mined"] or info["placed"]:
                dr, dc = _FACE_DELTA[info["facing"]]
                cell = (e._pos[0] + dr, e._pos[1] + dc)
                (mine_pts if info["mined"] else bridge_pts).append(cell)
            if term:
                reached[i] = True; active[i] = False
            elif trunc:
                active[i] = False
        done = torch.zeros(n_traj, device=device)
        if not active.any():
            break
    return trajs, reached, final_commit, commit_pts, mine_pts, bridge_pts


def compute_matrix(policy, view_size, env_size, env_width, cargs, device,
                   n_maps, n_traj, seed_start, max_steps):
    """3×3 counts → row-normalized commit matrix + per-category success."""
    counts = np.zeros((3, 3), dtype=np.float64)        # [category, commit]
    succ = {c: [] for c in CATEGORIES}
    gh = cargs.get("goal_half", 1)
    for ci, cat in enumerate(CATEGORIES):
        for j in range(n_maps):
            rec = generate_commit_map(size=env_size, width=env_width,
                                      seed=seed_start + j, category=cat,
                                      tree_frac=cargs.get("tree_frac", 0.03),
                                      goal_half=(gh if (gh is not None and gh >= 0) else None))
            _, reached, fcommit, *_ = batched_rollout(
                policy, rec, n_traj, view_size, max_steps, device)
            for v in fcommit:
                counts[ci, int(v)] += 1
            succ[cat].extend(reached.tolist())
    matrix = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    succ = {c: float(np.mean(v)) if v else 0.0 for c, v in succ.items()}
    return matrix, succ, counts


def plot_matrix(matrix, succ, counts, title, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(COMMIT_NAMES, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{c}\n(succ {succ[c]:.0%})" for c in CATEGORIES], fontsize=10)
    ax.set_xlabel("committed skill", fontsize=11)
    ax.set_ylabel("map category (belief)", fontsize=11)
    for i in range(3):
        for j in range(3):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12,
                    color="white" if v < 0.6 else "black", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of episodes")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


def plot_grid(policy, view_size, env_size, env_width, cargs, device,
              n_seeds, n_traj, seed_start, max_steps, title, out_path):
    gh = cargs.get("goal_half", 1)
    fig, axes = plt.subplots(len(CATEGORIES), n_seeds,
                             figsize=(n_seeds * 3.0, len(CATEGORIES) * 2.0))
    axes = np.atleast_2d(axes)
    for ci, cat in enumerate(CATEGORIES):
        for sj in range(n_seeds):
            rec = generate_commit_map(size=env_size, width=env_width,
                                      seed=seed_start + sj, category=cat,
                                      tree_frac=cargs.get("tree_frac", 0.03),
                                      goal_half=(gh if (gh is not None and gh >= 0) else None))
            trajs, reached, fcommit, commit_pts, mine_pts, bridge_pts = batched_rollout(
                policy, rec, n_traj, view_size, max_steps, device)
            ax = axes[ci, sj]
            ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
            for i, tr in enumerate(trajs):
                a = np.array(tr)
                jit = (np.random.rand(*a.shape) - 0.5) * 0.6
                ax.plot(a[:, 1] + jit[:, 1], a[:, 0] + jit[:, 0],
                        color="darkblue", lw=0.6, alpha=0.04 if reached[i] else 0.07)
            if mine_pts:
                m = np.array(mine_pts); ax.scatter(m[:, 1], m[:, 0], color="yellow", s=6, alpha=0.18, zorder=3, linewidths=0)
            if bridge_pts:
                b = np.array(bridge_pts); ax.scatter(b[:, 1], b[:, 0], color="red", s=6, alpha=0.18, zorder=3, linewidths=0)
            if commit_pts:
                cp = np.array(commit_pts); ax.scatter(cp[:, 1], cp[:, 0], color="magenta", s=18, marker="*", alpha=0.25, zorder=4, linewidths=0)
            ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=22, marker="s", edgecolors="k", zorder=5)
            fb = float((fcommit == 1).mean()); fm = float((fcommit == 2).mean()); fn = float((fcommit == 0).mean())
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{cat} s{seed_start+sj}  succ {reached.mean():.0%}\n"
                         f"build {fb:.0%}/mine {fm:.0%}/none {fn:.0%}", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, default=Path("paper/figures/bridge_tunnel_commit/ppo"))
    p.add_argument("--matrix-maps", type=int, default=20, help="held-out maps/category for the matrix")
    p.add_argument("--matrix-traj", type=int, default=16, help="stochastic rollouts/map for the matrix")
    p.add_argument("--grid-seeds", type=int, default=4, help="map columns in the trajectory grid")
    p.add_argument("--grid-traj", type=int, default=120, help="rollouts/map in the grid")
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    policy, cargs, view_size, env_size, env_width = _load_policy(args.checkpoint, device)
    tag = args.checkpoint.parent.name

    matrix, succ, counts = compute_matrix(
        policy, view_size, env_size, env_width, cargs, device,
        args.matrix_maps, args.matrix_traj, args.eval_seed_start, args.max_steps)
    print("commit matrix (rows=category, cols=none/build/mine):")
    for i, c in enumerate(CATEGORIES):
        print(f"  {c:9s} {matrix[i]}  succ={succ[c]:.2%}")
    plot_matrix(matrix, succ, counts,
                f"PPO+GRU  ·  belief→skill commit matrix\n{tag}",
                Path(str(args.out_prefix) + "_commit_matrix.png"))
    plot_grid(policy, view_size, env_size, env_width, cargs, device,
              args.grid_seeds, args.grid_traj, args.eval_seed_start, args.max_steps,
              f"PPO+GRU bridge_tunnel_commit  ·  {tag}  ·  {args.grid_traj} stochastic rollouts/map  ·  "
              f"path=blue mine=yellow bridge=red commit=★",
              Path(str(args.out_prefix) + "_traj.png"))


if __name__ == "__main__":
    main()
