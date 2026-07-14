#!/usr/bin/env python3
"""Evaluate a fork_wall btc PPO agent: per-category success table + trajectory grid.

fork_wall task: spawn -> corridor with category-revealing terrain (lakes/rocky/
balanced) -> a wall with a 3-cell passage -> two single-cell doors (top/bottom).
Only the door matching the category (lakes->bottom, rocky->top, balanced->either)
pays the reach bonus / counts as success; the other door still ends the episode
but with no bonus (a "wrong belief" outcome, distinct from timing out).

    python scripts/bridge_tunnel/eval_bridge_tunnel_forkwall.py \\
        --checkpoint outputs/ppo_checkpoints/ppo_gru_commit_forkwall/final.pt \\
        --out-prefix outputs/bridge_tunnel_forkwall/ppo
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
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cogniland.bridge_tunnel.mapgen import generate_commit_map, CATEGORIES  # noqa: E402
from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv  # noqa: E402
from cogniland.bridge_tunnel.policy import PPOGRUPolicy  # noqa: E402

_FACE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
COMMIT_NAMES = ["none", "commit_build", "commit_mine"]
_COMMIT_COLORS = {0: "#1f5fd0", 1: "#ffd000", 2: "#a800e6"}


def _load_policy(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cargs = ckpt["args"]
    env_size = cargs.get("env_size", 32)
    env_width = cargs.get("env_width") or 64
    view_size = cargs.get("view_size", 21)
    dummy = BridgeTunnelCommitEnv(size=env_size, width=env_width, view_size=view_size,
                                  fork_wall=cargs.get("fork_wall", True),
                                  goal_half=(cargs.get("goal_half", 0) if cargs.get("goal_half", 0) >= 0 else None))
    dummy.reset()
    sd = ckpt["policy"]
    obs_enc = cargs.get("obs_encoding", "embed")
    if "tile_embed.weight" in sd:
        n_tiles = int(sd["tile_embed.weight"].shape[0])
    else:
        n_tiles = int(sd["cnn.0.weight"].shape[1]) - 2
        obs_enc = "onehot"
    n_act = int(sd["actor.weight"].shape[0])
    belief_classes = int(sd["belief.weight"].shape[0]) if "belief.weight" in sd else 0
    policy = PPOGRUPolicy(dummy.observation_space, num_actions=n_act,
                          gru_hidden=cargs.get("gru_hidden", 128),
                          embed_dim=cargs.get("embed_dim", 256),
                          num_tile_classes=n_tiles, obs_encoding=obs_enc,
                          belief_classes=belief_classes).to(device)
    policy.load_state_dict(sd)
    policy.eval()
    return policy, cargs, view_size, env_size, env_width


@torch.no_grad()
def batched_rollout(policy, rec, n_traj, view_size, max_steps, device):
    """``n_traj`` stochastic rollouts on one fixed fork_wall map, in lockstep."""
    H, W = rec.terrain.shape
    envs = [BridgeTunnelCommitEnv(map_record=rec, size=H, width=W, view_size=view_size,
                                  max_steps=max_steps) for _ in range(n_traj)]
    obs = [e.reset()[0] for e in envs]
    h = torch.zeros(1, n_traj, policy.gru_hidden, device=device)
    done = torch.zeros(n_traj, device=device)
    active = np.ones(n_traj, dtype=bool)
    success = np.zeros(n_traj, dtype=bool)       # reached the CORRECT door
    reached_any = np.zeros(n_traj, dtype=bool)    # reached EITHER door
    final_commit = np.zeros(n_traj, dtype=np.int64)
    final_pos = [None] * n_traj
    trajs = [[tuple(e._pos)] for e in envs]
    commits = [[0] for _ in envs]
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
            commits[i].append(int(info["commit"]))
            final_commit[i] = info["commit"]
            if info["committed_now"]:
                commit_pts.append(tuple(e._pos))
            if info["mined"] or info["placed"]:
                dr, dc = _FACE_DELTA[info["facing"]]
                cell = (e._pos[0] + dr, e._pos[1] + dc)
                (mine_pts if info["mined"] else bridge_pts).append(cell)
            if term:
                success[i] = bool(info["reached_target"])
                reached_any[i] = bool(info.get("reached_any_target", info["reached_target"]))
                final_pos[i] = e._pos
                active[i] = False
            elif trunc:
                active[i] = False
        done = torch.zeros(n_traj, device=device)
        if not active.any():
            break
    return dict(trajs=trajs, commits=commits, success=success, reached_any=reached_any,
                final_commit=final_commit, final_pos=final_pos,
                commit_pts=commit_pts, mine_pts=mine_pts, bridge_pts=bridge_pts)


def _door_of(rec, pos):
    if pos is None:
        return "none"
    if pos in rec.top_goal_cells:
        return "top"
    if pos in rec.bottom_goal_cells:
        return "bottom"
    return "other"


def evaluate(policy, view_size, env_size, env_width, cargs, device,
            n_maps, n_traj, seed_start, max_steps, passage_half, wall_margin):
    gh = cargs.get("goal_half", 0)
    gh = gh if (gh is not None and gh >= 0) else None
    rows = {}
    per_cat_rollouts = {}
    for cat in CATEGORIES:
        succ, wrong, timeout, commit_b, commit_m, commit_n, lens = [], [], [], [], [], [], []
        maps_rollouts = []
        for j in range(n_maps):
            rec = generate_commit_map(size=env_size, width=env_width, seed=seed_start + j,
                                      category=cat, tree_frac=cargs.get("tree_frac", 0.03),
                                      goal_half=gh, fork_wall=True,
                                      passage_half=passage_half, wall_margin=wall_margin)
            out = batched_rollout(policy, rec, n_traj, view_size, max_steps, device)
            succ.extend(out["success"].tolist())
            reached_any = out["reached_any"]
            wrong.extend((reached_any & ~out["success"]).tolist())
            timeout.extend((~reached_any).tolist())
            commit_b.extend((out["final_commit"] == 1).tolist())
            commit_m.extend((out["final_commit"] == 2).tolist())
            commit_n.extend((out["final_commit"] == 0).tolist())
            lens.extend([len(t) - 1 for t in out["trajs"]])
            maps_rollouts.append((rec, out))
        rows[cat] = dict(
            success=float(np.mean(succ)), wrong_door=float(np.mean(wrong)),
            timeout=float(np.mean(timeout)), commit_build=float(np.mean(commit_b)),
            commit_mine=float(np.mean(commit_m)), commit_none=float(np.mean(commit_n)),
            mean_len=float(np.mean(lens)), n=len(succ),
        )
        per_cat_rollouts[cat] = maps_rollouts
    overall_n = sum(rows[c]["n"] for c in CATEGORIES)
    overall = {
        k: float(sum(rows[c][k] * rows[c]["n"] for c in CATEGORIES) / overall_n)
        for k in ("success", "wrong_door", "timeout", "commit_build", "commit_mine", "commit_none", "mean_len")
    }
    return rows, overall, per_cat_rollouts


def print_table(rows, overall):
    hdr = f"{'category':10s} {'success':>8s} {'wrong_door':>10s} {'timeout':>8s} {'build':>7s} {'mine':>7s} {'none':>7s} {'len':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for c in CATEGORIES:
        r = rows[c]
        print(f"{c:10s} {r['success']:8.1%} {r['wrong_door']:10.1%} {r['timeout']:8.1%} "
              f"{r['commit_build']:7.1%} {r['commit_mine']:7.1%} {r['commit_none']:7.1%} {r['mean_len']:6.0f}")
    print("-" * len(hdr))
    print(f"{'overall':10s} {overall['success']:8.1%} {overall['wrong_door']:10.1%} {overall['timeout']:8.1%} "
          f"{overall['commit_build']:7.1%} {overall['commit_mine']:7.1%} {overall['commit_none']:7.1%} {overall['mean_len']:6.0f}")


def _draw_path(ax, pos, cm, color_by_outcome):
    pos = np.asarray(pos, dtype=float)
    cm = np.asarray(cm)
    if len(pos) < 2:
        return
    jit = (np.random.rand(*pos.shape) - 0.5) * 0.6
    xy = np.stack([(pos + jit)[:, 1], (pos + jit)[:, 0]], axis=1)
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    colors = [_COMMIT_COLORS.get(int(c), "gray") for c in cm[1:]]
    lc = LineCollection(segs, colors=colors, linewidths=0.6, alpha=0.06)
    ax.add_collection(lc)


def plot_grid(per_cat_rollouts, n_seeds, title, out_path):
    fig, axes = plt.subplots(len(CATEGORIES), n_seeds,
                             figsize=(n_seeds * 3.0, len(CATEGORIES) * 2.0))
    axes = np.asarray(axes).reshape(len(CATEGORIES), n_seeds)
    for ci, cat in enumerate(CATEGORIES):
        for sj in range(n_seeds):
            rec, out = per_cat_rollouts[cat][sj]
            ax = axes[ci, sj]
            ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
            for i, tr in enumerate(out["trajs"]):
                _draw_path(ax, tr, out["commits"][i], None)
            top_ok = rec.correct_target in ("top", "either")
            bot_ok = rec.correct_target in ("bottom", "either")
            for cells, ok in ((rec.top_goal_cells, top_ok), (rec.bottom_goal_cells, bot_ok)):
                if cells:
                    ys = [r for r, c in cells]; xs = [c for r, c in cells]
                    ax.scatter(xs, ys, c=("lime" if ok else "red"), s=26, marker="s", edgecolors="k", zorder=5)
            ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=22, marker="o", edgecolors="k", zorder=5)
            ax.set_xticks([]); ax.set_yticks([])
            succ = out["success"].mean(); wrong = (out["reached_any"] & ~out["success"]).mean()
            ax.set_title(f"{cat} s{sj}  succ {succ:.0%} wrong {wrong:.0%}", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, default=Path("outputs/bridge_tunnel_forkwall/ppo"))
    p.add_argument("--eval-maps", type=int, default=16, help="held-out maps/category")
    p.add_argument("--eval-traj", type=int, default=32, help="stochastic rollouts/map")
    p.add_argument("--grid-seeds", type=int, default=4)
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--passage-half", type=int, default=1)
    p.add_argument("--wall-margin", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    policy, cargs, view_size, env_size, env_width = _load_policy(args.checkpoint, device)
    tag = args.checkpoint.parent.name

    rows, overall, per_cat_rollouts = evaluate(
        policy, view_size, env_size, env_width, cargs, device,
        args.eval_maps, args.eval_traj, args.eval_seed_start, args.max_steps,
        args.passage_half, args.wall_margin)
    print_table(rows, overall)

    plot_grid(per_cat_rollouts, args.grid_seeds,
              f"PPO+GRU bridge_tunnel fork_wall  ·  {tag}  ·  {args.eval_traj} rollouts/map  ·  "
              f"green=correct door / red=decoy  ·  path color: blue=none/yellow=build/purple=mine",
              Path(str(args.out_prefix) + "_traj.png"))

    import json
    out_json = Path(str(args.out_prefix) + "_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"per_category": rows, "overall": overall, "checkpoint": str(args.checkpoint)}, f, indent=2)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
