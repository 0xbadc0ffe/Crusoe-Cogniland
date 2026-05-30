#!/usr/bin/env python3
"""200-trajectory stochastic-policy grids for a trained zebra_nav agent.

For each of ``--n-maps`` map seeds (a subplot), samples ``--n-traj`` rollouts
from the **stochastic** policy on that fixed map and overlays them with low
alpha, so the spread of paths is visible: a policy that always bridges (or
always mines) bunches all paths to one side of every obsidian wall regardless
of which side is thinner, whereas a cue-following policy splits per stripe
toward the thin side.

Reached rollouts are drawn cyan, failed ones red. Each subplot is annotated
with the success rate and the thin-side accuracy aggregated over all
``n_traj`` rollouts × stripes.

    python scripts/zebra_traj_grid.py \\
        --checkpoint checkpoints/zebra_sweep/<run>/final.pt \\
        --n-maps 6 --n-traj 200 --out mapgen_preview/zebra_traj_<run>.png
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
from cogniland.zebra_nav import generate_zebra_map, tiles as T  # noqa: E402
from cogniland.zebra_nav.env import ZebraNavEnv  # noqa: E402
from train_ppo_zebra import PPOGRUPolicy  # noqa: E402


# facing-id → (dr, dc); matches env F_UP/F_DOWN/F_LEFT/F_RIGHT = 0/1/2/3
_FACE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


@torch.no_grad()
def batched_rollout(policy, rec, n_traj, view_size, max_steps, device):
    """Roll ``n_traj`` stochastic rollouts on one fixed map in lockstep. Returns
    (trajectories, reached[bool], thin_correct, thin_total, mine_pts, bridge_pts)
    where mine_pts / bridge_pts are the cells where a MINE (rock→grass) or PLACE
    (water→wood) succeeded, aggregated over all rollouts."""
    H, W = rec.terrain.shape
    envs = [ZebraNavEnv(map_record=rec, size=H, width=W, view_size=view_size,
                        max_steps=max_steps) for _ in range(n_traj)]
    obs = [e.reset()[0] for e in envs]
    h = torch.zeros(1, n_traj, policy.gru_hidden, device=device)
    done = torch.zeros(n_traj, device=device)
    active = np.ones(n_traj, dtype=bool)
    reached = np.zeros(n_traj, dtype=bool)
    trajs = [[tuple(e._pos)] for e in envs]
    mine_pts, bridge_pts = [], []

    for _ in range(max_steps):
        mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
        sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
        gru_out, h = policy._gru_forward({"minimap": mm, "scalars": sc},
                                         done[None], h)
        logits, _ = policy._heads(gru_out.squeeze(0))
        acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            o, r, term, trunc, info = e.step(int(acts[i]))
            obs[i] = o
            trajs[i].append(tuple(e._pos))
            if info["mined"] or info["placed"]:        # cell in front that was altered
                dr, dc = _FACE_DELTA[info["facing"]]
                cell = (e._pos[0] + dr, e._pos[1] + dc)
                (mine_pts if info["mined"] else bridge_pts).append(cell)
            if term:
                reached[i] = True; active[i] = False
            elif trunc:
                active[i] = False
        done = torch.zeros(n_traj, device=device)   # no resets; one episode each
        if not active.any():
            break

    thin_c = thin_t = 0
    for e in envs:
        c, t = e._thin_side_accuracy()
        thin_c += c; thin_t += t
    return trajs, reached, thin_c, thin_t, mine_pts, bridge_pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--n-maps", type=int, default=6)
    p.add_argument("--n-traj", type=int, default=200)
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cargs = ckpt["args"]
    env_size = cargs.get("env_size", 32)
    env_width = cargs.get("env_width") or env_size
    view_size = cargs.get("view_size", 11)
    orientation = cargs.get("orientation", "diagonal")
    device = torch.device(args.device)

    dummy = ZebraNavEnv(size=env_size, width=env_width, view_size=view_size)
    dummy.reset()
    n_tiles = int(ckpt["policy"]["tile_embed.weight"].shape[0])   # match training-time NUM_TILES
    n_act = int(ckpt["policy"]["actor.weight"].shape[0])
    policy = PPOGRUPolicy(dummy.observation_space, num_actions=n_act,
                          gru_hidden=cargs.get("gru_hidden", 128),
                          embed_dim=cargs.get("embed_dim", 256),
                          num_tile_classes=n_tiles).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    # fewer columns for wide (rectangular) maps so each panel stays legible
    aspect = env_width / env_size
    ncol = 2 if aspect >= 1.5 else 3
    nrow = int(np.ceil(args.n_maps / ncol))
    pw = 3.4 * max(1.0, aspect * 0.6)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * pw, nrow * 3.0))
    axes = np.atleast_1d(axes).flatten()

    all_succ, all_tc, all_tt = [], 0, 0
    for j in range(args.n_maps):
        seed = args.eval_seed_start + j
        rec = generate_zebra_map(size=env_size, width=env_width, seed=seed,
                                 n_stripes=cargs.get("n_stripes", 4),
                                 thick_half=cargs.get("thick_half", 3),
                                 thin_half=cargs.get("thin_half", 1),
                                 obsidian_half=cargs.get("obsidian_half", 1),
                                 orientation=orientation)
        trajs, reached, tc, tt, mine_pts, bridge_pts = batched_rollout(
            policy, rec, args.n_traj, view_size, args.max_steps, device)
        succ = float(reached.mean())
        all_succ.append(succ); all_tc += tc; all_tt += tt

        ax = axes[j]
        ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
        # dark-blue trajectories; failed rollouts a touch more visible
        for i, tr in enumerate(trajs):
            a = np.array(tr)
            jit = (np.random.rand(*a.shape) - 0.5) * 0.6   # tiny jitter to spread overlap
            ax.plot(a[:, 1] + jit[:, 1], a[:, 0] + jit[:, 0],
                    color="darkblue", lw=0.7, alpha=0.04 if reached[i] else 0.08)
        # overlay decisions: YELLOW = mine (rock tunnel), RED = bridge (over water)
        if mine_pts:
            m = np.array(mine_pts)
            ax.scatter(m[:, 1], m[:, 0], color="yellow", s=6, alpha=0.18, zorder=3, linewidths=0)
        if bridge_pts:
            b = np.array(bridge_pts)
            ax.scatter(b[:, 1], b[:, 0], color="red", s=6, alpha=0.18, zorder=3, linewidths=0)
        ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=28, marker="s",
                   edgecolors="k", zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"seed {seed}  succ {succ:.0%}", fontsize=8)
    for j in range(args.n_maps, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{args.checkpoint.parent.name}  ·  {args.n_traj} stochastic rollouts/map  ·  "
                 f"success {np.mean(all_succ):.0%}  ·  "
                 f"path=darkblue  mine=yellow  bridge=red", fontsize=11)
    fig.tight_layout()
    out = args.out or Path(f"mapgen_preview/zebra_traj_{args.checkpoint.parent.name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    thin_all = all_tc / max(1, all_tt)
    print(f"success={np.mean(all_succ):.2%}"
          + (f"  thin_side={thin_all:.2%} ({all_tc}/{all_tt})" if all_tt else ""))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
