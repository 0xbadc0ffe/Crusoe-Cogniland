#!/usr/bin/env python3
"""Evaluate a trained zebra_nav PPO agent: trajectory grids + thin-side accuracy.

For each of ``--n-maps`` held-out maps it rolls the (frozen) policy out, then:

* renders the trajectory over the map (cyan path, magenta = a build/mine step),
* decides, per stripe, **which side the agent crossed** (water if it crossed
  the ``t = C`` line at ``s < S_mid``, rock if ``s > S_mid``) and compares it
  to the stripe's thinner side — the *thin-side accuracy* is the fraction of
  stripe crossings that took the cheaper side.

Reports aggregate success rate, mean episode length, and thin-side accuracy.

    python scripts/eval_zebra_agent.py \\
        --checkpoint checkpoints/zebra_sweep/<run>/final.pt \\
        --n-maps 12 --out mapgen_preview/zebra_eval.png
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
from cogniland.zebra_nav.env import ZebraNavEnv, A_PLACE, A_MINE  # noqa: E402
from train_ppo_zebra import PPOGRUPolicy  # noqa: E402


@torch.no_grad()
def rollout(policy, rec, device, max_steps, deterministic=True, view_size=11):
    env = ZebraNavEnv(map_record=rec, size=rec.terrain.shape[0],
                      width=rec.terrain.shape[1], max_steps=max_steps, view_size=view_size)
    obs, _ = env.reset()
    h = torch.zeros(1, 1, policy.gru_hidden, device=device)
    done = torch.zeros(1, device=device)
    traj = [tuple(env._pos)]
    build_steps = []
    reached = False
    for _ in range(max_steps):
        ot = {k: torch.from_numpy(np.asarray(v)[None]).to(device) for k, v in obs.items()}
        logits_h = policy._gru_forward({k: v.unsqueeze(0) for k, v in ot.items()},
                                       done.unsqueeze(0), h)
        gru_out, h = logits_h
        logits, _ = policy._heads(gru_out.squeeze(0))
        a = int(torch.argmax(logits, -1)) if deterministic else \
            int(torch.distributions.Categorical(logits=logits).sample())
        obs, r, term, trunc, info = env.step(a)
        traj.append(tuple(env._pos))
        if a in (A_PLACE, A_MINE) and (info["placed"] or info["mined"]):
            build_steps.append(tuple(env._pos))
        done = torch.zeros(1, device=device)
        if term:
            reached = True
            break
        if trunc:
            break
    # orientation-aware thin-side accuracy via the env's own logic
    thin_ok, thin_tot = env._thin_side_accuracy()
    return traj, build_steps, reached, len(traj) - 1, thin_ok, thin_tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--n-maps", type=int, default=12)
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of argmax")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("mapgen_preview/zebra_eval.png"))
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cargs = ckpt["args"]
    env_size = cargs.get("env_size", 32)
    env_width = cargs.get("env_width") or env_size
    orientation = cargs.get("orientation", "natural")
    device = torch.device(args.device)

    # build a dummy env to recover obs space, then the policy
    dummy = ZebraNavEnv(size=env_size, width=env_width, view_size=cargs.get("view_size", 11))
    dummy.reset()
    n_tiles = int(ckpt["policy"]["tile_embed.weight"].shape[0])   # match training-time NUM_TILES
    policy = PPOGRUPolicy(dummy.observation_space, num_actions=6,
                          gru_hidden=cargs.get("gru_hidden", 128),
                          embed_dim=cargs.get("embed_dim", 256),
                          num_tile_classes=n_tiles).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    ncol = 4
    nrow = int(np.ceil(args.n_maps / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 3.4))
    axes = np.atleast_1d(axes).flatten()

    successes, lengths, thin_correct, thin_total = [], [], 0, 0
    for j in range(args.n_maps):
        seed = args.eval_seed_start + j
        rec = generate_zebra_map(
            size=env_size, width=env_width, seed=seed, orientation=orientation,
            water_frac=cargs.get("water_frac", 0.14),
            rock_frac=cargs.get("rock_frac", 0.14),
            tree_frac=cargs.get("tree_frac", 0.03),
            goal_half=cargs.get("goal_half"))
        traj, builds, reached, steps, n_ok, n_tot = rollout(
            policy, rec, device, args.max_steps, deterministic=not args.stochastic,
            view_size=cargs.get("view_size", 11))
        successes.append(float(reached)); lengths.append(steps)
        thin_correct += n_ok; thin_total += n_tot

        ax = axes[j]
        ax.imshow(T.TILE_COLORS[rec.terrain])
        tr = np.array(traj)
        ax.plot(tr[:, 1], tr[:, 0], color="cyan", lw=1.4, alpha=0.9)
        if builds:
            br = np.array(builds)
            ax.scatter(br[:, 1], br[:, 0], color="magenta", s=14, zorder=3)
        ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=30, marker="s")  # spawn
        ax.set_xticks([]); ax.set_yticks([])
        tag = "✓" if reached else "✗"
        ax.set_title(f"seed {seed}  {tag}  {steps} steps  thin {n_ok}/{n_tot}",
                     fontsize=9)
    for j in range(args.n_maps, len(axes)):
        axes[j].axis("off")

    succ = float(np.mean(successes))
    mlen = float(np.mean(lengths))
    thin_acc = thin_correct / max(1, thin_total)
    fig.suptitle(f"{args.checkpoint.parent.name}  ·  success {succ:.0%}  ·  "
                 f"mean {mlen:.0f} steps  ·  thin-side {thin_acc:.0%} "
                 f"({thin_correct}/{thin_total})", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"success={succ:.2%}  mean_steps={mlen:.1f}  "
          f"thin_side_acc={thin_acc:.2%} ({thin_correct}/{thin_total})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
