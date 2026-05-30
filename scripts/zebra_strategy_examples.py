#!/usr/bin/env python3
"""Render ONE clean example rollout for each strategy the zebra_nav agent uses:

    avoid  — reaches the goal with NO build/mine (walks around obstacles)
    bridge — places at least one WOOD over WATER
    tunnel — mines at least one ROCK into GRASS

Samples stochastic rollouts across several natural-map seeds, classifies each
trajectory by the build/mine events it triggered, then picks the most legible
example of each (longest detour for "avoid"; fewest builds for bridge/tunnel so
the single decision is obvious) and draws each as a single top-down panel:
path = blue line, bridge cells = red squares, mine cells = yellow squares,
spawn = white square, goal door = green ring.

    python scripts/zebra_strategy_examples.py \\
        --checkpoint models/zebra_nav/natural_agent.pt \\
        --out mapgen_preview/zebra_strategy_examples.png
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

_FACE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


@torch.no_grad()
def rollout(policy, rec, view_size, max_steps, device, seed):
    """One stochastic rollout on a fixed map. Returns dict with path, bridge/mine
    cells, reached flag."""
    torch.manual_seed(seed)
    env = ZebraNavEnv(map_record=rec, size=rec.terrain.shape[0],
                      width=rec.terrain.shape[1], view_size=view_size,
                      max_steps=max_steps)
    obs = env.reset()[0]
    h = torch.zeros(1, 1, policy.gru_hidden, device=device)
    done = torch.zeros(1, device=device)
    path = [tuple(env._pos)]
    bridge, mine = [], []
    reached = False
    for _ in range(max_steps):
        mm = torch.from_numpy(obs["minimap"])[None, None].to(device)
        sc = torch.from_numpy(obs["scalars"])[None, None].to(device)
        gru_out, h = policy._gru_forward({"minimap": mm, "scalars": sc}, done[None], h)
        logits, _ = policy._heads(gru_out.squeeze(0))
        a = int(torch.distributions.Categorical(logits=logits).sample())
        obs, r, term, trunc, info = env.step(a)
        path.append(tuple(env._pos))
        if info["mined"] or info["placed"]:
            dr, dc = _FACE_DELTA[info["facing"]]
            cell = (env._pos[0] + dr, env._pos[1] + dc)
            (mine if info["mined"] else bridge).append(cell)
        if term:
            reached = True
            break
        if trunc:
            break
    return dict(path=np.array(path), bridge=bridge, mine=mine,
                reached=reached, rec=rec, seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("models/zebra_nav/natural_agent.pt"))
    p.add_argument("--n-seeds", type=int, default=12)
    p.add_argument("--n-per-seed", type=int, default=40)
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("mapgen_preview/zebra_strategy_examples.png"))
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cargs = ckpt["args"]
    env_size = cargs.get("env_size", 32)
    env_width = cargs.get("env_width") or env_size
    view_size = cargs.get("view_size", 21)
    orientation = cargs.get("orientation", "natural")
    device = torch.device(args.device)

    dummy = ZebraNavEnv(size=env_size, width=env_width, view_size=view_size)
    dummy.reset()
    n_tiles = int(ckpt["policy"]["tile_embed.weight"].shape[0])
    n_act = int(ckpt["policy"]["actor.weight"].shape[0])
    policy = PPOGRUPolicy(dummy.observation_space, num_actions=n_act,
                          gru_hidden=cargs.get("gru_hidden", 128),
                          embed_dim=cargs.get("embed_dim", 256),
                          num_tile_classes=n_tiles).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    # collect classified rollouts across several maps
    avoid, bridges, tunnels = [], [], []
    for j in range(args.n_seeds):
        seed = args.eval_seed_start + j
        rec = generate_zebra_map(size=env_size, width=env_width, seed=seed,
                                 orientation=orientation,
                                 water_frac=cargs.get("water_frac", 0.14),
                                 rock_frac=cargs.get("rock_frac", 0.14),
                                 tree_frac=cargs.get("tree_frac", 0.03),
                                 goal_half=cargs.get("goal_half", 4))
        for k in range(args.n_per_seed):
            ro = rollout(policy, rec, view_size, args.max_steps, device,
                         seed=seed * 1000 + k)
            if not ro["reached"]:
                continue
            nb, nm = len(ro["bridge"]), len(ro["mine"])
            if nb == 0 and nm == 0:
                avoid.append(ro)
            elif nb > 0 and nm == 0:
                bridges.append(ro)
            elif nm > 0 and nb == 0:
                tunnels.append(ro)

    # pick the most legible example of each
    def pick(pool, key, default_msg):
        if not pool:
            print(f"WARNING: {default_msg}")
            return None
        return key(pool)

    ex_avoid = pick(avoid, lambda P: max(P, key=lambda r: len(r["path"])),
                    "no pure-avoid rollout found")            # longest detour
    ex_bridge = pick(bridges, lambda P: min(P, key=lambda r: len(r["bridge"])),
                     "no pure-bridge rollout found")          # fewest builds → clearest
    ex_tunnel = pick(tunnels, lambda P: min(P, key=lambda r: len(r["mine"])),
                     "no pure-tunnel rollout found")

    examples = [("avoid (walk around)", ex_avoid),
                ("bridge (place wood on water)", ex_bridge),
                ("tunnel (mine through rock)", ex_tunnel)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (title, ro) in zip(axes, examples):
        if ro is None:
            ax.set_title(f"{title}\n(none found)", fontsize=10)
            ax.axis("off")
            continue
        rec = ro["rec"]
        ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
        a = ro["path"]
        ax.plot(a[:, 1], a[:, 0], color="darkblue", lw=2.0, alpha=0.9, zorder=4)
        if ro["bridge"]:
            b = np.array(ro["bridge"])
            ax.scatter(b[:, 1], b[:, 0], color="red", s=70, marker="s",
                       edgecolors="k", lw=0.5, zorder=6, label="bridge")
        if ro["mine"]:
            m = np.array(ro["mine"])
            ax.scatter(m[:, 1], m[:, 0], color="yellow", s=70, marker="s",
                       edgecolors="k", lw=0.5, zorder=6, label="mine")
        ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=90, marker="s",
                   edgecolors="k", zorder=7)
        # goal door
        tr = np.array(np.where(rec.terrain == T.TARGET))
        if tr.size:
            ax.scatter(tr[1], tr[0], facecolors="none", edgecolors="lime",
                       s=40, lw=1.2, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\nseed {ro['rec'].__dict__.get('seed', '?')}  "
                     f"len {len(a)}  ·  {len(ro['bridge'])} bridge / {len(ro['mine'])} mine",
                     fontsize=9)
    fig.suptitle(f"{args.checkpoint.stem} — one stochastic rollout per strategy "
                 f"(blue=path, white=spawn, lime=goal, red=bridge, yellow=mine)",
                 fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"found: avoid={len(avoid)} bridge={len(bridges)} tunnel={len(tunnels)}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
