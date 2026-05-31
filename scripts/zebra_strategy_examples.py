#!/usr/bin/env python3
"""Render ONE clean example rollout per strategy the zebra_nav agent uses, with a
STRICT definition of "going through" an obstacle.

    avoid   — reaches the goal without traversing any obstacle body
    bridge  — path enters a WATER body and exits the OTHER side (>= min-cross cells)
    tunnel  — path enters a ROCK  body and exits the OTHER side (>= min-cross cells)

Stricter than counting build/mine events: placing/mining a *single* block, or
poking into a lake/mountain and backing out the same side, does NOT count as a
crossing. A crossing is a maximal run of consecutive path cells whose *original*
terrain is the obstacle tile, with >= ``--min-cross`` distinct cells AND distinct
entry/exit land cells (i.e. the agent went in one side and out the other). This
is exactly the segment-labelling rule we use for the activation-dataset strategy
labels.

The figure highlights the crossed lake/mountain body (translucent) and the in->out
path segment, so the bridging/mining is unambiguous.

    python scripts/zebra_strategy_examples.py \\
        --checkpoint models/zebra_nav/natural_centergoal3.pt \\
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
from scipy.ndimage import label as cc_label

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.zebra_nav import generate_zebra_map, tiles as T  # noqa: E402
from cogniland.zebra_nav.env import ZebraNavEnv  # noqa: E402
from train_ppo_zebra import PPOGRUPolicy  # noqa: E402


@torch.no_grad()
def rollout(policy, rec, view_size, max_steps, device, seed):
    """One stochastic rollout on a fixed map. Returns the path (list of cells),
    reached flag, and #build/#mine events."""
    torch.manual_seed(seed)
    env = ZebraNavEnv(map_record=rec, size=rec.terrain.shape[0],
                      width=rec.terrain.shape[1], view_size=view_size,
                      max_steps=max_steps)
    obs = env.reset()[0]
    h = torch.zeros(1, 1, policy.gru_hidden, device=device)
    done = torch.zeros(1, device=device)
    path = [tuple(int(x) for x in env._pos)]
    n_build = n_mine = 0
    reached = False
    for _ in range(max_steps):
        mm = torch.from_numpy(obs["minimap"])[None, None].to(device)
        sc = torch.from_numpy(obs["scalars"])[None, None].to(device)
        gru_out, h = policy._gru_forward({"minimap": mm, "scalars": sc}, done[None], h)
        logits, _ = policy._heads(gru_out.squeeze(0))
        a = int(torch.distributions.Categorical(logits=logits).sample())
        obs, r, term, trunc, info = env.step(a)
        path.append(tuple(int(x) for x in env._pos))
        n_build += int(bool(info["placed"]))
        n_mine += int(bool(info["mined"]))
        if term:
            reached = True
            break
        if trunc:
            break
    return dict(path=path, reached=reached, rec=rec, seed=seed,
                n_build=n_build, n_mine=n_mine)


def crossings(path, terrain, tile, min_cross):
    """Maximal runs of consecutive path cells whose ORIGINAL terrain == ``tile``,
    that constitute a genuine traversal: >= ``min_cross`` distinct obstacle cells
    AND bounded by two DISTINCT land cells (entered one side, exited the other —
    not a single-block clip, not an in-and-back-out retreat).

    Returns a list of dicts {cells:[...], entry:(r,c), exit:(r,c)}."""
    res, n, i = [], len(path), 0
    while i < n:
        if terrain[path[i]] == tile:
            j = i
            while j < n and terrain[path[j]] == tile:
                j += 1
            distinct = list(dict.fromkeys(path[i:j]))     # de-dup blocked-move repeats
            entry = path[i - 1] if i > 0 else None
            exit_ = path[j] if j < n else None
            if (len(distinct) >= min_cross and entry is not None
                    and exit_ is not None and entry != exit_):
                res.append({"cells": distinct, "entry": entry, "exit": exit_})
            i = j
        else:
            i += 1
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("models/zebra_nav/natural_centergoal3.pt"))
    p.add_argument("--n-seeds", type=int, default=12)
    p.add_argument("--n-per-seed", type=int, default=40)
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--min-cross", type=int, default=2,
                   help="min distinct obstacle cells crossed (in->out) to count as bridge/tunnel")
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

    _gh = cargs.get("goal_half")
    avoid, bridges, tunnels = [], [], []
    for j in range(args.n_seeds):
        seed = args.eval_seed_start + j
        rec = generate_zebra_map(size=env_size, width=env_width, seed=seed,
                                 orientation=orientation,
                                 water_frac=cargs.get("water_frac", 0.14),
                                 rock_frac=cargs.get("rock_frac", 0.14),
                                 tree_frac=cargs.get("tree_frac", 0.03),
                                 goal_half=(_gh if (_gh is not None and _gh >= 0) else None))
        terr = rec.terrain
        for k in range(args.n_per_seed):
            ro = rollout(policy, rec, view_size, args.max_steps, device, seed=seed * 1000 + k)
            if not ro["reached"]:
                continue
            wx = crossings(ro["path"], terr, T.WATER, args.min_cross)
            rx = crossings(ro["path"], terr, T.ROCK, args.min_cross)
            ro["water_cross"], ro["rock_cross"] = wx, rx
            if not wx and not rx:
                avoid.append(ro)
            elif wx and not rx:
                bridges.append(ro)
            elif rx and not wx:
                tunnels.append(ro)

    def _max_cells(ro, key):
        return max((len(c["cells"]) for c in ro[key]), default=0)

    # pick the clearest example of each (longest crossing for bridge/tunnel)
    ex_avoid = max(avoid, key=lambda r: len(r["path"])) if avoid else None
    ex_bridge = max(bridges, key=lambda r: _max_cells(r, "water_cross")) if bridges else None
    ex_tunnel = max(tunnels, key=lambda r: _max_cells(r, "rock_cross")) if tunnels else None

    panels = [("avoid — walk around", ex_avoid, None, None),
              ("bridge — into & out of a lake", ex_bridge, "water_cross", "red"),
              ("tunnel — into & out of a mountain", ex_tunnel, "rock_cross", "gold")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (title, ro, key, col) in zip(axes, panels):
        if ro is None:
            ax.set_title(f"{title}\n(none found)", fontsize=10); ax.axis("off"); continue
        rec = ro["rec"]; terr = rec.terrain; H, W = terr.shape
        ax.imshow(T.TILE_COLORS[terr], interpolation="nearest", zorder=0)
        a = np.array(ro["path"])

        ncells = 0
        if key is not None and ro[key]:
            cross = max(ro[key], key=lambda c: len(c["cells"]))
            ncells = len(cross["cells"])
            tile = T.WATER if key == "water_cross" else T.ROCK
            # highlight the WHOLE obstacle body the agent crossed (connected comp)
            lbl, _ = cc_label(terr == tile)
            cr0, cc0 = cross["cells"][0]
            body = (lbl == lbl[cr0, cc0])
            ov = np.zeros((H, W, 4))
            rgba = (1, 0, 0, 0.30) if col == "red" else (1, 0.84, 0, 0.34)
            ov[body] = rgba
            ax.imshow(ov, interpolation="nearest", zorder=1)
            # the in->out segment: entry land -> crossed cells -> exit land
            seg = np.array([cross["entry"], *cross["cells"], cross["exit"]])
            ax.plot(seg[:, 1], seg[:, 0], color=col, lw=4.0, alpha=0.95,
                    solid_capstyle="round", zorder=4)
            ax.scatter([cross["entry"][1]], [cross["entry"][0]], color="lime", s=80,
                       marker="o", edgecolors="k", lw=0.6, zorder=6)   # IN
            ax.scatter([cross["exit"][1]], [cross["exit"][0]], color="magenta", s=110,
                       marker="*", edgecolors="k", lw=0.6, zorder=6)    # OUT

        ax.plot(a[:, 1], a[:, 0], color="darkblue", lw=1.6, alpha=0.85, zorder=3)
        ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=90, marker="s",
                   edgecolors="k", zorder=7)
        tr = np.array(np.where(terr == T.TARGET))
        if tr.size:
            ax.scatter(tr[1], tr[0], facecolors="none", edgecolors="lime", s=40, lw=1.4, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        sub = (f"seed {ro['seed'] // 1000}  ·  len {len(a)}"
               + (f"  ·  crossed {ncells} {'water' if col=='red' else 'rock'} cells"
                  if key else "  ·  no obstacle traversed"))
        ax.set_title(f"{title}\n{sub}", fontsize=9.5)

    fig.suptitle(f"{args.checkpoint.stem} — one rollout per strategy  "
                 f"(strict: >= {args.min_cross} cells, in one side & out the other)\n"
                 f"blue=path · highlighted body=crossed lake/mountain · "
                 f"green ●=enter · magenta ★=exit · white ■=spawn · lime ○=goal",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"min_cross={args.min_cross}  found: avoid={len(avoid)} "
          f"bridge={len(bridges)} tunnel={len(tunnels)}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
