#!/usr/bin/env python3
"""Curate a fixed set of bridge_tunnel validation maps and pickle them.

The same pickled set is used by the demo (``play_bridge_tunnel.py --maps``) and by
evaluation, so "what you validate on" == "what you play". Stores a list of
``MapRecord`` dataclasses plus the generation metadata, and writes a preview PNG.

    python scripts/make_bridge_tunnel_val_maps.py                      # default: natural, 16 maps
    python scripts/make_bridge_tunnel_val_maps.py --n 12 --goal-half 4
"""
from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cogniland.bridge_tunnel import generate_bridge_tunnel_map, tiles as T  # noqa: E402
from cogniland.bridge_tunnel.mapgen import is_reachable  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orientation", default="natural", choices=("natural",),
                   help="only natural is supported (stripe orientations retired)")
    p.add_argument("--n", type=int, default=16, help="number of validation maps")
    p.add_argument("--seed-start", type=int, default=10_000,
                   help="held-out seeds (kept distinct from training's random seeds)")
    p.add_argument("--env-size", type=int, default=32)
    p.add_argument("--env-width", type=int, default=64)
    # natural knobs — keep in sync with the released agent's training config
    p.add_argument("--water-frac", type=float, default=0.14)
    p.add_argument("--rock-frac", type=float, default=0.14)
    p.add_argument("--tree-frac", type=float, default=0.03)
    p.add_argument("--goal-half", type=int, default=1, help="natural: central goal door half-height (default 1 = 3-cell door; <0 = whole right wall)")
    p.add_argument("--out", type=Path, default=Path("data/bridge_tunnel/val_maps.pkl"))
    args = p.parse_args()

    kw = dict(size=args.env_size, width=args.env_width, orientation=args.orientation)
    if args.orientation == "natural":
        kw.update(water_frac=args.water_frac, rock_frac=args.rock_frac,
                  tree_frac=args.tree_frac,
                  goal_half=(args.goal_half if args.goal_half >= 0 else None))

    recs, seeds = [], list(range(args.seed_start, args.seed_start + args.n))
    for s in seeds:
        rec = generate_bridge_tunnel_map(seed=s, **kw)
        assert is_reachable(rec), f"seed {s} not reachable"
        recs.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"orientation": args.orientation, "seeds": seeds,
                     "kwargs": kw, "records": recs}, f)
    print(f"wrote {len(recs)} {args.orientation} validation maps → {args.out}")

    # preview
    ncol = 2 if args.env_width >= 1.5 * args.env_size else 4
    nrow = int(np.ceil(args.n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 2.4))
    axes = np.atleast_1d(axes).flatten()
    for j, rec in enumerate(recs):
        ax = axes[j]
        ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
        ax.scatter([rec.spawn[1]], [rec.spawn[0]], c="white", s=24, marker="o", edgecolors="k")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"val map {j} (seed {recs[j].seed})", fontsize=8)
    for j in range(len(recs), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"bridge_tunnel validation maps — {args.orientation} {args.env_size}x{args.env_width}", fontsize=11)
    fig.tight_layout()
    prev = args.out.with_name(args.out.stem + "_preview.png")
    fig.savefig(prev, dpi=100)
    print(f"wrote preview → {prev}")


if __name__ == "__main__":
    main()
