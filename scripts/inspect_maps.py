#!/usr/bin/env python3
"""Generate a handful of maps and print/render them for visual inspection.

Useful while iterating on `mapgen.py`. Pass ``--show`` to pop up a Matplotlib
window with each generated map.

Examples
--------
    python scripts/inspect_maps.py --size 64 --num 6
    python scripts/inspect_maps.py --size 96 --num 4 --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogniland.nav import generate_map  # noqa: E402
from cogniland.nav.renderer import render_color_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64, choices=(32, 64, 96, 128))
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--map-type", default="random", choices=("random", "lake", "rocky"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="display via matplotlib")
    args = parser.parse_args()

    records = []
    for i in range(args.num):
        rec = generate_map(size=args.size, map_type=args.map_type, seed=args.seed + i)
        records.append(rec)
        margin = 100 * (rec.no_skill_cost - min(rec.raft_cost, rec.harness_cost)) / max(rec.no_skill_cost, 1e-6)
        print(
            f"[{i:02d}] type={rec.map_type:5s} correct={'raft' if rec.correct_object == 1 else 'harness'}  "
            f"spawn={tuple(int(x) for x in rec.spawn)} → target={tuple(int(x) for x in rec.target)}  "
            f"costs no/raft/harn=({rec.no_skill_cost:6.2f},{rec.raft_cost:6.2f},{rec.harness_cost:6.2f})  "
            f"shortcut={margin:4.1f}%"
        )

    if args.show:
        import matplotlib.pyplot as plt

        cols = min(args.num, 4)
        rows = (args.num + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, rec in zip(axes, records):
            img = render_color_grid(rec.terrain, cell_px=4)
            ax.imshow(img)
            ax.scatter([rec.spawn[1] * 4 + 2], [rec.spawn[0] * 4 + 2], c="lime", s=40, marker="s")
            ax.scatter([rec.target[1] * 4 + 2], [rec.target[0] * 4 + 2], c="gold", s=60, marker="*")
            ax.set_title(
                f"{rec.map_type}  no={rec.no_skill_cost:.1f}  r={rec.raft_cost:.1f}  h={rec.harness_cost:.1f}"
            )
            ax.axis("off")
        for ax in axes[len(records) :]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
