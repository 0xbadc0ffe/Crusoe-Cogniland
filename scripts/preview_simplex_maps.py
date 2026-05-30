#!/usr/bin/env python3
"""Preview grid for the legacy ``simplex`` (Crafter-style noise) generator.

Rows = biome (lake / rocky / balanced), cols = seeds. Each cell is one full
64x64 map rendered with the env tile palette, spawn (green dot) + target
(white star) marked. Use to eyeball the old training maps before switching
the trainer back to ``generator="simplex"``.

    python scripts/preview_simplex_maps.py [--generator simplex] [--size 64]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cogniland.nav.mapgen import generate_map  # noqa: E402
from cogniland.nav.tiles import TILE_COLORS  # noqa: E402

BIOMES = ["lake", "rocky", "balanced"]
SEEDS = [7, 13, 21, 42, 77]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="simplex",
                    choices=("simplex", "components", "composed"))
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nrow, ncol = len(BIOMES), len(SEEDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow))
    axes = np.atleast_2d(axes)
    for i, biome in enumerate(BIOMES):
        for j, seed in enumerate(SEEDS):
            ax = axes[i, j]
            rec = generate_map(size=args.size, map_type=biome, seed=seed,
                               generator=args.generator, max_retries=400)
            ax.imshow(TILE_COLORS[rec.terrain], interpolation="nearest")
            sr, sc = rec.spawn
            tr, tc = rec.target
            ax.scatter([sc], [sr], marker="o", s=40, facecolor="#39ff14",
                       edgecolor="black", lw=0.7, zorder=6)
            ax.scatter([tc], [tr], marker="*", s=110, facecolor="white",
                       edgecolor="black", lw=0.7, zorder=6)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"seed {seed}", fontsize=10)
            if j == 0:
                ax.set_ylabel(biome, fontsize=12)
    fig.suptitle(f"generator = {args.generator!r}  ·  {args.size}x{args.size}",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out) if args.out else ROOT / "mapgen_preview" / f"{args.generator}_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
