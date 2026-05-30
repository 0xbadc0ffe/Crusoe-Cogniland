#!/usr/bin/env python3
"""Generate the static (policy-free) figures for the crafter_in_cogniland doc.

Outputs into ``paper/figures/``:

* ``tiles/<name>.png``   — one composited 64px sprite icon per tile id.
* ``grid_size_biome.png`` — rows = map size (32/64/96/128),
                            cols = biome (lake/rocky/balanced).
* ``grid_seeds_64.png``   — 4x4 grid of size-64 maps over different seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cogniland.nav.mapgen import generate_map  # noqa: E402
from cogniland.nav.tiles import (  # noqa: E402
    GRASS, DIRT, SAND, WATER, ROCK, TARGET, OOB, TREE, LAVA, TILE_COLORS,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
SPRITES = ROOT / "src" / "cogniland" / "assets" / "sprites"

# ---------------------------------------------------------------- tile icons
# (base sprite, optional overlay sprite). target/tree sit on a grass base.
_ICON = {
    "grass":  ("grass.png", None),
    "dirt":   ("path.png", None),
    "sand":   ("sand.png", None),
    "water":  ("water.png", None),
    "rock":   ("stone.png", None),
    "target": ("grass.png", "flag.png"),
    "tree":   ("grass.png", "tree.png"),
    "lava":   ("lava.png", None),
}


def _load(name: str, px: int) -> Image.Image:
    return Image.open(SPRITES / name).convert("RGBA").resize((px, px), Image.NEAREST)


def export_tiles(px: int = 96) -> None:
    out = FIGDIR / "tiles"
    out.mkdir(parents=True, exist_ok=True)
    for name, (base, overlay) in _ICON.items():
        img = _load(base, px)
        if overlay is not None:
            ov = _load(overlay, px)
            img = Image.alpha_composite(img, ov)
        img.convert("RGB").save(out / f"{name}.png")
    # OOB = solid black square (off-map padding)
    Image.new("RGB", (px, px), (0, 0, 0)).save(out / "oob.png")
    print(f"wrote tile icons -> {out}")


# ----------------------------------------------------------------- map grids
def _render(rec) -> np.ndarray:
    return TILE_COLORS[rec.terrain]


def _draw_markers(ax, rec) -> None:
    sr, sc = rec.spawn
    tr, tc = rec.target
    ax.scatter([sc], [sr], marker="o", s=22, facecolor="lime",
               edgecolor="black", linewidth=0.6, zorder=4)
    ax.scatter([tc], [tr], marker="*", s=55, facecolor="gold",
               edgecolor="black", linewidth=0.6, zorder=4)


def grid_size_biome(sizes=(32, 64, 96, 128),
                    biomes=("balanced", "rocky", "lake"),
                    seed: int = 7) -> None:
    nr, nc = len(sizes), len(biomes)
    fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.6 * nr))
    for i, sz in enumerate(sizes):
        for j, bio in enumerate(biomes):
            ax = axes[i, j]
            rec = generate_map(size=sz, map_type=bio, seed=seed)
            ax.imshow(_render(rec), interpolation="nearest")
            _draw_markers(ax, rec)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(bio, fontsize=13)
            if j == 0:
                ax.set_ylabel(f"{sz}x{sz}", fontsize=13)
    fig.tight_layout()
    p = FIGDIR / "grid_size_biome.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")


def grid_seeds_64(seeds=(101, 102, 103, 104, 105), size=64,
                  biomes=("balanced", "rocky", "lake")) -> None:
    """Rows = biome (balanced/rocky/lake), cols = map seed, all at 64x64."""
    nr, nc = len(biomes), len(seeds)
    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.3 * nr))
    for i, bio in enumerate(biomes):
        for j, sd in enumerate(seeds):
            ax = axes[i, j]
            rec = generate_map(size=size, map_type=bio, seed=sd)
            ax.imshow(_render(rec), interpolation="nearest")
            _draw_markers(ax, rec)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"seed {sd}", fontsize=11)
            if j == 0:
                ax.set_ylabel(bio, fontsize=13)
    fig.tight_layout()
    p = FIGDIR / "grid_seeds_64.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    export_tiles()
    grid_size_biome()
    grid_seeds_64()
