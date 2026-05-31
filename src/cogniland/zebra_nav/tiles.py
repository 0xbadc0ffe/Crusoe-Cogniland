"""Tile id constants + render palette for zebra_nav (natural-only vocabulary).

Tile semantics
--------------
* ``GRASS``    — walkable empty land.
* ``WATER``    — impassable; if the agent faces a water cell and PLACEs, it
  turns into ``WOOD`` (a walkable bridge).
* ``ROCK``     — impassable; if the agent faces a rock cell and MINEs, it
  turns into ``GRASS``.
* ``WOOD``     — walkable bridge tile that PLACE created over water.
* ``TARGET``   — goal tile (the centre door / goal wall for natural maps).
* ``OOB``      — out-of-map sentinel used only in egocentric observations
  (pads cells outside the world).
* ``TREE``     — impassable AND inviolable: a tree patch the agent can neither
  mine nor bridge, so it must be walked around. The only impassable terrain.
* ``SAND`` / ``DIRT`` — purely cosmetic: behave **exactly like grass** (walkable,
  unit step cost), but rendered differently. SAND fringes water (beaches), DIRT
  fringes rock — just for visual variety / legibility.

Note: the obsidian wall + cue tiles of the retired stripe orientations are gone
(they caused phantom lava/diamond artifacts in the sprite renderers). TREE is
now the sole inviolable tile.
"""
from __future__ import annotations

import numpy as np

GRASS = 0
WATER = 1
ROCK = 2
WOOD = 3
TARGET = 4
OOB = 5
TREE = 6
SAND = 7
DIRT = 8

NUM_TILES = 9

TILE_NAMES = {
    GRASS: "grass",
    WATER: "water",
    ROCK: "rock",
    WOOD: "wood",
    TARGET: "target",
    OOB: "oob",
    TREE: "tree",
    SAND: "sand",
    DIRT: "dirt",
}

# RGB palette for visualisation (one row per tile id, 0..8)
TILE_COLORS = np.array(
    [
        (110, 173, 86),     # grass
        (61, 113, 184),     # water
        (110, 110, 110),    # rock
        (140, 90,  50),     # wood — brown
        (250, 220, 60),     # target — yellow
        (0, 0, 0),          # oob
        (24, 70, 32),       # tree — dark forest green
        (224, 205, 140),    # sand — pale tan (beach around water)
        (134, 104, 74),     # dirt — brown (around rock)
    ],
    dtype=np.uint8,
)

# ──────────────────────────────────────────────────────────────────────────

# SAND / DIRT are walkable look-alikes of grass.
_WALKABLE = (GRASS, WOOD, TARGET, SAND, DIRT)
# impassable AND inviolable (cannot be mined or bridged) — TREE only.
INVIOLABLE = (TREE,)


def is_walkable(tile: int) -> bool:
    return tile in _WALKABLE


__all__ = [
    "GRASS", "WATER", "ROCK", "WOOD", "TARGET", "OOB", "TREE", "SAND", "DIRT",
    "NUM_TILES", "TILE_NAMES", "TILE_COLORS", "INVIOLABLE",
    "is_walkable",
]
