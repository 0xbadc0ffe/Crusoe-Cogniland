"""Tile id constants + render palette for zebra_nav.

Tile semantics
--------------
* ``GRASS``    — walkable empty land.
* ``WATER``    — impassable; if the agent faces a water cell and PLACEs, it
  turns into ``WOOD`` (a walkable bridge).
* ``ROCK``     — impassable; if the agent faces a rock cell and MINEs, it
  turns into ``GRASS``.
* ``OBSIDIAN`` — impassable AND inviolable: cannot be mined or bridged. Used
  as the wall between the water and rock segments of a stripe.
* ``WOOD``     — walkable bridge tile that PLACE created over water.
* ``CUE_WATER_THIN``/``CUE_ROCK_THIN`` — informational tiles sitting on grass
  before each stripe. They reveal which of the two segments at the upcoming
  stripe is the thinner (shorter-to-cross) one. Walkable.
* ``TARGET``   — goal tile (a corner cell for stripe maps, the whole goal wall
  for natural maps).
* ``OOB``      — out-of-map sentinel used only in egocentric observations
  (pads cells outside the world).
* ``TREE``     — impassable AND inviolable (like obsidian, but a natural-map
  obstacle): a tree patch the agent can neither mine nor bridge, so it must be
  walked around.
* ``SAND`` / ``DIRT`` — purely cosmetic: behave **exactly like grass** (walkable,
  unit step cost), but rendered differently. SAND fringes water (beaches), DIRT
  fringes rock — just for visual variety / legibility.
"""
from __future__ import annotations

import numpy as np

GRASS = 0
WATER = 1
ROCK = 2
OBSIDIAN = 3
WOOD = 4
CUE_WATER_THIN = 5
CUE_ROCK_THIN = 6
TARGET = 7
OOB = 8
TREE = 9
SAND = 10
DIRT = 11

NUM_TILES = 12

TILE_NAMES = {
    GRASS: "grass",
    WATER: "water",
    ROCK: "rock",
    OBSIDIAN: "obsidian",
    WOOD: "wood",
    CUE_WATER_THIN: "cue_water_thin",
    CUE_ROCK_THIN: "cue_rock_thin",
    TARGET: "target",
    OOB: "oob",
    TREE: "tree",
    SAND: "sand",
    DIRT: "dirt",
}

# RGB palette for visualisation
TILE_COLORS = np.array(
    [
        (110, 173, 86),     # grass
        (61, 113, 184),     # water
        (110, 110, 110),    # rock
        (35,  20,  55),     # obsidian — near-black with a purple tinge
        (140, 90,  50),     # wood — brown
        (180, 200, 230),    # cue_water_thin — pale blue (mnemonic: water easier)
        (200, 200, 210),    # cue_rock_thin  — pale gray  (rock easier)
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
_WALKABLE = (GRASS, WOOD, TARGET, CUE_WATER_THIN, CUE_ROCK_THIN, SAND, DIRT)
_CUES = (CUE_WATER_THIN, CUE_ROCK_THIN)
# impassable AND inviolable (cannot be mined or bridged)
INVIOLABLE = (OBSIDIAN, TREE)


def is_walkable(tile: int) -> bool:
    return tile in _WALKABLE


def is_cue(tile: int) -> bool:
    return tile in _CUES


__all__ = [
    "GRASS", "WATER", "ROCK", "OBSIDIAN", "WOOD",
    "CUE_WATER_THIN", "CUE_ROCK_THIN", "TARGET", "OOB", "TREE", "SAND", "DIRT",
    "NUM_TILES", "TILE_NAMES", "TILE_COLORS", "INVIOLABLE",
    "is_walkable", "is_cue",
]
