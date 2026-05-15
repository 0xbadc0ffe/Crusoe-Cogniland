"""Tile id constants and helpers for the Cogniland navigation env."""

from __future__ import annotations

import numpy as np

GRASS = 0
DIRT = 1
SAND = 2
WATER = 3
ROCK = 4
TARGET = 5
OOB = 6
TREE = 7   # always impassable; sits on grass
LAVA = 8   # always impassable; sits in stone region

NUM_TILES = 9

TILE_NAMES = {
    GRASS: "grass",
    DIRT: "dirt",
    SAND: "sand",
    WATER: "water",
    ROCK: "rock",
    TARGET: "target",
    OOB: "oob",
    TREE: "tree",
    LAVA: "lava",
}

LAND_TILES = (GRASS, DIRT, SAND, TARGET)


def is_land(tile_id: int) -> bool:
    return tile_id in LAND_TILES


def is_water(tile_id: int) -> bool:
    return tile_id == WATER


def is_rock(tile_id: int) -> bool:
    return tile_id == ROCK


TILE_COLORS = np.array(
    [
        (110, 173, 86),   # grass
        (158, 122, 80),   # dirt
        (224, 198, 130),  # sand
        (61, 113, 184),   # water
        (110, 110, 110),  # rock
        (250, 220, 60),   # target
        (0, 0, 0),        # oob
        (50, 110, 50),    # tree (dark green)
        (210, 60, 30),    # lava (red-orange)
    ],
    dtype=np.uint8,
)
