"""Tile vocabulary for bridge_tunnel_commit.

Identical 9-tile vocabulary to ``cogniland.bridge_tunnel.tiles`` (single source
of truth — re-exported here so the commit variant can never drift from the base
env's tile ids / palette / walkability).
"""
from __future__ import annotations

from cogniland.bridge_tunnel.tiles import (  # noqa: F401
    DIRT, GRASS, INVIOLABLE, NUM_TILES, OOB, ROCK, SAND, TARGET, TILE_COLORS,
    TILE_NAMES, TREE, WATER, WOOD, is_walkable,
)

__all__ = [
    "GRASS", "WATER", "ROCK", "WOOD", "TARGET", "OOB", "TREE", "SAND", "DIRT",
    "NUM_TILES", "TILE_NAMES", "TILE_COLORS", "INVIOLABLE", "is_walkable",
]
