"""Constants for the pure-JAX zebra_nav env.

This is a JAX port of ``src/cogniland/zebra_nav/`` (PyTorch). The tile
vocabulary, action ids, facing ids, and reward shape mirror that env 1:1
so the two are behaviourally identical on the same map + action sequence.
See ``tests/test_zebra_jax_parity.py`` for the equivalence proof.

Natural-only vocabulary: obsidian + cue tiles are retired; TREE is the sole
inviolable tile. Must stay in lock-step with ``zebra_nav/tiles.py``.
"""
from __future__ import annotations

# ── tile ids (mirror src/cogniland/zebra_nav/tiles.py) ────────────────
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

# ── action ids (Discrete(6)) ──────────────────────────────────────────
A_UP = 0
A_DOWN = 1
A_LEFT = 2
A_RIGHT = 3
A_PLACE = 4     # water → wood (bridge) in the facing cell
A_MINE = 5      # rock  → grass (mine)  in the facing cell
NUM_ACTIONS = 6

# ── facing ids (same order as the move actions) ───────────────────────
F_UP = 0
F_DOWN = 1
F_LEFT = 2
F_RIGHT = 3

DEFAULT_VIEW_SIZE = 21


__all__ = [
    "GRASS", "WATER", "ROCK", "WOOD", "TARGET", "OOB", "TREE", "SAND", "DIRT",
    "NUM_TILES",
    "A_UP", "A_DOWN", "A_LEFT", "A_RIGHT", "A_PLACE", "A_MINE", "NUM_ACTIONS",
    "F_UP", "F_DOWN", "F_LEFT", "F_RIGHT", "DEFAULT_VIEW_SIZE",
]
