"""Constants for the pure-JAX bridge_tunnel_commit env.

A JAX port of ``src/cogniland/bridge_tunnel_commit/`` (PyTorch). Tile vocabulary
mirrors ``bridge_tunnel/tiles.py``; the action space adds the two COMMIT actions
and the env carries an irreversible commitment slot. Behaviourally identical to
``BridgeTunnelCommitEnv`` on the same map + action sequence
(see ``tests/test_bridge_tunnel_commit_jax_parity.py``).
"""
from __future__ import annotations

# ── tile ids (mirror bridge_tunnel/tiles.py) ──────────────────────────────
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

# ── action ids (Discrete(8)) ──────────────────────────────────────────
A_UP = 0
A_DOWN = 1
A_LEFT = 2
A_RIGHT = 3
A_BUILD = 4          # water → wood (only after committing to build)
A_MINE = 5           # rock  → grass (only after committing to mine)
A_COMMIT_BUILD = 6   # unlock build (once)
A_COMMIT_MINE = 7    # unlock mine  (once)
NUM_ACTIONS = 8

# ── commitment slot states ────────────────────────────────────────────
COMMIT_NONE = 0
COMMIT_BUILD = 1
COMMIT_MINE = 2

# ── facing ids (same order as the move actions) ───────────────────────
F_UP = 0
F_DOWN = 1
F_LEFT = 2
F_RIGHT = 3

DEFAULT_VIEW_SIZE = 21
N_SCALARS = 7        # facing one-hot (4) + step/max + commit_build + commit_mine


__all__ = [
    "GRASS", "WATER", "ROCK", "WOOD", "TARGET", "OOB", "TREE", "SAND", "DIRT",
    "NUM_TILES",
    "A_UP", "A_DOWN", "A_LEFT", "A_RIGHT", "A_BUILD", "A_MINE",
    "A_COMMIT_BUILD", "A_COMMIT_MINE", "NUM_ACTIONS",
    "COMMIT_NONE", "COMMIT_BUILD", "COMMIT_MINE",
    "F_UP", "F_DOWN", "F_LEFT", "F_RIGHT", "DEFAULT_VIEW_SIZE", "N_SCALARS",
]
