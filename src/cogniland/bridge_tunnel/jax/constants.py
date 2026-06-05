"""Constants for the pure-JAX bridge_tunnel env (both variants).

Mirrors ``cogniland.bridge_tunnel.tiles`` / ``env`` 1:1 so the JAX env is
behaviourally identical to the PyTorch ``BridgeTunnelEnv`` on the same map +
action sequence (see the parity tests). The variant (bt/btc) is carried as a
static ``commit`` flag on ``EnvParams``, not here.
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

# ── actions (Discrete(6)) ─────────────────────────────────────────────
A_UP = 0
A_DOWN = 1
A_LEFT = 2
A_RIGHT = 3
A_BUILD = 4      # water → wood  (bt: always; btc: first success commits to build)
A_MINE = 5       # rock  → grass (bt: always; btc: first success commits to mine)
NUM_ACTIONS = 6

# ── commitment slot states (btc) ──────────────────────────────────────
COMMIT_NONE = 0
COMMIT_BUILD = 1
COMMIT_MINE = 2

# ── facing ids ─────────────────────────────────────────────────────────
F_UP = 0
F_DOWN = 1
F_LEFT = 2
F_RIGHT = 3

DEFAULT_VIEW_SIZE = 21


__all__ = [
    "GRASS", "WATER", "ROCK", "WOOD", "TARGET", "OOB", "TREE", "SAND", "DIRT",
    "NUM_TILES", "A_UP", "A_DOWN", "A_LEFT", "A_RIGHT", "A_BUILD", "A_MINE",
    "NUM_ACTIONS", "COMMIT_NONE", "COMMIT_BUILD", "COMMIT_MINE",
    "F_UP", "F_DOWN", "F_LEFT", "F_RIGHT", "DEFAULT_VIEW_SIZE",
]
