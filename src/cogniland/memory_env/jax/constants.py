"""Tile / action / facing constants for the pure-JAX MemoryEnv.

The JAX env emits a SYMBOLIC egocentric tile-id minimap (not RGB). The tile
vocabulary encodes everything the agent needs: walls, the oriented+coloured cue
(4 distinct tiles -> shape AND colour are directly observable), and the two
coloured doors. The cue cell is non-overlappable (blocks movement) like the
MiniGrid OrientedKey; the doors are overlappable (stepping on one ends the
episode), like the MiniGrid _ColoredDoor (Goal-like).
"""
from __future__ import annotations

# ── tiles ────────────────────────────────────────────────────────────────
EMPTY = 0
WALL = 1
CUE_GREEN_UP = 2
CUE_BLUE_UP = 3
CUE_GREEN_DOWN = 4
CUE_BLUE_DOWN = 5
DOOR_GREEN = 6
DOOR_BLUE = 7
OOB = 8            # out-of-bounds padding for the egocentric crop
NUM_TILES = 9

# cue_type index -> cue tile id (CUE_TYPES order: green_up, blue_up, green_down, blue_down)
CUE_TILE = (CUE_GREEN_UP, CUE_BLUE_UP, CUE_GREEN_DOWN, CUE_BLUE_DOWN)

# ── factored tile attributes (index by tile id 0..NUM_TILES-1) ─────────────
# A SHARED colour / shape decomposition of each tile so the encoder can give the
# cue and the doors a COMMON colour feature (cue-green and door-green map to the
# same learned "green"), and expose the cue orientation as a separate "shape"
# feature. This makes colour→door trivial matching and gives clean, separable
# shape/colour latents for steering.
COLOR_NONE, COLOR_GREEN, COLOR_BLUE = 0, 1, 2
SHAPE_NONE, SHAPE_UP, SHAPE_DOWN = 0, 1, 2
N_COLOR, N_SHAPE = 3, 3
#              EMPTY WALL cGU cBU cGD cBD  dG dB OOB
TILE_COLOR = (   0,   0,   1,  2,  1,  2,   1, 2,  0)
TILE_SHAPE = (   0,   0,   1,  1,  2,  2,   0, 0,  0)

# ── actions (subset of MiniGrid's Discrete(7); 3..6 are no-ops in MemoryEnv) ─
A_LEFT = 0         # turn counter-clockwise
A_RIGHT = 1        # turn clockwise
A_FORWARD = 2      # step one cell in the facing direction
NUM_ACTIONS = 3

# ── facing (MiniGrid convention) ─────────────────────────────────────────
#   0 = EAST (+x), 1 = SOUTH (+y), 2 = WEST (-x), 3 = NORTH (-y)
DIR_EAST = 0
DIR_SOUTH = 1
DIR_WEST = 2
DIR_NORTH = 3
# dir -> (dx, dy) with +y pointing DOWN (row index increases downward)
DIR_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))

# ── cue catalogue (mirrors cogniland.memory_env.env) ─────────────────────
CUE_TYPES = ("green_up", "blue_up", "green_down", "blue_down")
# shape: up for idx {0,1}, down for {2,3};  colour: green for {0,2}, blue for {1,3}
CUE_IS_DOWN = (0, 0, 1, 1)      # 1 => correct branch is the LOWER row
CUE_IS_BLUE = (0, 1, 0, 1)      # 1 => target door colour is blue

# branch / selected-door sentinels for EnvState int fields
BRANCH_NONE = 0
BRANCH_UP = 1
BRANCH_DOWN = 2
DOOR_NONE = 0
SEL_GREEN = 1
SEL_BLUE = 2

__all__ = [
    "EMPTY", "WALL", "CUE_GREEN_UP", "CUE_BLUE_UP", "CUE_GREEN_DOWN",
    "CUE_BLUE_DOWN", "DOOR_GREEN", "DOOR_BLUE", "OOB", "NUM_TILES", "CUE_TILE",
    "A_LEFT", "A_RIGHT", "A_FORWARD", "NUM_ACTIONS",
    "DIR_EAST", "DIR_SOUTH", "DIR_WEST", "DIR_NORTH", "DIR_VEC",
    "CUE_TYPES", "CUE_IS_DOWN", "CUE_IS_BLUE",
    "TILE_COLOR", "TILE_SHAPE", "N_COLOR", "N_SHAPE",
    "COLOR_NONE", "COLOR_GREEN", "COLOR_BLUE", "SHAPE_NONE", "SHAPE_UP", "SHAPE_DOWN",
    "BRANCH_NONE", "BRANCH_UP", "BRANCH_DOWN", "DOOR_NONE", "SEL_GREEN", "SEL_BLUE",
]
