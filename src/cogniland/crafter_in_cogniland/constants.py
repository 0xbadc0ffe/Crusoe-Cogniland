"""Constants for the crafter_in_cogniland JAX env.

This env is a pure-JAX port of `src/cogniland/nav/`. Same physics, same
rewards, same tile vocabulary — just expressed as a Gymnax-style
``Environment`` so the whole train loop can be `jax.jit`-compiled
end-to-end.
"""
from __future__ import annotations

# ── tile ids (mirror src/cogniland/nav/tiles.py) ──────────────────────
GRASS = 0
DIRT = 1
SAND = 2
WATER = 3
ROCK = 4
TARGET = 5
OOB = 6
TREE = 7
LAVA = 8

NUM_TERRAIN_TILES = 9

# Egocentric crop in tiles. ``minimap`` shape is (DIAMETER, DIAMETER).
# View_size in the PyTorch env (default 21) is exactly the diameter.
DEFAULT_VIEW_SIZE = 21

# ── action ids (discrete-only; both envs share the same two terminal
# build actions — the PyTorch env now matches this scheme) ──────────────
ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3
ACTION_BUILD_RAFT    = 4
ACTION_BUILD_HARNESS = 5
NUM_ACTIONS = 6

# ── active-object ids (mirror src/cogniland/nav/skills.py) ────────────
OBJ_NONE    = 0
OBJ_RAFT    = 1
OBJ_HARNESS = 2
NUM_OBJECTS = 3

# ── reward constants (mirror skills.py) ───────────────────────────────
# PBRS shaping provides the gradient toward target; the +1 sparse terminal
# reward marks success on reaching it.
SLACK_PENALTY = -0.02
SHAPING_COEF = 0.01
REACH_BONUS = 1.0

# ── slip mechanic (mirror skills.py) ──────────────────────────────────
SLIP_PROB_DEFAULT = 0.75    # water/rock without the matching item; trees always
SLIP_PROB_MINOR = 0.30      # bare-handed sand/dirt apron tax.
SLIP_PROB_LAND_WITH_SKILL = 0.50
                            # Land weight tax (2026-05-28, lowered from 0.75):
                            # grass/sand/dirt all slip 50 %% whenever ANY
                            # skill is committed.
SLIP_PROB_GRASS = SLIP_PROB_LAND_WITH_SKILL  # deprecated alias (pre-2026-05-28)
SLIP_WEIGHT_LAND = SLIP_PROB_MINOR   # deprecated alias (old "weight tax" name)


__all__ = [
    "GRASS", "DIRT", "SAND", "WATER", "ROCK", "TARGET", "OOB", "TREE", "LAVA",
    "NUM_TERRAIN_TILES", "DEFAULT_VIEW_SIZE",
    "ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT",
    "ACTION_BUILD_RAFT", "ACTION_BUILD_HARNESS", "NUM_ACTIONS",
    "OBJ_NONE", "OBJ_RAFT", "OBJ_HARNESS", "NUM_OBJECTS",
    "SLACK_PENALTY", "SHAPING_COEF", "REACH_BONUS",
    "SLIP_PROB_DEFAULT", "SLIP_PROB_MINOR", "SLIP_PROB_LAND_WITH_SKILL",
    "SLIP_PROB_GRASS", "SLIP_WEIGHT_LAND",
]
