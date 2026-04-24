"""Engine constants — single source of truth for the JAX env.

Terrain ordering, tile-class enum, action IDs, visibility radii, and the
drain lookup table. Every other module imports from here; never introduce
magic numbers in `dynamics.py` or `env.py`.
"""

from __future__ import annotations

import jax.numpy as jnp

# ── Terrain ────────────────────────────────────────────────────────────────
TERRAIN_NAMES: tuple[str, ...] = (
    "ocean", "deep_water", "water",
    "beach", "sandy", "grassland",
    "forest", "rocky", "mountains",
)
NUM_TERRAINS = len(TERRAIN_NAMES)
WATER_MAX_IDX = 2  # terrain_idx <= 2 is water

# ── Tile-class enum (0..13, 14 total) ──────────────────────────────────────
# The minimap is one channel of int8s taking values in this enum. Terrain
# classes 1..9 = TERRAIN_NAMES[0..8] + 1; overrides on top.
TILE_UNSEEN = 0
TILE_BERRY = 10
TILE_TARGET_YES = 11
TILE_TARGET_NO = 12
TILE_DEADLY = 13
NUM_TILE_CLASSES = 14

# ── Actions ────────────────────────────────────────────────────────────────
# 0..3 cardinal moves, 4 forage, 5..7 craft raft/rope/shoes.
NUM_ACTIONS = 8
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3
ACTION_FORAGE = 4
ACTION_CRAFT_RAFT, ACTION_CRAFT_ROPE, ACTION_CRAFT_SHOES = 5, 6, 7

# Row/col deltas per action id. Non-movement actions carry (0,0).
MOVE_DELTAS = jnp.array(
    [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
    dtype=jnp.int32,
)

# Tool id per action (0 = no craft). raft=1, rope=2, shoes=3.
ACTION_TO_TOOL = jnp.array([0, 0, 0, 0, 0, 1, 2, 3], dtype=jnp.int32)

TOOL_NONE, TOOL_RAFT, TOOL_ROPE, TOOL_SHOES = 0, 1, 2, 3

# ── Map / minimap geometry ─────────────────────────────────────────────────
MAP_SIZE = 128
MINIMAP_RADIUS = 22
MINIMAP_DIAMETER = 2 * MINIMAP_RADIUS + 1  # 45
TARGET_GAP = 3  # YES and NO targets sit TARGET_GAP cols apart on the same row
SPAWN_TARGET_CLEAR_HALF = 3  # 7×7 water-free box (half-width 3 = 7 cells)

# ── Tile effects (drain, heal, craft) ──────────────────────────────────────
# Base HP drain per terrain (index into TERRAIN_NAMES).
HP_DRAIN = jnp.array(
    [8.0, 5.0, 3.0, 1.0, 1.0, 1.0, 2.0, 6.0, 8.0],
    dtype=jnp.float32,
)

# Drain overrides when a specific tool is equipped. One row per tool
# id (0=none, 1=raft, 2=rope, 3=shoes); values of -1 mean "fall through
# to HP_DRAIN". ``shoes_drain_grassland`` only applies once
# ``consec_grass >= SHOES_K``; until then grassland uses the base drain.
_DRAIN_BY_TOOL = jnp.stack([
    HP_DRAIN,                                             # none
    HP_DRAIN.at[0].set(4.0).at[1].set(2.0).at[2].set(1.0),  # raft -> ocean/dw/water
    HP_DRAIN.at[7].set(1.0).at[8].set(2.0),                # rope -> rocky/mountains
    HP_DRAIN,                                             # shoes handled separately
], axis=0).astype(jnp.float32)   # [4 tools, 9 terrains]
DRAIN_BY_TOOL = _DRAIN_BY_TOOL

SHOES_DRAIN_GRASSLAND = jnp.float32(0.5)
SHOES_K = jnp.int32(10)

GRASS_IDX = jnp.int32(TERRAIN_NAMES.index("grassland"))
FOREST_IDX = jnp.int32(TERRAIN_NAMES.index("forest"))

# ── Heal / gather ─────────────────────────────────────────────────────────
BERRY_HEAL = jnp.float32(100.0)   # full heal, non-consumable berry tiles
FOREST_WOOD = jnp.int32(10)
HP_MAX = jnp.float32(100.0)
WOOD_MAX = jnp.int32(100)
INIT_HP = jnp.float32(100.0)
CRAFT_COST = jnp.int32(100)

# ── Visibility (per-terrain disk radius) ──────────────────────────────────
# Index by terrain id; result is clipped to MINIMAP_RADIUS.
VIS_PER_TERRAIN = jnp.array(
    [22, 18, 14, 12, 12, 12, 10, 18, 22], dtype=jnp.int32,
)

# ── Reward defaults ────────────────────────────────────────────────────────
DEFAULT_REACH_BONUS = jnp.float32(150.0)
DEFAULT_STEP_PENALTY = jnp.float32(0.02)
DEFAULT_SHAPING_COEF = jnp.float32(0.3)
DEFAULT_HP_COEF = jnp.float32(0.06)
DEFAULT_DEATH_PENALTY = jnp.float32(0.0)
DEFAULT_MAX_STEPS = jnp.int32(1000)

# ── Difficulty bands (max Euclidean spawn↔target distance) ────────────────
# 0 = easy, 1 = medium, 2 = hard. No minimum distance is ever enforced —
# the agent may spawn right next to the target in any difficulty.
DIFFICULTY_EASY = 0
DIFFICULTY_MEDIUM = 1
DIFFICULTY_HARD = 2
MAX_EUCLID_BY_DIFFICULTY = jnp.array([20.0, 50.0, jnp.inf], dtype=jnp.float32)

# Spawn/target search caps — stay on one map per episode. First draw a
# water-free target; try up to ``SPAWN_TRIES_PER_TARGET`` spawn
# positions uniformly on the map (within the Euclidean cap). If none
# valid, swap the target (up to ``TARGET_TRIES_PER_MAP`` times). On
# total exhaustion we fall back to any grassland tile on *this* map
# — every real biome has at least one grassland cell, so no
# map-resample path.
#
# Budget notes: on balanced maps only ~17 % of cells pass the 7×7
# water-free check, so target validity per draw is ~5 %. 50 target
# tries gives ≈ 1 − 0.95⁵⁰ ≈ 92 % pure-target success; combined with
# the grass fallback we effectively never produce a water-adjacent
# spawn in practice.
SPAWN_TRIES_PER_TARGET = 50
TARGET_TRIES_PER_MAP = 50

# ── Tasks ─────────────────────────────────────────────────────────────────
# Seven task slots; the one-hot ``task_embedding`` obs is always this size.
#
# Task 0 — REACH:        reach either the YES or NO target. Gets the base
#                        ``reach_bonus`` on touch. Success = reached.
# Task 1/2/3 — BIOME CLASSIFY: "reach YES iff biome==X, else reach NO".
#     1 = archipelago, 2 = grassland, 3 = highland.
#     Bonus: +correct_answer_bonus on the correctly-chosen target.
#     Success = correct classification.
# Task 4/5/6 — CRAFT:    craft a specific tool (raft/rope/shoes) during
#     the episode. Bonus +craft_bonus on the step the tool is crafted.
#     Success = tool was crafted at least once this episode.
TASK_EMBEDDING_DIM = 7

TASK_REACH = 0
TASK_CLS_ARCHIPELAGO = 1
TASK_CLS_GRASSLAND = 2
TASK_CLS_HIGHLAND = 3
TASK_CRAFT_RAFT = 4
TASK_CRAFT_ROPE = 5
TASK_CRAFT_SHOES = 6

# Biome id table (indexing ``maps.BIOME_NAMES``: balanced=0, archipelago=1,
# grassland=2, highland=3). -1 means "task doesn't care about biome".
TASK_BIOME_FOR_CLS = jnp.array([-1, 1, 2, 3, -1, -1, -1], dtype=jnp.int32)

# Tool id table (0=none, 1=raft, 2=rope, 3=shoes). 0 means "no craft
# requirement" (non-craft tasks).
TASK_TOOL_FOR_CRAFT = jnp.array([0, 0, 0, 0, 1, 2, 3], dtype=jnp.int32)

# Task-bonus defaults (legacy values).
DEFAULT_CORRECT_ANSWER_BONUS = jnp.float32(100.0)
DEFAULT_CRAFT_BONUS = jnp.float32(100.0)
