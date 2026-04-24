"""Static params (`EnvParams`) + per-episode mutable state (`EnvState`).

Both are flax.struct.dataclass pytrees so they flow through jax.jit, vmap,
and tree_map untouched. `EnvParams` carries every hyperparameter **and** the
map dataset tensors (uploaded once). `EnvState` carries everything that can
change during an episode.

All JAX-array fields use ``struct.field(default_factory=...)`` because
dataclasses forbids mutable defaults — a jnp array is mutable to dataclass
even though it behaves as immutable semantically.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from cogniland_jax import constants as C


@struct.dataclass
class EnvParams:
    # ── Game constants ────────────────────────────────────────────────
    max_steps: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.int32(1000),
    )
    reach_bonus: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.float32(50.0),
    )
    step_penalty: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.float32(0.02),
    )
    shaping_coef: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.float32(0.3),
    )
    hp_coef: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.float32(0.12),
    )
    death_penalty: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.float32(0.0),
    )
    correct_answer_bonus: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.float32(100.0),
    )
    craft_bonus: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.float32(100.0),
    )
    difficulty: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.int32(C.DIFFICULTY_HARD),
    )

    # ── Map dataset (static, uploaded once) ───────────────────────────
    terrain_idx: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros((1, C.MAP_SIZE, C.MAP_SIZE), dtype=jnp.int8),
    )
    berry_mask: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros((1, C.MAP_SIZE, C.MAP_SIZE), dtype=jnp.bool_),
    )
    vis_lut_packed: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros(
            (1, C.MAP_SIZE, C.MAP_SIZE, 254), dtype=jnp.uint8,
        ),
    )
    biome_id: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros((1,), dtype=jnp.int32),
    )

    @staticmethod
    def from_map_arrays(
        terrain_idx: jnp.ndarray,
        berry_mask: jnp.ndarray,
        vis_lut_packed: jnp.ndarray,
        biome_id: jnp.ndarray,
        **overrides,
    ) -> "EnvParams":
        return EnvParams(
            terrain_idx=terrain_idx.astype(jnp.int8),
            berry_mask=berry_mask.astype(jnp.bool_),
            vis_lut_packed=vis_lut_packed.astype(jnp.uint8),
            biome_id=biome_id.astype(jnp.int32),
            **overrides,
        )

    @property
    def num_maps(self) -> int:
        return self.terrain_idx.shape[0]


@struct.dataclass
class EnvState:
    pos_r: jnp.ndarray
    pos_c: jnp.ndarray
    hp: jnp.ndarray
    wood: jnp.ndarray
    tool: jnp.ndarray
    consec_grass: jnp.ndarray
    steps: jnp.ndarray
    map_idx: jnp.ndarray
    spawn_r: jnp.ndarray
    spawn_c: jnp.ndarray
    yes_r: jnp.ndarray
    yes_c: jnp.ndarray
    no_r: jnp.ndarray
    no_c: jnp.ndarray
    mid_r: jnp.ndarray
    mid_c: jnp.ndarray
    ctg: jnp.ndarray
    ctg_spawn: jnp.ndarray
    task_id: jnp.ndarray
    terminated: jnp.ndarray
    last_action: jnp.ndarray
    crafted_this_step: jnp.ndarray
