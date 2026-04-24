"""JAX tile-class minimap renderer.

Produces the int8 ``[45, 45]`` minimap the agent observes. All logic lives
inside a jittable function; no numpy, no side effects.

Priority on collision (matches the numpy env):
    TARGET_YES > TARGET_NO > BERRY > DEADLY > terrain > UNSEEN.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cogniland_jax import constants as C
from cogniland_jax.state import EnvParams, EnvState


def render_minimap(state: EnvState, params: EnvParams) -> jnp.ndarray:
    """Return the int8 45×45 minimap centred on the agent.

    Visibility: terrain-dependent disk radius, intersected with the
    offline-raycast visibility LUT (resolves height-based occlusion).
    """
    terrain_idx_map = params.terrain_idx[state.map_idx]   # [H, W] int8
    berry_mask_map = params.berry_mask[state.map_idx]      # [H, W] bool
    vis_packed = params.vis_lut_packed[
        state.map_idx, state.pos_r, state.pos_c
    ]                                                        # [P] uint8

    D = C.MINIMAP_DIAMETER
    R = C.MINIMAP_RADIUS
    full = jnp.unpackbits(vis_packed, bitorder="little")[: D * D]
    full = full.reshape(D, D).astype(jnp.bool_)

    # Per-terrain disk mask.
    t_here = terrain_idx_map[state.pos_r, state.pos_c]
    t_here = jnp.clip(t_here, 0, C.NUM_TERRAINS - 1).astype(jnp.int32)
    vis_r = C.VIS_PER_TERRAIN[t_here]
    yy, xx = jnp.meshgrid(
        jnp.arange(-R, R + 1), jnp.arange(-R, R + 1), indexing="ij",
    )
    disk = (yy * yy + xx * xx) <= (vis_r * vis_r)
    vis_mask = full & disk

    # Gather the 45×45 patch.
    di = jnp.arange(-R, R + 1, dtype=jnp.int32)
    rows = state.pos_r + di[:, None]
    cols = state.pos_c + di[None, :]
    in_bounds = (rows >= 0) & (rows < C.MAP_SIZE) & (cols >= 0) & (cols < C.MAP_SIZE)
    rows_c = jnp.clip(rows, 0, C.MAP_SIZE - 1)
    cols_c = jnp.clip(cols, 0, C.MAP_SIZE - 1)

    t_raw = terrain_idx_map[rows_c, cols_c]                      # [D, D] int8
    b_raw = berry_mask_map[rows_c, cols_c]                        # [D, D] bool
    valid = vis_mask & in_bounds

    # Base: terrain + 1 (so class 1..9); unseen → 0; deadly → TILE_DEADLY.
    base = (t_raw.astype(jnp.int16) + 1)
    base = jnp.where(valid, base, jnp.int16(0))
    base = jnp.where(valid & (t_raw == -1), jnp.int16(C.TILE_DEADLY), base)
    # Berry overlay.
    is_berry_cell = valid & b_raw & (t_raw != -1)
    base = jnp.where(is_berry_cell, jnp.int16(C.TILE_BERRY), base)

    # Target overlays (NO first, YES overrides).
    for tr, tc, tile_val in (
        (state.no_r, state.no_c, C.TILE_TARGET_NO),
        (state.yes_r, state.yes_c, C.TILE_TARGET_YES),
    ):
        ty = tr - state.pos_r + R
        tx = tc - state.pos_c + R
        in_patch = (ty >= 0) & (ty < D) & (tx >= 0) & (tx < D)
        ty_c = jnp.clip(ty, 0, D - 1)
        tx_c = jnp.clip(tx, 0, D - 1)
        visible = in_patch & vis_mask[ty_c, tx_c]
        base = jax.lax.cond(
            visible,
            lambda b: b.at[ty_c, tx_c].set(jnp.int16(tile_val)),
            lambda b: b,
            base,
        )

    return base.astype(jnp.int8)


def build_scalars(state: EnvState, params: EnvParams) -> jnp.ndarray:
    """6-dim scalar vector: [compass_x, compass_y, tile_class/10, hp, wood, tool].

    ``tile_class`` raw values: 0..8 for the nine terrain classes,
    **10 for berry** (kept distinct from the max terrain 8 so the
    normalised signal has a clear gap; mountains → 0.8, berry → 1.0).
    """
    terrain_idx_map = params.terrain_idx[state.map_idx]
    berry_mask_map = params.berry_mask[state.map_idx]
    dr = state.mid_r.astype(jnp.float32) - state.pos_r.astype(jnp.float32)
    dc = state.mid_c.astype(jnp.float32) - state.pos_c.astype(jnp.float32)
    dist = jnp.maximum(jnp.sqrt(dr * dr + dc * dc), 1e-6)
    cx = dc / dist
    cy = dr / dist

    t = jnp.clip(terrain_idx_map[state.pos_r, state.pos_c], 0, C.NUM_TERRAINS - 1).astype(jnp.float32)
    is_berry = berry_mask_map[state.pos_r, state.pos_c]
    tile_cls = jnp.where(is_berry, jnp.float32(10.0), t) / 10.0
    hp_n = state.hp / C.HP_MAX
    wood_n = state.wood.astype(jnp.float32) / C.WOOD_MAX.astype(jnp.float32)
    tool_n = state.tool.astype(jnp.float32) / 3.0

    return jnp.stack([cx, cy, tile_cls, hp_n, wood_n, tool_n], axis=-1)


def build_task_embedding(state: EnvState) -> jnp.ndarray:
    return jax.nn.one_hot(state.task_id, C.TASK_EMBEDDING_DIM, dtype=jnp.float32)


def build_obs(state: EnvState, params: EnvParams) -> dict[str, jnp.ndarray]:
    return {
        "minimap": render_minimap(state, params),
        "scalars": build_scalars(state, params),
        "task_embedding": build_task_embedding(state),
    }
