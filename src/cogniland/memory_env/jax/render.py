"""Symbolic observation for the JAX MemoryEnv.

obs = {"minimap": (V,V) int8 egocentric axis-aligned tile crop (OOB-padded),
       "scalars": (5,) float32 = [facing one-hot (4), step_count/max_steps]}.

The cue tile encodes BOTH shape (up/down) and colour (4 distinct cue tiles), and
the two doors are coloured tiles, so everything the policy needs is in the tile
ids. The small egocentric crop (default V=5) yields the memory structure: the
cue leaves view a few cells past the room and the doors only appear near the end.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import constants as C
from .state import EnvParams, EnvState

_CUE_TILE = jnp.asarray(C.CUE_TILE, dtype=jnp.int8)


def _full_grid(state: EnvState, params: EnvParams) -> jax.Array:
    """Base terrain with the cue + the two coloured doors overlaid."""
    g = params.base_terrain
    g = g.at[state.cue_y, state.cue_x].set(_CUE_TILE[state.cue_type])
    top = jnp.where(state.door_green_top, C.DOOR_GREEN, C.DOOR_BLUE).astype(jnp.int8)
    bot = jnp.where(state.door_green_top, C.DOOR_BLUE, C.DOOR_GREEN).astype(jnp.int8)
    g = g.at[params.row_door_top, params.x_doorcol].set(top)
    g = g.at[params.row_door_bot, params.x_doorcol].set(bot)
    return g


def egocentric_minimap(state: EnvState, params: EnvParams) -> jax.Array:
    V = params.view_size
    half = V // 2
    H, W = params.height, params.width
    g = _full_grid(state, params)
    rr = state.agent_y - half + jnp.arange(V)
    cc = state.agent_x - half + jnp.arange(V)
    in_b = ((rr >= 0) & (rr < H))[:, None] & ((cc >= 0) & (cc < W))[None, :]
    crop = g[jnp.clip(rr, 0, H - 1)][:, jnp.clip(cc, 0, W - 1)]
    return jnp.where(in_b, crop, jnp.int8(C.OOB)).astype(jnp.int8)


def scalars_obs(state: EnvState, params: EnvParams) -> jax.Array:
    facing = jax.nn.one_hot(state.agent_dir, 4, dtype=jnp.float32)
    frac = (state.step_count.astype(jnp.float32) / jnp.float32(params.max_steps))
    return jnp.concatenate([facing, frac[None]], axis=-1)


def build_obs(state: EnvState, params: EnvParams) -> dict:
    return {"minimap": egocentric_minimap(state, params),
            "scalars": scalars_obs(state, params)}
