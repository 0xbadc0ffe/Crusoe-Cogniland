"""Egocentric tile-id minimap renderer for the pure-JAX bridge_tunnel env.

A bit-for-bit port of ``BridgeTunnelEnv._make_obs`` / ``_egocentric_crop``:

* ``minimap`` : (V, V) int8 egocentric crop of the *current* (mutable) terrain
  centred on the agent. Cells outside the world are filled with ``OOB`` (=8).
  Unlike crafter_in_cogniland, the agent's own cell is NOT overwritten — the
  raw underlying tile id is shown.
* ``scalars`` : (5,) float32 = [facing one-hot (4), step / max_steps].
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import constants as C
from .state import EnvParams, EnvState


def egocentric_minimap(state: EnvState, params: EnvParams) -> jax.Array:
    """Return the agent's egocentric tile-id crop, shape (V, V) int8."""
    V = params.view_size
    half = V // 2
    terrain = state.terrain                        # (H, W) int8 — mutable
    H = params.height
    W = params.width
    rr = state.agent_r - half + jnp.arange(V, dtype=jnp.int32)[:, None]   # (V, 1)
    cc = state.agent_c - half + jnp.arange(V, dtype=jnp.int32)[None, :]   # (1, V)
    in_bounds = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    safe_rr = jnp.clip(rr, 0, H - 1)
    safe_cc = jnp.clip(cc, 0, W - 1)
    crop = terrain[safe_rr, safe_cc]
    crop = jnp.where(in_bounds, crop, jnp.int8(C.OOB))
    return crop.astype(jnp.int8)                   # (V, V) int8


def scalars_obs(state: EnvState, params: EnvParams) -> jax.Array:
    """(5,) float32 — [facing one-hot (4), step / max_steps]."""
    face_oh = jax.nn.one_hot(state.facing, 4, dtype=jnp.float32)
    step_norm = state.step_count.astype(jnp.float32) / float(max(1, params.max_steps))
    return jnp.concatenate([face_oh, step_norm[None]])


def build_obs(state: EnvState, params: EnvParams) -> dict:
    """Dict observation matching the bridge_tunnel convention.

    Keys:
        minimap : int8  (V, V)
        scalars : float (5,)
    """
    return {
        "minimap": egocentric_minimap(state, params),
        "scalars": scalars_obs(state, params),
    }
