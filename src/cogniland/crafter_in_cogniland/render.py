"""Egocentric tile-id minimap renderer (pure JAX).

The agent always sees a ``(V, V)`` window centred on its current cell.
Cells outside the map's bounds are filled with ``OOB`` (=6) so the model
learns "this side is impassable" without seeing a special token. The
agent's own cell is *replaced* by the TARGET tile id when overlapping
with the target (so the agent perceives the goal underfoot), otherwise
the underlying tile id is left untouched.

Returns int8 to keep the buffer small (the encoder will one-hot the
class ids itself).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import constants as C
from .state import EnvParams, EnvState


def egocentric_minimap(state: EnvState, params: EnvParams) -> jax.Array:
    """Return the agent's egocentric tile-id crop, shape (V, V)."""
    V = params.view_size
    half = V // 2
    terrain = params.terrain[state.map_idx]                # (H, W) int8
    H, W = terrain.shape
    # build the (V, V) absolute coordinate grids
    rr = state.agent_r - half + jnp.arange(V, dtype=jnp.int32)[:, None]   # (V, 1)
    cc = state.agent_c - half + jnp.arange(V, dtype=jnp.int32)[None, :]   # (1, V)
    in_bounds = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    safe_rr = jnp.clip(rr, 0, H - 1)
    safe_cc = jnp.clip(cc, 0, W - 1)
    crop = terrain[safe_rr, safe_cc]
    crop = jnp.where(in_bounds, crop, jnp.int8(C.OOB))
    return crop  # (V, V) int8


def compass_unit_vector(state: EnvState, params: EnvParams) -> jax.Array:
    """(2,) float32 unit vector from agent → target. Used as a scalar input."""
    tr = params.target[state.map_idx, 0]
    tc = params.target[state.map_idx, 1]
    dr = (tr - state.agent_r).astype(jnp.float32)
    dc = (tc - state.agent_c).astype(jnp.float32)
    norm = jnp.sqrt(dr * dr + dc * dc) + 1e-6
    return jnp.stack([dr / norm, dc / norm])  # (compass_row, compass_col)


def scalars_obs(state: EnvState, params: EnvParams) -> jax.Array:
    """(4,) float32 — [compass_r, compass_c, build_active, step/max].

    NOTE: the *identity* of the active object is intentionally NOT
    observable — only the binary ``build_active`` flag is. The agent
    has to remember which item it committed to (raft vs harness) from
    the build action it took, which is a partial-observability problem
    that requires recurrent memory to solve.
    """
    compass = compass_unit_vector(state, params)
    build_active = (state.active_object != C.OBJ_NONE).astype(jnp.float32)
    step_norm = state.step_count.astype(jnp.float32) / float(params.max_steps)
    return jnp.concatenate([compass, jnp.stack([build_active, step_norm])])


def build_obs(state: EnvState, params: EnvParams) -> dict:
    """Dict observation matching the JAX Dreamer convention.

    Keys:
        minimap : int8  (V, V)
        scalars : float (4,)
    """
    return {
        "minimap": egocentric_minimap(state, params),
        "scalars": scalars_obs(state, params),
    }
