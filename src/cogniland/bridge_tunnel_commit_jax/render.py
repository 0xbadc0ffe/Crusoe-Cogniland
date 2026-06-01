"""Egocentric tile-id minimap renderer for the pure-JAX bridge_tunnel_commit env.

Port of ``BridgeTunnelCommitEnv._make_obs`` / ``_egocentric_crop``:

* ``minimap`` : (V, V) int8 egocentric crop of the current (mutable) terrain,
  OOB-padded. The agent's own cell shows the raw tile id (not overwritten).
* ``scalars`` : (7,) float32 = [facing one-hot (4), step/max, commit==build, commit==mine].
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import constants as C
from .state import EnvParams, EnvState


def egocentric_minimap(state: EnvState, params: EnvParams) -> jax.Array:
    V = params.view_size
    half = V // 2
    terrain = state.terrain
    H = params.height
    W = params.width
    rr = state.agent_r - half + jnp.arange(V, dtype=jnp.int32)[:, None]
    cc = state.agent_c - half + jnp.arange(V, dtype=jnp.int32)[None, :]
    in_bounds = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    safe_rr = jnp.clip(rr, 0, H - 1)
    safe_cc = jnp.clip(cc, 0, W - 1)
    crop = terrain[safe_rr, safe_cc]
    crop = jnp.where(in_bounds, crop, jnp.int8(C.OOB))
    return crop.astype(jnp.int8)


def scalars_obs(state: EnvState, params: EnvParams) -> jax.Array:
    """(7,) float32 — [facing one-hot (4), step/max, commit_build, commit_mine]."""
    face_oh = jax.nn.one_hot(state.facing, 4, dtype=jnp.float32)
    step_norm = state.step_count.astype(jnp.float32) / float(max(1, params.max_steps))
    commit_build = (state.commit == C.COMMIT_BUILD).astype(jnp.float32)
    commit_mine = (state.commit == C.COMMIT_MINE).astype(jnp.float32)
    return jnp.concatenate([
        face_oh,
        step_norm[None],
        commit_build[None],
        commit_mine[None],
    ])


def build_obs(state: EnvState, params: EnvParams) -> dict:
    return {
        "minimap": egocentric_minimap(state, params),
        "scalars": scalars_obs(state, params),
    }
