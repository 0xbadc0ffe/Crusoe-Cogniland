"""Egocentric obs for the pure-JAX bridge_tunnel env (both variants).

* ``minimap`` (V,V) int8 — egocentric crop, OOB-padded.
* ``scalars`` — bt: ``[facing(4), step/max]`` (5); btc: ``[facing(4), step/max,
  commit_build, commit_mine]`` (7). Selected by the static ``params.commit``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import constants as C
from .state import EnvParams, EnvState


def egocentric_minimap(state: EnvState, params: EnvParams) -> jax.Array:
    V = params.view_size; half = V // 2
    H, W = params.height, params.width
    rr = state.agent_r - half + jnp.arange(V, dtype=jnp.int32)[:, None]
    cc = state.agent_c - half + jnp.arange(V, dtype=jnp.int32)[None, :]
    in_bounds = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    crop = state.terrain[jnp.clip(rr, 0, H - 1), jnp.clip(cc, 0, W - 1)]
    return jnp.where(in_bounds, crop, jnp.int8(C.OOB)).astype(jnp.int8)


def scalars_obs(state: EnvState, params: EnvParams) -> jax.Array:
    face_oh = jax.nn.one_hot(state.facing, 4, dtype=jnp.float32)
    step_norm = state.step_count.astype(jnp.float32) / float(max(1, params.max_steps))
    if params.commit:
        cb = (state.commit == C.COMMIT_BUILD).astype(jnp.float32)
        cm = (state.commit == C.COMMIT_MINE).astype(jnp.float32)
        return jnp.concatenate([face_oh, step_norm[None], cb[None], cm[None]])
    return jnp.concatenate([face_oh, step_norm[None]])


def build_obs(state: EnvState, params: EnvParams) -> dict:
    return {"minimap": egocentric_minimap(state, params),
            "scalars": scalars_obs(state, params)}
