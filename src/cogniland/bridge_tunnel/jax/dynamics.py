"""Step logic for the pure-JAX bridge_tunnel env (both variants).

A bit-for-bit JAX port of ``BridgeTunnelEnv.step``. The variant is a static
``params.commit`` flag, so the build/mine + shaping branches resolve at trace
time (no runtime cost):

* bt  (commit=False): BUILD/MINE always act; single ctg field; reward =
  slack + reach + shaping − build_cost.
* btc (commit=True): first successful BUILD/MINE commits; opposite tool locked
  (no-op + illegal_penalty); 3-field commit-indexed ctg; +commit_cost on commit.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import constants as C
from .state import EnvParams, EnvState

_FACE_DROW = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)
_FACE_DCOL = jnp.array([0, 0, -1, 1], dtype=jnp.int32)


def _is_walkable(tile):
    return ((tile == C.GRASS) | (tile == C.WOOD) | (tile == C.TARGET)
            | (tile == C.SAND) | (tile == C.DIRT))


def reset(rng: jax.Array, params: EnvParams) -> EnvState:
    map_idx = jax.random.randint(rng, (), 0, params.num_maps)
    return EnvState(
        map_idx=map_idx, terrain=params.terrain[map_idx],
        agent_r=params.spawn[map_idx, 0], agent_c=params.spawn[map_idx, 1],
        facing=jnp.int32(C.F_RIGHT), commit=jnp.int32(C.COMMIT_NONE),
        step_count=jnp.int32(0),
    )


def step(rng, state, action, params):
    H, W = params.height, params.width
    commit_prev = state.commit
    is_move = action < 4
    is_build = action == C.A_BUILD
    is_mine = action == C.A_MINE

    if params.commit:
        ctg_all = params.ctg[state.map_idx]                 # (3,H,W)
        ctg_prev = ctg_all[commit_prev, state.agent_r, state.agent_c]
    else:
        ctg_field = params.ctg[state.map_idx]               # (H,W)
        ctg_prev = ctg_field[state.agent_r, state.agent_c]

    new_facing = jnp.where(is_move, jnp.clip(action, 0, 3), state.facing)
    dr = _FACE_DROW[new_facing]; dc = _FACE_DCOL[new_facing]
    fr = state.agent_r + dr; fc = state.agent_c + dc
    in_bounds = (fr >= 0) & (fr < H) & (fc >= 0) & (fc < W)
    safe_fr = jnp.clip(fr, 0, H - 1); safe_fc = jnp.clip(fc, 0, W - 1)
    front = state.terrain[safe_fr, safe_fc]

    can_step = is_move & in_bounds & _is_walkable(front)
    new_r = jnp.where(can_step, fr, state.agent_r)
    new_c = jnp.where(can_step, fc, state.agent_c)
    reached = can_step & (front == C.TARGET)

    is_none = commit_prev == C.COMMIT_NONE
    if params.commit:
        do_place = is_build & (commit_prev != C.COMMIT_MINE) & in_bounds & (front == C.WATER)
        do_mine = is_mine & (commit_prev != C.COMMIT_BUILD) & in_bounds & (front == C.ROCK)
        committed_build = do_place & is_none
        committed_mine = do_mine & is_none
        new_commit = jnp.where(committed_build, jnp.int32(C.COMMIT_BUILD),
                      jnp.where(committed_mine, jnp.int32(C.COMMIT_MINE), commit_prev))
        committed_now = committed_build | committed_mine
        illegal = (is_build & (commit_prev == C.COMMIT_MINE)) | (is_mine & (commit_prev == C.COMMIT_BUILD))
    else:
        do_place = is_build & in_bounds & (front == C.WATER)
        do_mine = is_mine & in_bounds & (front == C.ROCK)
        new_commit = commit_prev
        committed_now = jnp.zeros_like(reached)
        illegal = jnp.zeros_like(reached)

    new_tile = jnp.where(do_place, jnp.int8(C.WOOD),
                         jnp.where(do_mine, jnp.int8(C.GRASS), front))
    new_terrain = jnp.where(do_place | do_mine,
                            state.terrain.at[safe_fr, safe_fc].set(new_tile), state.terrain)

    if params.commit:
        ctg_curr = ctg_all[new_commit, new_r, new_c]
    else:
        ctg_curr = ctg_field[new_r, new_c]

    reward = jnp.float32(params.slack_penalty)
    reward = reward + params.reach_bonus * reached.astype(jnp.float32)
    reward = reward - params.build_cost * (do_place | do_mine).astype(jnp.float32)
    if params.commit:
        reward = reward - params.commit_cost * committed_now.astype(jnp.float32)
        reward = reward - params.illegal_penalty * illegal.astype(jnp.float32)
    reward = reward + params.shaping_coef * (ctg_prev - params.gamma * ctg_curr)

    new_step_count = state.step_count + 1
    terminated = reached
    truncated = (~terminated) & (new_step_count >= params.max_steps)
    done = terminated | truncated
    new_state = EnvState(
        map_idx=state.map_idx, terrain=new_terrain, agent_r=new_r, agent_c=new_c,
        facing=new_facing, commit=new_commit, step_count=new_step_count,
    )
    info = {
        "reached_target": reached, "is_terminal": terminated,
        "placed": do_place, "mined": do_mine,
        "commit": new_commit, "committed_now": committed_now,
        "category": params.category[state.map_idx],
    }
    return new_state, jnp.float32(reward), done, info
