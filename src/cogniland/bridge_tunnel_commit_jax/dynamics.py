"""Step logic for the pure-JAX bridge_tunnel_commit env.

A bit-for-bit JAX port of ``BridgeTunnelCommitEnv.step``. Reward:

    r = slack_penalty
      + shaping_coef · (ctg_prev − γ · ctg_curr)         # commit-aware PBRS
      + reach_bonus · [moved onto TARGET]
      − build_cost · [BUILD on water | MINE on rock, while committed]

``ctg_prev`` indexes the static cost-to-go field by the **pre-action**
commitment + pre-action position; ``ctg_curr`` indexes by the **post-action**
commitment + post-action position. Committing this step therefore switches the
potential's field (rock/water becomes a wall in the wrong field), which is what
shapes the commit decision.

Movement: actions 0–3 update ``facing`` even when blocked. BUILD/MINE/COMMIT
never move the agent and never change ``facing``. BUILD edits a faced WATER cell
only while committed to build; MINE edits a faced ROCK cell only while committed
to mine. COMMIT_BUILD / COMMIT_MINE set the slot only from NONE (otherwise no-op).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import constants as C
from .state import EnvParams, EnvState


# facing-id → (dr, dc); F_UP/F_DOWN/F_LEFT/F_RIGHT = 0/1/2/3
_FACE_DROW = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)
_FACE_DCOL = jnp.array([0, 0, -1, 1], dtype=jnp.int32)


def _is_walkable(tile: jax.Array) -> jax.Array:
    return (
        (tile == C.GRASS)
        | (tile == C.WOOD)
        | (tile == C.TARGET)
        | (tile == C.SAND)
        | (tile == C.DIRT)
    )


def reset(rng: jax.Array, params: EnvParams) -> EnvState:
    """Sample a random map; spawn at its stored spawn, facing right, uncommitted."""
    map_idx = jax.random.randint(rng, (), 0, params.num_maps)
    sr = params.spawn[map_idx, 0]
    sc = params.spawn[map_idx, 1]
    return EnvState(
        map_idx=map_idx,
        terrain=params.terrain[map_idx],          # (H, W) — mutable copy
        agent_r=sr,
        agent_c=sc,
        facing=jnp.int32(C.F_RIGHT),
        commit=jnp.int32(C.COMMIT_NONE),
        step_count=jnp.int32(0),
    )


def step(
    rng: jax.Array,
    state: EnvState,
    action: jax.Array,
    params: EnvParams,
) -> tuple[EnvState, jax.Array, jax.Array, dict]:
    """Single env step. Returns (new_state, reward, done, info)."""
    H = params.height
    W = params.width
    ctg_all = params.ctg[state.map_idx]              # (3, H, W) commit-indexed fields

    is_move = action < 4
    is_build = action == C.A_BUILD
    is_mine = action == C.A_MINE
    is_cbuild = action == C.A_COMMIT_BUILD
    is_cmine = action == C.A_COMMIT_MINE

    commit_prev = state.commit
    # ctg at the pre-action commitment + position
    ctg_prev = ctg_all[commit_prev, state.agent_r, state.agent_c]

    # --- facing update (move actions only) ---
    new_facing = jnp.where(is_move, jnp.clip(action, 0, 3), state.facing)

    dr = _FACE_DROW[new_facing]
    dc = _FACE_DCOL[new_facing]
    fr = state.agent_r + dr
    fc = state.agent_c + dc
    in_bounds = (fr >= 0) & (fr < H) & (fc >= 0) & (fc < W)
    safe_fr = jnp.clip(fr, 0, H - 1)
    safe_fc = jnp.clip(fc, 0, W - 1)
    front_tile = state.terrain[safe_fr, safe_fc]

    # --- move ---
    can_step = is_move & in_bounds & _is_walkable(front_tile)
    new_r = jnp.where(can_step, fr, state.agent_r)
    new_c = jnp.where(can_step, fc, state.agent_c)
    reached = can_step & (front_tile == C.TARGET)

    # --- commitment update (only from NONE) ---
    is_none = commit_prev == C.COMMIT_NONE
    new_commit = jnp.where(is_none & is_cbuild, jnp.int32(C.COMMIT_BUILD),
                  jnp.where(is_none & is_cmine, jnp.int32(C.COMMIT_MINE),
                            commit_prev))

    # --- BUILD: water → wood (only if committed to build) ---
    do_place = is_build & (commit_prev == C.COMMIT_BUILD) & in_bounds & (front_tile == C.WATER)
    # --- MINE: rock → grass (only if committed to mine) ---
    do_mine = is_mine & (commit_prev == C.COMMIT_MINE) & in_bounds & (front_tile == C.ROCK)
    new_tile = jnp.where(do_place, jnp.int8(C.WOOD),
                         jnp.where(do_mine, jnp.int8(C.GRASS), front_tile))
    new_terrain = jnp.where(
        do_place | do_mine,
        state.terrain.at[safe_fr, safe_fc].set(new_tile),
        state.terrain,
    )

    # --- reward (ctg_curr indexes the post-action commitment + position) ---
    ctg_curr = ctg_all[new_commit, new_r, new_c]
    reward = jnp.float32(params.slack_penalty)
    reward = reward + params.reach_bonus * reached.astype(jnp.float32)
    reward = reward - params.build_cost * (do_place | do_mine).astype(jnp.float32)
    reward = reward + params.shaping_coef * (ctg_prev - params.gamma * ctg_curr)

    new_step_count = state.step_count + 1
    terminated = reached
    truncated = (~terminated) & (new_step_count >= params.max_steps)
    done = terminated | truncated

    new_state = EnvState(
        map_idx=state.map_idx,
        terrain=new_terrain,
        agent_r=new_r,
        agent_c=new_c,
        facing=new_facing,
        commit=new_commit,
        step_count=new_step_count,
    )
    info = {
        "reached_target": reached,
        "is_terminal": terminated,
        "placed": do_place,
        "mined": do_mine,
        "commit": new_commit,
        "category": params.category[state.map_idx],
    }
    return new_state, jnp.float32(reward), done, info
