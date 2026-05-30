"""Step logic for crafter_in_cogniland (pure JAX).

Mirrors src/cogniland/nav/nav_env.py + skills.py. The reward is

    r = SLACK_PENALTY                                  # paid every step
      + SHAPING_COEF · (ctg_prev - ctg_curr)           # PBRS on Δctg
      + REACH_BONUS · [reached target]

Walkability: water/rock/tree are always walkable but high-slip; lava and
OOB are blocked. Slip semantics (2026-05-28 hard-land weight tax):
    - water  : 0.75 unless agent carries raft       (raft → 0.0)
    - rock   : 0.75 unless agent carries harness    (harness → 0.0)
    - tree   : always 0.75
    - sand   : 0.75 if any skill is committed; 0.30 bare-handed
    - dirt   : 0.75 if any skill is committed; 0.30 bare-handed
    - grass  : 0.75 if any skill is committed; SLIP_PROB_GRASS_NOSKILL bare-handed (sweep knob)
    - target : never slips
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import constants as C
from .state import EnvParams, EnvState


# 4 cardinal moves
_MOVE_DROW = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)
_MOVE_DCOL = jnp.array([0, 0, -1, 1], dtype=jnp.int32)


def _is_walkable(tile: jax.Array) -> jax.Array:
    """True if ``tile`` can be stepped onto (regardless of object).

    Walkable: GRASS, DIRT, SAND, TARGET, WATER, ROCK, TREE.
    Blocked:  OOB, LAVA. (Out-of-bounds is handled separately.)
    """
    return (
        (tile == C.GRASS)
        | (tile == C.DIRT)
        | (tile == C.SAND)
        | (tile == C.TARGET)
        | (tile == C.WATER)
        | (tile == C.ROCK)
        | (tile == C.TREE)
    )


def _slip_chance(active_object: jax.Array, tile: jax.Array) -> jax.Array:
    """Probability that a move onto ``tile`` slips (stays in place).

    Mirror of ``nav/skills.py``: RAFT zeroes water, HARNESS zeroes rock; trees
    always slip 75 %%. **Hard-land weight tax (2026-05-28):** when ANY skill
    is committed, grass / sand / dirt all slip at ``SLIP_PROB_LAND_WITH_SKILL``
    (75 %%). Bare-handed: sand/dirt slip at ``SLIP_PROB_MINOR`` (30 %%),
    grass at 0 %%. The target tile never slips.
    """
    has_skill = active_object != C.OBJ_NONE
    p_water = jnp.where(active_object == C.OBJ_RAFT,    0.0, C.SLIP_PROB_DEFAULT)
    p_rock  = jnp.where(active_object == C.OBJ_HARNESS, 0.0, C.SLIP_PROB_DEFAULT)
    p_grass = jnp.where(has_skill, C.SLIP_PROB_LAND_WITH_SKILL, 0.0)
    p_sand  = jnp.where(has_skill, C.SLIP_PROB_LAND_WITH_SKILL, C.SLIP_PROB_MINOR)
    p_dirt  = jnp.where(has_skill, C.SLIP_PROB_LAND_WITH_SKILL, C.SLIP_PROB_MINOR)
    p = jnp.zeros_like(p_water)  # target / anything else — never slips
    p = jnp.where(tile == C.GRASS, p_grass, p)
    p = jnp.where(tile == C.SAND,  p_sand,  p)
    p = jnp.where(tile == C.DIRT,  p_dirt,  p)
    p = jnp.where(tile == C.WATER, p_water, p)
    p = jnp.where(tile == C.ROCK,  p_rock,  p)
    p = jnp.where(tile == C.TREE,  C.SLIP_PROB_DEFAULT, p)
    return p


def _ctg_for_object(
    params: EnvParams, state: EnvState, active_object: jax.Array
) -> jax.Array:
    """Return (H, W) ctg array for the agent's current map under ``active_object``."""
    none = params.ctg_none[state.map_idx]
    raft = params.ctg_raft[state.map_idx]
    harn = params.ctg_harness[state.map_idx]
    # avoid in-graph data-dependent indexing on a constant: pick via jnp.where
    a_raft = active_object == C.OBJ_RAFT
    a_harn = active_object == C.OBJ_HARNESS
    arr = jnp.where(a_raft[None, None], raft, none)
    arr = jnp.where(a_harn[None, None], harn, arr)
    return arr


def reset(rng: jax.Array, params: EnvParams) -> tuple[EnvState, jax.Array]:
    """Sample a random map and return the initial state + first observation."""
    rng_map, _ = jax.random.split(rng)
    map_idx = jax.random.randint(rng_map, (), 0, params.num_maps)
    sr = params.spawn[map_idx, 0]
    sc = params.spawn[map_idx, 1]
    state = EnvState(
        map_idx=map_idx,
        agent_r=sr,
        agent_c=sc,
        facing=jnp.int32(1),
        active_object=jnp.int32(C.OBJ_NONE),
        step_count=jnp.int32(0),
        last_ctg=params.ctg_none[map_idx, sr, sc],
    )
    return state


def step(
    rng: jax.Array,
    state: EnvState,
    action: jax.Array,
    params: EnvParams,
) -> tuple[EnvState, jax.Array, jax.Array, dict]:
    """Single env step. Returns (new_state, reward, done, info)."""
    rng_slip, _ = jax.random.split(rng)

    move_action = action < 4
    is_build = ~move_action
    build_raft = action == C.ACTION_BUILD_RAFT
    build_harness = action == C.ACTION_BUILD_HARNESS

    # --- proposed move ---
    dr = jnp.where(move_action, _MOVE_DROW[jnp.clip(action, 0, 3)], 0)
    dc = jnp.where(move_action, _MOVE_DCOL[jnp.clip(action, 0, 3)], 0)
    nr = state.agent_r + dr
    nc = state.agent_c + dc
    H = W = params.map_size
    in_bounds = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)

    # tile we'd step onto (use safe gather with clipped indices; mask later)
    safe_nr = jnp.clip(nr, 0, H - 1)
    safe_nc = jnp.clip(nc, 0, W - 1)
    target_tile = params.terrain[state.map_idx, safe_nr, safe_nc]
    walkable = _is_walkable(target_tile)
    can_step = move_action & in_bounds & walkable

    # slip — stay in place if a slip occurs
    p_slip = _slip_chance(state.active_object, target_tile)
    u = jax.random.uniform(rng_slip, ())
    slipped = can_step & (u < p_slip)
    actually_moved = can_step & ~slipped

    new_r = jnp.where(actually_moved, nr, state.agent_r)
    new_c = jnp.where(actually_moved, nc, state.agent_c)

    # --- build action (commit only if currently NONE) ---
    is_none = state.active_object == C.OBJ_NONE
    new_obj = jnp.where(
        is_build & is_none & build_raft,    jnp.int32(C.OBJ_RAFT),
        jnp.where(
            is_build & is_none & build_harness, jnp.int32(C.OBJ_HARNESS),
            state.active_object,
        ),
    )
    # facing only updates on a move action; doesn't change on build/slip
    new_facing = jnp.where(move_action, jnp.clip(action, 0, 3), state.facing)

    # --- new ctg (under the (possibly updated) active_object, at the
    # (possibly updated) position) ---
    ctg_arr = _ctg_for_object(params, state, new_obj)
    new_ctg = ctg_arr[new_r, new_c]

    # --- reward ---
    delta_ctg = state.last_ctg - new_ctg            # positive = closer
    delta_ctg = jnp.where(actually_moved, delta_ctg, 0.0)  # only paid on a real move
    moved_tile = params.terrain[state.map_idx, new_r, new_c]
    reached = actually_moved & (moved_tile == C.TARGET)
    reward = (
        C.SLACK_PENALTY
        + C.SHAPING_COEF * delta_ctg
        + C.REACH_BONUS * reached.astype(jnp.float32)
    )

    new_step_count = state.step_count + 1
    terminated = reached
    truncated = new_step_count >= params.max_steps
    done = terminated | truncated

    new_state = EnvState(
        map_idx=state.map_idx,
        agent_r=new_r,
        agent_c=new_c,
        facing=new_facing,
        active_object=new_obj,
        step_count=new_step_count,
        last_ctg=new_ctg,
    )
    info = {
        "reached_target": reached,
        "active_object": new_obj,
        "map_type": params.map_type[state.map_idx],
        "slipped": slipped,
        "is_terminal": terminated,
    }
    return new_state, reward, done, info
