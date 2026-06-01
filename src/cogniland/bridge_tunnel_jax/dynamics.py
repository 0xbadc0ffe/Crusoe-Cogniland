"""Step logic for the pure-JAX bridge_tunnel env.

A bit-for-bit JAX port of ``BridgeTunnelEnv.step`` (PyTorch). The reward is

    r = slack_penalty                                       # every step
      + shaping_coef · (ctg_prev − γ · ctg_curr)            # PBRS, every step
      + reach_bonus · [moved onto TARGET]                   # sparse terminal
      − build_cost · [PLACE on water | MINE on rock]        # build tax

where ``ctg`` is the STATIC-terrain min-action cost-to-go computed once at
reset (never recomputed after a build/mine) — so the PBRS potential is just an
indexed lookup into the precomputed ``ctg`` field. ``ctg_prev`` is read at the
agent's pre-action position and ``ctg_curr`` at its post-action position; the
shaping is applied on EVERY step (including blocked moves and PLACE/MINE),
matching the PyTorch env (which only gates the *position change*, not the
shaping term).

Movement: actions 0–3 always update ``facing`` even when the move is blocked.
PLACE/MINE never move the agent and never update ``facing``. Reaching the goal
only happens on a successful MOVE onto a TARGET cell.
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
    """Mirror of ``bridge_tunnel.tiles.is_walkable``: GRASS, WOOD, TARGET,
    SAND, DIRT."""
    return (
        (tile == C.GRASS)
        | (tile == C.WOOD)
        | (tile == C.TARGET)
        | (tile == C.SAND)
        | (tile == C.DIRT)
    )


def reset(rng: jax.Array, params: EnvParams) -> EnvState:
    """Sample a random map and return the initial state.

    Spawn / facing match the PyTorch env: spawn is the stored per-map spawn
    (centre of the left edge for natural maps), facing starts at F_RIGHT.
    """
    map_idx = jax.random.randint(rng, (), 0, params.num_maps)
    sr = params.spawn[map_idx, 0]
    sc = params.spawn[map_idx, 1]
    return EnvState(
        map_idx=map_idx,
        terrain=params.terrain[map_idx],          # (H, W) — mutable copy
        agent_r=sr,
        agent_c=sc,
        facing=jnp.int32(C.F_RIGHT),
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
    ctg = params.ctg[state.map_idx]                  # (H, W) static potential field

    is_move = action < 4
    is_place = action == C.A_PLACE
    is_mine = action == C.A_MINE

    # ctg at the pre-action position (read before anything moves)
    ctg_prev = ctg[state.agent_r, state.agent_c]

    # --- facing update (move actions only; PLACE/MINE keep facing) ---
    new_facing = jnp.where(is_move, jnp.clip(action, 0, 3), state.facing)

    # the facing cell delta uses the (possibly updated) facing — for a move the
    # facing equals the action; for PLACE/MINE it is the unchanged facing.
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

    # --- PLACE: water → wood in front ---
    do_place = is_place & in_bounds & (front_tile == C.WATER)
    # --- MINE: rock → grass in front ---
    do_mine = is_mine & in_bounds & (front_tile == C.ROCK)
    new_tile = jnp.where(do_place, jnp.int8(C.WOOD),
                         jnp.where(do_mine, jnp.int8(C.GRASS), front_tile))
    new_terrain = jnp.where(
        do_place | do_mine,
        state.terrain.at[safe_fr, safe_fc].set(new_tile),
        state.terrain,
    )

    # --- reward ---
    ctg_curr = ctg[new_r, new_c]
    reward = jnp.float32(params.slack_penalty)
    reward = reward + params.reach_bonus * reached.astype(jnp.float32)
    reward = reward - params.build_cost * (do_place | do_mine).astype(jnp.float32)
    # PBRS shaping applied every step (matches the PyTorch env: gated only by
    # shaping_coef != 0, which it is for natural_agent).
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
        step_count=new_step_count,
    )
    info = {
        "reached_target": reached,
        "is_terminal": terminated,
        "placed": do_place,
        "mined": do_mine,
    }
    return new_state, jnp.float32(reward), done, info
