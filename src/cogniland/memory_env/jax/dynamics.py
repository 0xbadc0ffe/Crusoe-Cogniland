"""Pure-JAX MemoryEnv dynamics: reset + step (branchless, jit/vmap-friendly).

Mirrors `_MemoryMiniGridEnv.step` in cogniland.memory_env.env at the MDP level
(movement, wall/cue/marker blocking, marker-door opening + branch reward, door
termination, PBRS shaping, truncation). Interventions (forced_branch /
suppress_*) default OFF in MiniGrid and are not modelled here (training never
uses them).

Marker doors (2026-07): each branch corridor is blocked mid-way by a neutral
marker door (MARK_A top / MARK_B bottom — branch-identity evidence, no colour
semantics). A_OPEN with the door directly ahead opens it (it stays open). The
branch reward is paid at the OPEN event, not at branch entry: branch_bonus for
the direction-correct marker, wrong_branch_penalty for the wrong one.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import constants as C
from .state import EnvParams, EnvState

_CUE_IS_DOWN = jnp.asarray(C.CUE_IS_DOWN, dtype=jnp.bool_)
_CUE_IS_BLUE = jnp.asarray(C.CUE_IS_BLUE, dtype=jnp.bool_)
_DX = jnp.asarray([v[0] for v in C.DIR_VEC], dtype=jnp.int32)
_DY = jnp.asarray([v[1] for v in C.DIR_VEC], dtype=jnp.int32)


def make_state(params: EnvParams, cue_type, door_green_top, cue_x, cue_y) -> EnvState:
    """Build the initial EnvState for explicit episode params (used by reset + parity)."""
    return EnvState(
        agent_x=jnp.int32(params.x_precue_start),
        agent_y=jnp.int32(params.my),
        agent_dir=jnp.int32(C.DIR_EAST),
        cue_type=jnp.int32(cue_type),
        door_green_top=jnp.asarray(door_green_top, dtype=jnp.bool_),
        cue_x=jnp.int32(cue_x),
        cue_y=jnp.int32(cue_y),
        taken_branch=jnp.int32(C.BRANCH_NONE),
        selected_door=jnp.int32(C.DOOR_NONE),
        mark_top_open=jnp.bool_(False),
        mark_bot_open=jnp.bool_(False),
        step_count=jnp.int32(0),
        prev_phi=jnp.float32(min(params.x_precue_start, params.x_doorcol)),
        terminated=jnp.bool_(False),
        done=jnp.bool_(False),
    )


def reset(rng: jax.Array, params: EnvParams) -> EnvState:
    k_cue, k_door, k_cx, k_cy, k_rand = jax.random.split(rng, 5)
    cue_type = jax.random.choice(k_cue, 4, p=params.cue_probs)
    # curriculum: with prob door_random_prob randomise door colours, else fixed
    # (green top) so colour->door is unconditional.
    door_green_top = jnp.where(jax.random.bernoulli(k_rand, params.door_random_prob),
                               jax.random.bernoulli(k_door, 0.5), True)
    cue_x = jax.random.randint(k_cx, (), params.x_room_start, params.x_room_end)
    cue_y = jnp.where(jax.random.bernoulli(k_cy, 0.5),
                      params.row_room_up, params.row_room_lo)
    return make_state(params, cue_type, door_green_top, cue_x, cue_y)


def step(rng, state: EnvState, action, params: EnvParams):
    action = jnp.asarray(action, dtype=jnp.int32)
    d = state.agent_dir

    # turns change facing; forward moves in the CURRENT facing.
    turn_l = action == C.A_LEFT
    turn_r = action == C.A_RIGHT
    fwd = action == C.A_FORWARD
    opn = action == C.A_OPEN
    new_dir = (d + turn_r.astype(jnp.int32) - turn_l.astype(jnp.int32)) % 4

    tx = state.agent_x + _DX[d]
    ty = state.agent_y + _DY[d]
    tile = params.base_terrain[ty, tx]
    # marker doors: the cell ahead is a CLOSED marker door -> blocks forward
    ahead_mark_top = (tx == params.x_mark) & (ty == params.row_up)
    ahead_mark_bot = (tx == params.x_mark) & (ty == params.row_lo)
    closed_ahead = ((ahead_mark_top & ~state.mark_top_open)
                    | (ahead_mark_bot & ~state.mark_bot_open))
    blocked = ((tile == C.WALL) | ((tx == state.cue_x) & (ty == state.cue_y))
               | closed_ahead)
    can_move = fwd & jnp.logical_not(blocked)
    nx = jnp.where(can_move, tx, state.agent_x)
    ny = jnp.where(can_move, ty, state.agent_y)

    # ── marker-door opening + branch reward (paid at the OPEN event) ─────
    # Reward only the FIRST marker opened in the episode (the commitment),
    # else the reconnect corridor allows farming the correct-marker bonus
    # after passing through the wrong branch.
    open_top = opn & ahead_mark_top & ~state.mark_top_open
    open_bot = opn & ahead_mark_bot & ~state.mark_bot_open
    first_open = ~state.mark_top_open & ~state.mark_bot_open
    new_mark_top = state.mark_top_open | open_top
    new_mark_bot = state.mark_bot_open | open_bot
    correct_is_down = _CUE_IS_DOWN[state.cue_type]
    opened_correct = ((open_top & ~correct_is_down) | (open_bot & correct_is_down)) & first_open
    opened_wrong = ((open_top | open_bot) & first_open) & ~opened_correct

    # ── branch entry (recorded for analysis; no reward here any more) ────
    in_branch = (nx >= params.x_branch_start) & (nx <= params.x_branch_end)
    on_up = ny == params.row_up
    on_lo = ny == params.row_lo
    newly = (state.taken_branch == C.BRANCH_NONE) & in_branch & (on_up | on_lo)
    branch_val = jnp.where(on_up, C.BRANCH_UP, C.BRANCH_DOWN)
    new_taken = jnp.where(newly, branch_val, state.taken_branch)

    # ── door termination ─────────────────────────────────────────────────
    on_top = (nx == params.x_doorcol) & (ny == params.row_door_top)
    on_bot = (nx == params.x_doorcol) & (ny == params.row_door_bot)
    on_door = on_top | on_bot
    sel_green = jnp.where(on_top, state.door_green_top,
                          jnp.logical_not(state.door_green_top))
    sel_door = jnp.where(on_door, jnp.where(sel_green, C.SEL_GREEN, C.SEL_BLUE),
                         C.DOOR_NONE)
    target_green = jnp.logical_not(_CUE_IS_BLUE[state.cue_type])
    success = on_door & (sel_green == target_green)
    wrong = on_door & jnp.logical_not(success)

    # ── reward (step_penalty + marker-open branch reward + door + PBRS) ──
    phi = jnp.minimum(nx, params.x_doorcol).astype(jnp.float32)
    reward = (jnp.float32(params.step_penalty)
              + jnp.where(opened_correct, params.branch_bonus, 0.0)
              + jnp.where(opened_wrong, params.wrong_branch_penalty, 0.0)
              + jnp.where(success, params.success_reward, 0.0)
              + jnp.where(wrong, params.wrong_door_reward, 0.0)
              + params.shaping_coef * (phi - state.prev_phi))

    step_count = state.step_count + 1
    terminated = on_door
    truncated = step_count >= params.max_steps
    done = terminated | truncated

    new_state = EnvState(
        agent_x=nx, agent_y=ny, agent_dir=new_dir,
        cue_type=state.cue_type, door_green_top=state.door_green_top,
        cue_x=state.cue_x, cue_y=state.cue_y,
        taken_branch=new_taken, selected_door=sel_door,
        mark_top_open=new_mark_top, mark_bot_open=new_mark_bot,
        step_count=step_count, prev_phi=phi,
        terminated=terminated, done=done,
    )
    info = {
        "reached_target": success,
        "is_terminal": terminated,
        "category": jnp.int32(0),
    }
    return new_state, reward.astype(jnp.float32), done, info
