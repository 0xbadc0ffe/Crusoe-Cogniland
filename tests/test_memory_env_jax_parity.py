"""MDP parity: pure-JAX MemoryEnv == MiniGrid MemoryEnv.

The JAX env emits a SYMBOLIC obs (tile ids) while the MiniGrid env emits RGB, so
parity is checked at the MDP level — reward, termination, truncation, branch
entry and door selection — NOT on pixels. For each episode we read the MiniGrid
env's sampled (cue_type, door positions, cue cell), construct the JAX state with
the same values (cross-RNG draws can't be identical), replay the same action
sequence (oracle + random) and assert step-for-step agreement.

Requires both ``minigrid`` and ``jax`` (the full project env / `.venv`).
"""
import numpy as np
import pytest

pytest.importorskip("minigrid")
pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from cogniland.memory_env import MemoryEnv, MemoryEnvConfig, oracle_action
from cogniland.memory_env.jax import (
    MemoryJaxEnv, EnvParams, make_state, constants as C,
)

CUE_TYPES = ("green_up", "blue_up", "green_down", "blue_down")
_TB = {None: C.BRANCH_NONE, "up": C.BRANCH_UP, "down": C.BRANCH_DOWN}
_SD = {None: C.DOOR_NONE, "green": C.SEL_GREEN, "blue": C.SEL_BLUE}
# MiniGrid-native action id -> JAX action id (toggle 5 == A_OPEN 3; others equal)
_MG2JAX = {0: 0, 1: 1, 2: 2, 5: 3}


def test_jax_matches_minigrid_dynamics():
    params = EnvParams.from_config()            # defaults == MemoryEnvConfig() defaults
    jenv = MemoryJaxEnv(params)
    key = jax.random.PRNGKey(0)
    step = jax.jit(lambda st, a: jenv.step_env(key, st, a, params))

    rng = np.random.default_rng(0)
    n_checked = 0
    n_oracle_solved = 0
    for ep in range(60):
        env = MemoryEnv(MemoryEnvConfig())
        _, info = env.reset(seed=1000 + ep)
        ct = CUE_TYPES.index(info["cue_type"])
        st = make_state(params, ct, info["door_position_green"] == "top",
                        int(env._mg._cue_pos[0]), int(env._mg._cue_pos[1]))
        use_oracle = (ep % 2 == 0)

        info_cur = info
        term = trunc = False
        t = 0
        while not (term or trunc) and t < 260:
            a = oracle_action(info_cur) if use_oracle else int(rng.choice([0, 1, 2, 5]))
            _, pt_r, term, trunc, info_cur = env.step(a)

            _, st, jr, jdone, jinfo = step(st, jnp.int32(_MG2JAX[a]))
            jr = float(jr); jterm = bool(jinfo["is_terminal"]); jdone = bool(jdone)
            jtrunc = jdone and not jterm
            ctx = f"ep{ep}(oracle={use_oracle}) t{t} a{a}"
            assert abs(jr - float(pt_r)) < 1e-5, f"{ctx}: reward jax={jr} mg={pt_r}"
            assert jterm == bool(term), f"{ctx}: terminated jax={jterm} mg={term}"
            assert jtrunc == bool(trunc), f"{ctx}: truncated jax={jtrunc} mg={trunc}"
            assert int(st.taken_branch) == _TB[info_cur["taken_branch"]], f"{ctx}: branch"
            assert int(st.selected_door) == _SD[info_cur["selected_door_color"]], f"{ctx}: door"
            t += 1
            n_checked += 1
        if use_oracle and info_cur["success"]:
            n_oracle_solved += 1

    assert n_checked > 3000, f"only checked {n_checked} steps"
    assert n_oracle_solved == 30, f"oracle should solve all 30 oracle episodes, got {n_oracle_solved}"


def test_jax_marker_door_mechanics():
    """Closed marker blocks; A_OPEN opens it (pays the branch reward), stays open."""
    params = EnvParams.from_config(shaping_coef=0.0)
    key = jax.random.PRNGKey(0)
    # cue green_up (idx 0): correct branch is UP -> top marker is the correct one
    st = make_state(params, 0, True, params.x_room_start, params.row_room_up)
    st = st.replace(agent_x=jnp.int32(params.x_mark - 1),
                    agent_y=jnp.int32(params.row_up),
                    agent_dir=jnp.int32(C.DIR_EAST))
    from cogniland.memory_env.jax import step as jstep, build_obs

    # marker visible ahead in the egocentric crop
    mm = np.asarray(build_obs(st, params)["minimap"])
    assert (mm == C.MARK_A).sum() == 1

    # forward into the closed door: blocked
    s1, r1, _, _ = jstep(key, st, jnp.int32(C.A_FORWARD), params)
    assert int(s1.agent_x) == params.x_mark - 1 and not bool(s1.mark_top_open)

    # open: door opens, branch_bonus paid, agent does not move
    s2, r2, _, _ = jstep(key, s1, jnp.int32(C.A_OPEN), params)
    assert bool(s2.mark_top_open) and not bool(s2.mark_bot_open)
    assert abs(float(r2) - 0.5) < 1e-6          # branch_bonus at the OPEN event
    assert int(s2.agent_x) == params.x_mark - 1
    assert (np.asarray(build_obs(s2, params)["minimap"]) == C.MARK_A).sum() == 0

    # second open is a no-op (no double reward); forward now passes through
    s3, r3, _, _ = jstep(key, s2, jnp.int32(C.A_OPEN), params)
    assert abs(float(r3)) < 1e-6
    s4, _, _, _ = jstep(key, s3, jnp.int32(C.A_FORWARD), params)
    assert int(s4.agent_x) == params.x_mark

    # wrong marker (bottom, cue is up): opening pays wrong_branch_penalty (0 by default)
    params_pen = EnvParams.from_config(shaping_coef=0.0, wrong_branch_penalty=-1.0)
    sb = make_state(params_pen, 0, True, params_pen.x_room_start, params_pen.row_room_up)
    sb = sb.replace(agent_x=jnp.int32(params_pen.x_mark - 1),
                    agent_y=jnp.int32(params_pen.row_lo),
                    agent_dir=jnp.int32(C.DIR_EAST))
    sb1, rb, _, _ = jstep(key, sb, jnp.int32(C.A_OPEN), params_pen)
    assert bool(sb1.mark_bot_open) and abs(float(rb) + 1.0) < 1e-6

    # only the FIRST open pays: correct marker opened AFTER the wrong one -> 0
    # (kills the loop-around bonus-farming exploit through the reconnect corridor)
    sc = sb1.replace(agent_x=jnp.int32(params_pen.x_mark - 1),
                     agent_y=jnp.int32(params_pen.row_up),
                     agent_dir=jnp.int32(C.DIR_EAST))
    sc1, rc, _, _ = jstep(key, sc, jnp.int32(C.A_OPEN), params_pen)
    assert bool(sc1.mark_top_open) and abs(float(rc)) < 1e-6
