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
            a = oracle_action(info_cur) if use_oracle else int(rng.integers(0, 3))
            _, pt_r, term, trunc, info_cur = env.step(a)

            _, st, jr, jdone, jinfo = step(st, jnp.int32(a))
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
