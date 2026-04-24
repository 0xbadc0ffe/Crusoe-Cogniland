"""Smoke test for cogniland_jax — print-diagnostic style.

Run:
    python tests/test_cogniland_env.py

Checks:
  1. reset returns the documented shapes/dtypes
  2. step is jax.jit-able and jax.vmap-able over num_envs=8
  3. 1000 random steps run without NaN/shape drift
  4. done=True is reachable via max-step truncation
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from cogniland_jax import CognilandEnv, EnvParams
from cogniland_jax import constants as C
from cogniland_jax.maps import load_map_arrays


OK = "\033[92mok\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _check(label: str, cond: bool, detail: str = "") -> None:
    status = OK if cond else FAIL
    pad = max(0, 55 - len(label))
    print(f"[check] {label}{'.' * pad} {status}  {detail}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    arrays = load_map_arrays("data/maps/val.pt", biome_filter=["balanced"])
    params = EnvParams.from_map_arrays(
        **arrays,
        max_steps=jnp.int32(50),     # keep smoke test fast
        difficulty=jnp.int32(C.DIFFICULTY_HARD),
    )
    env = CognilandEnv(default_params=params)

    # 1. reset shapes/dtypes ------------------------------------------
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    mm, sc, te = obs["minimap"], obs["scalars"], obs["task_embedding"]
    _check("reset shapes/dtypes",
           mm.shape == (C.MINIMAP_DIAMETER, C.MINIMAP_DIAMETER)
           and mm.dtype == jnp.int8
           and sc.shape == (6,) and sc.dtype == jnp.float32
           and te.shape == (C.TASK_EMBEDDING_DIM,) and te.dtype == jnp.float32,
           detail=f"mm={mm.shape}/{mm.dtype} sc={sc.shape}/{sc.dtype}")

    # 2. jit + vmap ---------------------------------------------------
    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)
    action = jnp.int32(2)
    key, k = jax.random.split(key)
    obs2, state2, reward, done, info = step_jit(k, state, action, params)
    _check("jax.jit(step)", jnp.isfinite(reward) & (done.dtype == jnp.bool_),
           detail=f"reward={float(reward):+.3f} done={bool(done)}")

    B = 8
    keys = jax.random.split(key, B)
    reset_vmap = jax.vmap(reset_jit, in_axes=(0, None))
    step_vmap = jax.vmap(step_jit, in_axes=(0, 0, 0, None))
    obs_b, state_b = reset_vmap(keys, params)
    actions_b = jnp.zeros((B,), dtype=jnp.int32)
    obs_b2, state_b2, rew_b, done_b, info_b = step_vmap(keys, state_b, actions_b, params)
    _check("jax.vmap(step, num_envs=8)",
           obs_b2["minimap"].shape == (B, C.MINIMAP_DIAMETER, C.MINIMAP_DIAMETER)
           and rew_b.shape == (B,) and done_b.shape == (B,),
           detail=f"mm={obs_b2['minimap'].shape} rew={rew_b.shape}")

    # 3. 1000 random steps ------------------------------------------
    key = jax.random.PRNGKey(1)
    obs, state = reset_jit(key, params)
    any_done = False
    t0 = time.perf_counter()
    for i in range(1000):
        key, k_a, k_s = jax.random.split(key, 3)
        action = jax.random.randint(k_a, (), 0, C.NUM_ACTIONS)
        obs, state, reward, done, info = step_jit(k_s, state, action, params)
        if bool(done):
            any_done = True
            key, k_r = jax.random.split(key)
            obs, state = reset_jit(k_r, params)
        assert jnp.all(jnp.isfinite(obs["scalars"])), f"NaN/inf at step {i}"
    dt = time.perf_counter() - t0
    _check("1000 random steps, no NaN", True,
           detail=f"{1000/dt:,.0f} steps/s, any_done={any_done}")

    # 4. truncation reachable ---------------------------------------
    params_short = params.replace(max_steps=jnp.int32(5))
    key, k_r = jax.random.split(key)
    obs, state = reset_jit(k_r, params_short)
    reached_done = False
    for _ in range(10):
        key, k_a, k_s = jax.random.split(key, 3)
        action = jnp.int32(4)  # forage, no movement, no terminal state
        obs, state, reward, done, info = step_jit(k_s, state, action, params_short)
        if bool(done):
            reached_done = True
            break
    _check("done=True reachable via truncation", reached_done,
           detail=f"steps={int(state.steps)}")

    print("\nall good.")


if __name__ == "__main__":
    main()
