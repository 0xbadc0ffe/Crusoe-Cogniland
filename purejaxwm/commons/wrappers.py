"""Env wrappers, ported from https://github.com/mttga/purejaxql/blob/main/purejaxql/utils/craftax_wrappers.py

Intentionally tiny. `commons/` starts with wrappers and ideally nothing else, and grows reactively *only* when 
a primitive has proved itself useful across at least three real implementations without constraining any of them.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct


class GymnaxWrapper:
    """Base class that proxies attribute access to the wrapped env."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """vmap-based vectorization over num_envs."""

    def __init__(self, env, num_envs: int):
        super().__init__(env)
        self.num_envs = num_envs
        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, sub = jax.random.split(rng)
        rngs = jax.random.split(sub, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, sub = jax.random.split(rng)
        rngs = jax.random.split(sub, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)
        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Single-env auto-reset (standard Gymnax semantics)."""

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, sub1 = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(sub1, state, action, params)
        rng, sub2 = jax.random.split(rng)
        obs_re, state_re = self._env.reset(sub2, params)

        def select(done, x_done, x_step):
            return jax.lax.select(done, x_done, x_step)

        state = jax.tree_util.tree_map(lambda r, s: select(done, r, s), state_re, state_st)
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """Efficient vectorized auto-reset.

    Only `num_envs // reset_ratio` fresh resets are computed per step, and they are
    distributed to whichever envs are actually done. Saves wall-clock at the cost of
    a small probability of duplicate reset allocation. Ported verbatim from
    purejaxql's craftax_wrappers.py (battle-tested on Craftax).
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)
        assert num_envs % reset_ratio == 0, "Reset ratio must divide num_envs."
        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        self.num_resets = num_envs // reset_ratio
        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, sub = jax.random.split(rng)
        rngs = jax.random.split(sub, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, sub1 = jax.random.split(rng)
        rngs_step = jax.random.split(sub1, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs_step, state, action, params)

        rng, sub2 = jax.random.split(rng)
        rngs_reset = jax.random.split(sub2, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs_reset, params)

        rng, sub3 = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)
        being_reset = jax.random.choice(
            sub3,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree_util.tree_map(lambda x: x[reset_indexes], state_re)

        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_util.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)
            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Tracks episode returns and lengths; exposes them via the info dict."""

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_ep_return = state.episode_returns + reward
        new_ep_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_ep_return * (1 - done),
            episode_lengths=new_ep_length * (1 - done),
            returned_episode_returns=(
                state.returned_episode_returns * (1 - done) + new_ep_return * done
            ),
            returned_episode_lengths=(
                state.returned_episode_lengths * (1 - done) + new_ep_length * done
            ),
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info
