"""Gymnax-style pure-JAX MemoryEnv (mirrors bridge_tunnel/jax/env.py)."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from . import constants as C
from . import dynamics as dyn
from .render import build_obs
from .state import EnvParams, EnvState


class MemoryJaxEnv(environment.Environment[EnvState, EnvParams]):
    """Pure-JAX MemoryEnv. Symbolic egocentric obs; Discrete(4) actions
    (turn-L / turn-R / forward / open-marker-door)."""

    def __init__(self, default_params: EnvParams | None = None):
        super().__init__()
        self._default_params = default_params

    @property
    def default_params(self) -> EnvParams:
        if self._default_params is None:
            raise RuntimeError("MemoryJaxEnv needs default EnvParams (EnvParams.from_config(...))")
        return self._default_params

    @property
    def name(self) -> str:
        return "MemoryJax-v0"

    @property
    def num_actions(self) -> int:
        return C.NUM_ACTIONS

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(C.NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        V = params.view_size
        return spaces.Dict({
            "minimap": spaces.Box(low=0, high=C.NUM_TILES - 1, shape=(V, V), dtype=jnp.int8),
            "scalars": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=jnp.float32),
        })

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict({})

    def reset_env(self, key, params):
        state = dyn.reset(key, params)
        return build_obs(state, params), state

    def step_env(self, key, state, action, params):
        action = jnp.asarray(action, dtype=jnp.int32)
        new_state, reward, done, info = dyn.step(key, state, action, params)
        return build_obs(new_state, params), new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self.reset_env(key, params if params is not None else self.default_params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        p = params if params is not None else self.default_params
        return self.step_env(key, state, action, p)
