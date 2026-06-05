"""Gymnax-style wrapper around the pure-JAX bridge_tunnel dynamics (both variants).

Drop-in for the purejaxwm BatchEnv/AutoReset/Log wrapper chain that drives
DreamerV3. The variant is the static ``params.commit`` flag (obs scalar dim and
step logic follow it)."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from . import constants as C
from . import dynamics as dyn
from .render import build_obs
from .state import EnvParams, EnvState


class BridgeTunnelJaxEnv(environment.Environment[EnvState, EnvParams]):
    """Pure-JAX bridge_tunnel env; variant via the static ``params.commit`` flag."""

    def __init__(self, default_params: EnvParams | None = None):
        super().__init__()
        self._default_params = default_params

    @property
    def default_params(self) -> EnvParams:
        if self._default_params is None:
            raise RuntimeError(
                "BridgeTunnelJaxEnv needs EnvParams with map arrays "
                "(bridge_tunnel.jax.maps.generate_map_dataset + EnvParams.from_map_arrays).")
        return self._default_params

    @property
    def name(self) -> str:
        return "BridgeTunnelJax-v0"

    @property
    def num_actions(self) -> int:
        return C.NUM_ACTIONS

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(C.NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        V = params.view_size
        n_scalars = 7 if params.commit else 5
        return spaces.Dict({
            "minimap": spaces.Box(low=0, high=C.NUM_TILES - 1, shape=(V, V), dtype=jnp.int8),
            "scalars": spaces.Box(low=-1.0, high=1.0, shape=(n_scalars,), dtype=jnp.float32),
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
        return self.step_env(key, state, action, params if params is not None else self.default_params)


# back-compat alias (the commit jax env was a separate class name)
BridgeTunnelCommitJaxEnv = BridgeTunnelJaxEnv
