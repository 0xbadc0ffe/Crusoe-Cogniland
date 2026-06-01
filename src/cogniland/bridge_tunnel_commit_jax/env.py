"""Gymnax-style ``Environment`` wrapper around the pure-JAX bridge_tunnel_commit
dynamics — drop-in for the purejaxwm BatchEnv/AutoReset/Log wrapper chain that
drives DreamerV3 (mirrors ``bridge_tunnel_jax/env.py``)."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from . import constants as C
from . import dynamics as dyn
from .render import build_obs
from .state import EnvParams, EnvState


class BridgeTunnelCommitJaxEnv(environment.Environment[EnvState, EnvParams]):
    """The pure-JAX bridge_tunnel_commit env (natural maps, commit mechanic)."""

    def __init__(self, default_params: EnvParams | None = None):
        super().__init__()
        self._default_params = default_params

    @property
    def default_params(self) -> EnvParams:
        if self._default_params is None:
            raise RuntimeError(
                "BridgeTunnelCommitJaxEnv needs an EnvParams with map arrays. Use "
                "`bridge_tunnel_commit_jax.maps.generate_map_dataset(...)` + "
                "`EnvParams.from_map_arrays(...)` first."
            )
        return self._default_params

    @property
    def name(self) -> str:
        return "BridgeTunnelCommitJax-v0"

    @property
    def num_actions(self) -> int:
        return C.NUM_ACTIONS

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(C.NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        V = params.view_size
        return spaces.Dict({
            "minimap": spaces.Box(
                low=0, high=C.NUM_TILES - 1, shape=(V, V), dtype=jnp.int8,
            ),
            "scalars": spaces.Box(low=-1.0, high=1.0, shape=(C.N_SCALARS,), dtype=jnp.float32),
        })

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict({})

    # ── Lifecycle ────────────────────────────────────────────────────

    def reset_env(self, key: jax.Array, params: EnvParams) -> tuple[dict, EnvState]:
        state = dyn.reset(key, params)
        obs = build_obs(state, params)
        return obs, state

    def step_env(self, key: jax.Array, state: EnvState, action, params: EnvParams):
        action = jnp.asarray(action, dtype=jnp.int32)
        new_state, reward, done, info = dyn.step(key, state, action, params)
        obs = build_obs(new_state, params)
        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: jax.Array, params: EnvParams | None = None):
        if params is None:
            params = self.default_params
        return self.reset_env(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key: jax.Array, state: EnvState, action, params: EnvParams | None = None):
        if params is None:
            params = self.default_params
        return self.step_env(key, state, action, params)
