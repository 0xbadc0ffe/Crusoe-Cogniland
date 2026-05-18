"""Gymnax-style ``Environment`` wrapper around the crafter_in_cogniland dynamics.

Provides ``reset(key, params)`` and ``step(key, state, action, params)``
methods with shapes that vmap cleanly, so the standard purejaxwm
``BatchEnvWrapper`` / ``AutoResetEnvWrapper`` / ``LogWrapper`` chain
works unchanged.

If you've used ``gymnax``, this should feel familiar — the only quirk is
that ``EnvParams`` (the map dataset) is not hashable, so JIT
``static_argnums`` skip the ``params`` argument. The wrapper closes over
``default_params`` instead.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from . import constants as C
from . import dynamics as dyn
from .render import build_obs
from .state import EnvParams, EnvState


class CrafterInCognilandEnv(environment.Environment[EnvState, EnvParams]):
    """The crafter_in_cogniland JAX env."""

    def __init__(self, default_params: EnvParams | None = None):
        super().__init__()
        self._default_params = default_params

    @property
    def default_params(self) -> EnvParams:
        if self._default_params is None:
            raise RuntimeError(
                "CrafterInCognilandEnv needs an EnvParams with map arrays. "
                "Use `crafter_in_cogniland.maps.load_map_arrays(...)` + "
                "`EnvParams.from_map_arrays(...)` first."
            )
        return self._default_params

    @property
    def name(self) -> str:
        return "CrafterInCogniland-v0"

    @property
    def num_actions(self) -> int:
        return C.NUM_ACTIONS

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(C.NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        V = params.view_size
        return spaces.Dict({
            "minimap": spaces.Box(
                low=0, high=C.NUM_TERRAIN_TILES - 1,
                shape=(V, V), dtype=jnp.int8,
            ),
            "scalars": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=jnp.float32),
        })

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict({})

    # ── Lifecycle ────────────────────────────────────────────────────

    def reset_env(self, key: jax.Array, params: EnvParams) -> tuple[dict, EnvState]:
        state = dyn.reset(key, params)
        obs = build_obs(state, params)
        return obs, state

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[dict, EnvState, jax.Array, jax.Array, dict]:
        action = jnp.asarray(action, dtype=jnp.int32)
        new_state, reward, done, info = dyn.step(key, state, action, params)
        obs = build_obs(new_state, params)
        return obs, new_state, reward, done, info

    # ── Gymnax 0.x compat — gymnax.environments.environment defines
    # ``reset`` / ``step`` that call into ``reset_env`` / ``step_env`` and
    # already handle the auto-reset-on-done pattern. We just need to make
    # sure ``params`` is concrete at JIT time so we declare it static.

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: jax.Array, params: EnvParams | None = None):
        if params is None:
            params = self.default_params
        return self.reset_env(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams | None = None,
    ):
        if params is None:
            params = self.default_params
        return self.step_env(key, state, action, params)
