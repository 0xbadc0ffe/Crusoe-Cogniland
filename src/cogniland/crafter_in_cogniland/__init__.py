"""crafter_in_cogniland — pure-JAX navigation env with a one-shot build commitment.

A JAX port of ``cogniland.nav.CognilandNavEnv`` designed for end-to-end
``jax.jit`` training under purejaxwm.dreamerv3. The env mechanics
(reward shape, slip, build commit, target reach) are identical to the
PyTorch nav env; the observation is a tile-id minimap + scalars (no
RGB).

Usage
-----
    from cogniland.crafter_in_cogniland import (
        CrafterInCognilandEnv, EnvParams,
        load_map_arrays,
    )

    arrays = load_map_arrays("data/maps/crafter_in_cogniland_train.pkl")
    params = EnvParams.from_map_arrays(**arrays, max_steps=1000, view_size=21)
    env = CrafterInCognilandEnv(default_params=params)

    obs, state = env.reset(jax.random.PRNGKey(0))
    obs, state, r, done, info = env.step(jax.random.PRNGKey(1), state, action=0)
"""
from __future__ import annotations

from . import constants
from .env import CrafterInCognilandEnv
from .maps import (
    generate_map_dataset,
    load_map_arrays,
    save_map_arrays,
)
from .render import build_obs, egocentric_minimap, scalars_obs
from .state import EnvParams, EnvState


__all__ = [
    "constants",
    "CrafterInCognilandEnv",
    "EnvParams",
    "EnvState",
    "generate_map_dataset",
    "load_map_arrays",
    "save_map_arrays",
    "build_obs",
    "egocentric_minimap",
    "scalars_obs",
]
