"""Pure-JAX Gymnax-style Cogniland environment.

Public API:

    from cogniland_jax import CognilandEnv, EnvParams, EnvState
    from cogniland_jax.maps import load_map_arrays

    arrays = load_map_arrays("data/maps/train.pt", biome_filter=["balanced"])
    params = EnvParams.from_map_arrays(**arrays, difficulty=0)   # easy
    env = CognilandEnv(default_params=params)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(key, state, action, params)

Difficulty bands (max Euclidean spawn-to-target distance, no minimum):
    0 (easy)   →  20
    1 (medium) →  50
    2 (hard)   →  ∞
"""

from cogniland_jax import constants
from cogniland_jax.env import CognilandEnv
from cogniland_jax.state import EnvParams, EnvState

__all__ = ["CognilandEnv", "EnvParams", "EnvState", "constants"]
