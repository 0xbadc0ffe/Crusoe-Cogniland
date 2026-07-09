"""Pure-JAX (Gymnax-style) MemoryEnv — symbolic egocentric obs, Discrete(3).

Mirrors the structure of ``cogniland.bridge_tunnel.jax`` so it plugs into the
pure-JAX DreamerV3 trainer. The MDP (dynamics/reward/termination/cue-branch-door)
matches ``cogniland.memory_env.env`` (verified by tests/test_memory_env_jax_parity);
the observation is symbolic tile-ids rather than RGB pixels.
"""
from . import constants
from .dynamics import make_state, reset, step
from .env import MemoryJaxEnv
from .maps import build_geometry
from .render import build_obs
from .state import EnvParams, EnvState

__all__ = [
    "MemoryJaxEnv", "EnvParams", "EnvState", "constants",
    "build_geometry", "build_obs", "reset", "step", "make_state",
]
