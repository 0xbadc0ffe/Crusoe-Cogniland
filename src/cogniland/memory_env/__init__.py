"""memory_env — pixel-observation memory task for DreamerV3 latent analysis.

The MiniGrid-based environment lives in :mod:`cogniland.memory_env.env` and
imports ``minigrid``. Those symbols are loaded **lazily** (PEP 562) so the
pure-JAX subpackage :mod:`cogniland.memory_env.jax` can be imported in a
JAX-only environment that has no ``minigrid`` installed. ``from
cogniland.memory_env import MemoryEnv`` still works — it triggers the lazy load.
"""
from __future__ import annotations

import importlib

_LAZY = {
    name: ".env"
    for name in (
        "MemoryEnv", "MemoryEnvConfig", "PHASES", "CUE_TYPES", "OrientedKey",
        "make_memory_env", "oracle_action", "evaluate", "record_trajectory",
    )
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
