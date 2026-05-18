"""Shared utilities. Starts almost empty by design (DESIGN.md Principle 3).

Current contents:
  - `wrappers.py`: Gymnax-style env wrappers (Principle 3 founding contents).
  - `precision.py`: `resolve_dtype` — the bfloat16 compute-dtype contract shared by
    every baseline (`cfg.compute_dtype` → `jnp.dtype`).
"""
from purejaxwm.commons.precision import resolve_dtype
from purejaxwm.commons.wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    GymnaxWrapper,
    LogEnvState,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)

__all__ = [
    "AutoResetEnvWrapper",
    "BatchEnvWrapper",
    "GymnaxWrapper",
    "LogEnvState",
    "LogWrapper",
    "OptimisticResetVecEnvWrapper",
    "resolve_dtype",
]
