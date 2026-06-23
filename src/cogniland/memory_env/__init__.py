"""memory_env — pixel-observation memory task for DreamerV3 latent analysis.

Exports the env, its config, the oracle policy, the evaluation helper and the
trajectory logger. See :mod:`cogniland.memory_env.env` for the task rationale.
"""
from .env import (
    MemoryEnv,
    MemoryEnvConfig,
    PHASES,
    CUE_TYPES,
    OrientedKey,
    make_memory_env,
    oracle_action,
    evaluate,
    record_trajectory,
)

__all__ = [
    "MemoryEnv",
    "MemoryEnvConfig",
    "PHASES",
    "CUE_TYPES",
    "OrientedKey",
    "make_memory_env",
    "oracle_action",
    "evaluate",
    "record_trajectory",
]
