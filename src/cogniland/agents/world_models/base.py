"""Stub for world model base types — replaced by Agent 4 with full implementation."""

from typing import Any
import chex


@chex.dataclass
class WorldModelParams:
    encoder: Any
    decoder: Any
    dynamics: Any
    reward: Any
    continuation: Any


@chex.dataclass
class WorldModelState:
    pass
