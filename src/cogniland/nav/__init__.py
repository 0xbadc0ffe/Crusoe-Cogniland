"""Cogniland navigation environment — POMDP with a belief/commitment skill."""

from . import skills, tiles
from .mapgen import MapGenError, MapRecord, generate_map, shortest_path_cost
from .nav_env import BUILD_ACTION, NUM_ACTIONS, CognilandNavEnv
from .renderer import SpriteSheet


def __getattr__(name):  # PEP 562 — lazy import for optional torch wrapper
    if name == "TorchTensorWrapper":
        from .wrappers import TorchTensorWrapper
        return TorchTensorWrapper
    raise AttributeError(f"module 'cogniland.nav' has no attribute {name!r}")


__all__ = [
    "BUILD_ACTION",
    "CognilandNavEnv",
    "MapGenError",
    "MapRecord",
    "NUM_ACTIONS",
    "SpriteSheet",
    "TorchTensorWrapper",
    "generate_map",
    "shortest_path_cost",
    "skills",
    "tiles",
]
