"""Pure-JAX (Gymnax-style) bridge_tunnel env — both variants in one package.

Behaviourally identical to ``cogniland.bridge_tunnel.BridgeTunnelEnv(variant=...)``
on the same map + action sequence (parity tests). The variant is the static
``EnvParams.commit`` flag.
"""
from __future__ import annotations

from . import constants
from .env import BridgeTunnelJaxEnv, BridgeTunnelCommitJaxEnv
from .maps import (
    generate_map_dataset, load_map_arrays, records_to_arrays, save_map_arrays,
)
from .state import EnvParams, EnvState

__all__ = [
    "BridgeTunnelJaxEnv", "BridgeTunnelCommitJaxEnv", "EnvParams", "EnvState",
    "constants", "generate_map_dataset", "records_to_arrays",
    "load_map_arrays", "save_map_arrays",
]
