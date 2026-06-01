"""Pure-JAX (Gymnax-style) port of the bridge_tunnel natural-maps task.

Behaviourally identical to ``cogniland.bridge_tunnel.BridgeTunnelEnv`` on the SAME map
+ action sequence (proven in ``tests/test_bridge_tunnel_jax_parity.py``), but expressed
as a vmappable / jittable Gymnax env so it can drive the pure-JAX DreamerV3
trainer (``scripts/dreamerv3_bridge_tunnel.py``).
"""
from __future__ import annotations

from . import constants
from .env import BridgeTunnelJaxEnv
from .maps import (
    generate_map_dataset,
    load_map_arrays,
    records_to_arrays,
    save_map_arrays,
)
from .state import EnvParams, EnvState

__all__ = [
    "BridgeTunnelJaxEnv",
    "EnvParams",
    "EnvState",
    "constants",
    "generate_map_dataset",
    "records_to_arrays",
    "load_map_arrays",
    "save_map_arrays",
]
