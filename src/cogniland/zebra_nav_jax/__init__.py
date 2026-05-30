"""Pure-JAX (Gymnax-style) port of the zebra_nav natural-maps task.

Behaviourally identical to ``cogniland.zebra_nav.ZebraNavEnv`` on the SAME map
+ action sequence (proven in ``tests/test_zebra_jax_parity.py``), but expressed
as a vmappable / jittable Gymnax env so it can drive the pure-JAX DreamerV3
trainer (``scripts/dreamerv3_zebra_nav.py``).
"""
from __future__ import annotations

from . import constants
from .env import ZebraNavJaxEnv
from .maps import (
    generate_map_dataset,
    load_map_arrays,
    records_to_arrays,
    save_map_arrays,
)
from .state import EnvParams, EnvState

__all__ = [
    "ZebraNavJaxEnv",
    "EnvParams",
    "EnvState",
    "constants",
    "generate_map_dataset",
    "records_to_arrays",
    "load_map_arrays",
    "save_map_arrays",
]
