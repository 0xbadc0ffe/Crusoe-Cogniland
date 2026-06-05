"""DEPRECATED shim → cogniland.bridge_tunnel.jax (variant=bt).

The JAX env is now unified; use cogniland.bridge_tunnel.jax with the static
EnvParams.commit flag. This re-exports it for backward compatibility.
"""
from cogniland.bridge_tunnel.jax import (  # noqa: F401
    constants, EnvParams, EnvState, BridgeTunnelJaxEnv, BridgeTunnelCommitJaxEnv,
    records_to_arrays, load_map_arrays, save_map_arrays,
)
from . import maps as _maps
generate_map_dataset = _maps.generate_map_dataset

__all__ = ["constants", "EnvParams", "EnvState", "BridgeTunnelJaxEnv",
           "BridgeTunnelJaxEnv", "BridgeTunnelCommitJaxEnv",
           "generate_map_dataset", "records_to_arrays",
           "load_map_arrays", "save_map_arrays"]
