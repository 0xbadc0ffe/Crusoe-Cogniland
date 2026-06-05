"""DEPRECATED shim → cogniland.bridge_tunnel.jax.maps (variant=bt)."""
from cogniland.bridge_tunnel.jax.maps import (  # noqa: F401
    records_to_arrays, load_map_arrays, save_map_arrays, NATURAL_KWARGS,
)
from cogniland.bridge_tunnel.jax import maps as _m


def generate_map_dataset(*a, **k):
    k.setdefault("variant", "bt")
    return _m.generate_map_dataset(*a, **k)
