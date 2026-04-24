"""Convenience API for the map dataset.

Re-exports the two primitives a user typically needs:

    from cogniland_jax.dataset import build_dataset, load_map_arrays

    build_dataset(output_dir="data/maps")                        # write .pt files
    arrays = load_map_arrays("data/maps/train.pt",
                             biome_filter=["balanced"])          # → jnp arrays
"""

from cogniland_jax.mapgen.build import build_dataset
from cogniland_jax.maps import load_map_arrays

__all__ = ["build_dataset", "load_map_arrays"]
