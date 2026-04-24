"""Load pre-generated map dataset into JAX arrays.

The datasets produced by ``scripts/generate_dataset.py`` are PyTorch `.pt`
files containing ``terrain_idx``, ``berry_mask``, ``visibility_lut``, and
``biomes``. We load them via torch and convert to jnp arrays once; from
then on they live inside ``EnvParams`` as static device tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import jax.numpy as jnp
import numpy as np
import torch

BIOME_NAMES: tuple[str, ...] = ("balanced", "archipelago", "grassland", "highland")


def _biome_to_id(names: Iterable[str]) -> np.ndarray:
    name_to_id = {n: i for i, n in enumerate(BIOME_NAMES)}
    return np.array([name_to_id.get(str(n), -1) for n in names], dtype=np.int32)


def load_map_arrays(
    path: str | Path,
    biome_filter: Optional[Iterable[str]] = None,
) -> dict[str, jnp.ndarray]:
    """Return a dict of jnp arrays ready for ``EnvParams.from_map_arrays``.

    Keys: terrain_idx [N, H, W] int8, berry_mask [N, H, W] bool,
    vis_lut_packed [N, H, W, P] uint8, biome_id [N] int32.
    """
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    terrain_idx = raw["terrain_idx"].numpy().astype(np.int8)
    berry_mask = raw["berry_mask"].numpy().astype(bool)
    vis_lut = raw["visibility_lut"].numpy().astype(np.uint8)
    biomes = list(raw.get("biomes", ["unknown"] * terrain_idx.shape[0]))

    if biome_filter is not None:
        allowed = {str(b) for b in biome_filter}
        keep = np.array([str(b) in allowed for b in biomes], dtype=bool)
        if not keep.any():
            raise ValueError(
                f"biome_filter {sorted(allowed)} matched 0 maps in {path}"
            )
        terrain_idx = terrain_idx[keep]
        berry_mask = berry_mask[keep]
        vis_lut = vis_lut[keep]
        biomes = [b for b, k in zip(biomes, keep) if k]

    return {
        "terrain_idx": jnp.asarray(terrain_idx),
        "berry_mask": jnp.asarray(berry_mask),
        "vis_lut_packed": jnp.asarray(vis_lut),
        "biome_id": jnp.asarray(_biome_to_id(biomes)),
    }
