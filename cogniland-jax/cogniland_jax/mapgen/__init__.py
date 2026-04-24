"""Self-contained Cogniland map generation.

Produces the map dataset .pt files consumed by ``cogniland_jax.maps``:

    terrain_idx     int8   [N, 128, 128]    -1 = deadly border
    berry_mask      bool   [N, 128, 128]
    visibility_lut  uint8  [N, 128, 128, 254]   packed Bresenham LUT
    rgb             uint8  [N, 128, 128, 3]     visualisation (not fed to agent)
    heightmap       float32 [N, 128, 128]       post-border, used by visibility
    biomes          list[str] (length N)
    seeds           list[int] (length N)

Run ``python -m cogniland_jax.mapgen`` or the ``scripts/generate_dataset.py``
CLI to build one.
"""

from cogniland_jax.mapgen import terrain, visibility
from cogniland_jax.mapgen.terrain import (
    ALL_BIOMES,
    BIOME_THRESHOLDS,
    CROP_SIZE,
    GEN_SIZE,
    TERRAIN_NAMES,
    apply_biome_mods,
    center_crop,
    colorize_gradient,
    generate_map,
    generate_raw_heightmap,
    paint_deadly_border,
    sample_berry_mask,
)
from cogniland_jax.mapgen.visibility import (
    compute_visibility_luts,
    compute_map_lut,
)

__all__ = [
    "terrain",
    "visibility",
    "ALL_BIOMES",
    "BIOME_THRESHOLDS",
    "CROP_SIZE",
    "GEN_SIZE",
    "TERRAIN_NAMES",
    "apply_biome_mods",
    "center_crop",
    "colorize_gradient",
    "generate_map",
    "generate_raw_heightmap",
    "paint_deadly_border",
    "sample_berry_mask",
    "compute_map_lut",
    "compute_visibility_luts",
]
