"""Precompute per-map visibility LUTs used by ``CognilandEnv``.

For every cell ``(r, c)`` of every heightmap we run the same Bresenham
raycasting that ``env._compute_occlusion_mask`` does, at the maximum vis
radius (22 = mountains). The resulting 45x45 bool mask is bit-packed with
``np.packbits`` into 254 bytes. Per map that's ``128 * 128 * 254`` ≈ 4.16 MB;
a 256-map train split is ~1.06 GB.

At runtime ``CognilandEnv`` looks up the packed mask, unpacks it, and ANDs
with a precomputed circular disk of the agent's current terrain vis radius —
this is provably identical to computing the mask at that smaller radius.

Also defines ``make_circular_masks`` so the env can build its per-radius disk
dict without duplicating the logic.
"""

from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Optional

import numpy as np

from cogniland.envs.env import (
    MINIMAP_DIAMETER,
    MINIMAP_RADIUS,
    _compute_occlusion_mask,
)


# Number of bytes per packed [45, 45] bool mask: ceil(2025 / 8) = 254
_PACKED_BYTES = (MINIMAP_DIAMETER * MINIMAP_DIAMETER + 7) // 8


def make_circular_masks(max_radius: int = MINIMAP_RADIUS) -> dict[int, np.ndarray]:
    """Return ``{r: [D, D] bool disk of radius r}`` for r in 1..max_radius.

    The env ANDs the precomputed (radius-22) occlusion mask with one of
    these disks based on the agent's current terrain.
    """
    D = MINIMAP_DIAMETER
    R = MINIMAP_RADIUS
    yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
    dist_sq = yy * yy + xx * xx
    masks: dict[int, np.ndarray] = {}
    for r in range(1, max_radius + 1):
        mask = np.zeros((D, D), dtype=bool)
        mask[dist_sq <= r * r] = True
        masks[r] = mask
    return masks


def compute_map_lut(heightmap: np.ndarray, max_radius: int = MINIMAP_RADIUS) -> np.ndarray:
    """Return ``uint8 [H, W, 254]`` — packed visibility mask per cell.

    Runs ``_compute_occlusion_mask`` at the max vis radius once per position.
    """
    H, W = heightmap.shape
    D = MINIMAP_DIAMETER
    # Work in an uint8 buffer so we can pack once at the end.
    packed = np.zeros((H, W, _PACKED_BYTES), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            mask = _compute_occlusion_mask(heightmap, r, c, max_radius, H, W)
            # packbits over the flattened mask (2025 bits -> 254 bytes, last
            # byte is zero-padded).
            packed[r, c] = np.packbits(mask.ravel(), bitorder="little")
    return packed


def _worker(args: tuple[int, np.ndarray, int]) -> tuple[int, np.ndarray]:
    i, heightmap, max_radius = args
    return i, compute_map_lut(heightmap, max_radius)


def compute_visibility_luts(
    heightmaps: np.ndarray,
    num_workers: Optional[int] = None,
    max_radius: int = MINIMAP_RADIUS,
) -> np.ndarray:
    """Return ``uint8 [N, H, W, 254]`` for ``heightmaps: [N, H, W] float``.

    Parallelises across maps (per-map precompute is ~10 s single-threaded on
    an average CPU core). ``num_workers=None`` picks ``os.cpu_count()``.
    Use ``num_workers=1`` to run inline without spawning a pool.
    """
    N, H, W = heightmaps.shape
    out = np.empty((N, H, W, _PACKED_BYTES), dtype=np.uint8)

    if num_workers == 1 or num_workers == 0:
        for i in range(N):
            out[i] = compute_map_lut(heightmaps[i], max_radius)
            print(f"  visibility LUT: map {i + 1}/{N}", flush=True)
        return out

    nw = num_workers or max(1, os.cpu_count() or 1)
    jobs = [(i, heightmaps[i], max_radius) for i in range(N)]
    with Pool(processes=nw) as pool:
        done = 0
        for i, lut in pool.imap_unordered(_worker, jobs, chunksize=1):
            out[i] = lut
            done += 1
            print(f"  visibility LUT: {done}/{N}", flush=True)
    return out
