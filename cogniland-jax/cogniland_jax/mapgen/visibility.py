"""Self-contained visibility LUT precompute (Bresenham raycast + bit-pack).

For every cell ``(r, c)`` of every heightmap we cast rays at the maximum
visibility radius (22 — same as the runtime ``constants.MINIMAP_RADIUS``),
record which cells inside the 45×45 patch are visible, and bit-pack the
result. The env then ANDs this LUT with a per-terrain circular disk at
runtime so the same precompute serves every visibility radius.

Mirrors ``scripts/precompute_visibility.py`` from the legacy repo, but
inlines the occlusion implementation so this module has no external
engine dependencies.
"""

from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Optional

import numpy as np

from cogniland_jax.constants import MINIMAP_DIAMETER, MINIMAP_RADIUS

CLEAR_TOLERANCE = 0.15
_PACKED_BYTES = (MINIMAP_DIAMETER * MINIMAP_DIAMETER + 7) // 8  # 254


def _cast_ray(
    visible: np.ndarray,
    heightmap: np.ndarray,
    center_r: int,
    center_c: int,
    center_h: float,
    R: int,
    end_y: int,
    end_x: int,
    vis_radius: int,
    H: int,
    W: int,
) -> None:
    """Cast a single Bresenham ray from centre to (end_y, end_x)."""
    y0, x0 = R, R
    y1, x1 = end_y, end_x
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    blocked = False
    first = True
    cy, cx = y0, x0
    while True:
        if not first:
            dist_sq = (cy - R) ** 2 + (cx - R) ** 2
            if dist_sq > vis_radius * vis_radius:
                break
            wr = center_r + (cy - R)
            wc = center_c + (cx - R)
            if not (0 <= wr < H and 0 <= wc < W):
                break
            if not blocked:
                visible[cy, cx] = True
                cell_h = heightmap[wr, wc]
                if cell_h >= center_h + CLEAR_TOLERANCE:
                    blocked = True
        first = False
        if cx == x1 and cy == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx += sx
        if e2 < dx:
            err += dx
            cy += sy


def compute_occlusion_mask(
    heightmap: np.ndarray,
    center_r: int,
    center_c: int,
    vis_radius: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Height-based Bresenham visibility at ``vis_radius`` around (r, c)."""
    R = MINIMAP_RADIUS
    D = MINIMAP_DIAMETER
    visible = np.zeros((D, D), dtype=bool)
    visible[R, R] = True

    center_h = 0.0
    if 0 <= center_r < H and 0 <= center_c < W:
        center_h = float(heightmap[center_r, center_c])

    perimeter = []
    for i in range(D):
        perimeter.append((0, i))
        perimeter.append((D - 1, i))
    for i in range(1, D - 1):
        perimeter.append((i, 0))
        perimeter.append((i, D - 1))

    for py, px in perimeter:
        _cast_ray(visible, heightmap, center_r, center_c, center_h,
                  R, py, px, vis_radius, H, W)
    return visible


def compute_map_lut(
    heightmap: np.ndarray,
    max_radius: int = MINIMAP_RADIUS,
) -> np.ndarray:
    """Return ``uint8 [H, W, 254]`` — bit-packed visibility mask per cell."""
    H, W = heightmap.shape
    packed = np.zeros((H, W, _PACKED_BYTES), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            mask = compute_occlusion_mask(heightmap, r, c, max_radius, H, W)
            packed[r, c] = np.packbits(mask.ravel(), bitorder="little")
    return packed


def _worker(args):
    i, heightmap, max_radius = args
    return i, compute_map_lut(heightmap, max_radius)


def compute_visibility_luts(
    heightmaps: np.ndarray,
    num_workers: Optional[int] = None,
    max_radius: int = MINIMAP_RADIUS,
) -> np.ndarray:
    """Return ``uint8 [N, H, W, 254]`` for ``heightmaps: [N, H, W] float``.

    Parallelises across maps. ``num_workers=None`` picks ``os.cpu_count()``.
    ``num_workers<=1`` runs inline without spawning a pool (useful for
    debugging a single map).
    """
    N, H, W = heightmaps.shape
    out = np.empty((N, H, W, _PACKED_BYTES), dtype=np.uint8)

    if num_workers is not None and num_workers <= 1:
        for i in range(N):
            out[i] = compute_map_lut(heightmaps[i], max_radius)
            print(f"  visibility LUT: {i + 1}/{N}", flush=True)
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


__all__ = [
    "compute_occlusion_mask",
    "compute_map_lut",
    "compute_visibility_luts",
]
