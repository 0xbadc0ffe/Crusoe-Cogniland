"""Batched Cogniland environment — pure numpy, no PyTorch/JAX inside.

Runs B parallel games simultaneously. Each game has:
  - An agent with HP, wood, tool, position
  - A 128x128 map with terrain, berries, heightmap
  - Spawn and target positions
  - 8 actions: 4 cardinal moves, forage, craft_raft, craft_rope, craft_shoes

Observations:
  - minimap: int8 [B, 45, 45] — per-cell tile-class id. All salient entities
      live in this single channel: 0=unseen, 1..9=base terrain, 10=berry,
      11=target_yes, 12=target_no, 13=deadly border. Overlays (berry / target)
      override the base terrain at that cell.
  - scalars: float32 [B, 6] — compass, terrain, hp, wood, tool
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import jax
import jax.numpy as jnp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from cogniland.envs.tile_effects import TileEffects, drain_for

# Terrain class names — must match generate_maps.py order
TERRAIN_NAMES = [
    "ocean", "deep_water", "water", "beach", "sandy",
    "grassland", "forest", "rocky", "mountains",
]

# Action definitions
# 0-3: cardinal movement (up, down, left, right)
# 4: forage
# 5-7: craft raft, rope, shoes
NUM_ACTIONS = 8
MOVE_DELTAS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int32)
CRAFT_TOOLS = {5: "raft", 6: "rope", 7: "shoes"}

# Default terrain visibility radii — kept in sync with
# configs/env/cogniland.yaml::env.terrain_vis_radius. All values must be
# <= MINIMAP_RADIUS (22); precomputed LUTs raycast at 22 and the runtime
# ANDs with a per-terrain disk, so raising a radius up to 22 does NOT
# require dataset regeneration.
DEFAULT_TERRAIN_VIS = {
    "ocean": 22, "deep_water": 18, "water": 14,
    "beach": 12, "sandy": 12, "grassland": 12,
    "forest": 10, "rocky": 18, "mountains": 22,
}

# Minimap config
MINIMAP_RADIUS = 22
MINIMAP_DIAMETER = 2 * MINIMAP_RADIUS + 1  # 45

# Height tolerance for occlusion
CLEAR_TOLERANCE = 0.15

# Tile-class enum for the int8 minimap. Exactly one label per cell —
# berry / target / deadly override the base terrain.
TILE_UNSEEN = 0
# terrain classes 1..9 = TERRAIN_NAMES[0..8] + 1
TILE_BERRY = 10
TILE_TARGET_YES = 11
TILE_TARGET_NO = 12
TILE_DEADLY = 13
NUM_TILE_CLASSES = 14


def _load_maps(maps_path: str, biome_filter=None) -> dict[str, np.ndarray]:
    """Load map dataset and return numpy arrays.

    Args:
        maps_path: path to the .pt file.
        biome_filter: optional iterable of biome names to keep (subsets the
            loaded arrays). ``None`` keeps everything.
    """
    data = torch.load(maps_path, map_location="cpu", weights_only=False)
    if "visibility_lut" not in data:
        raise RuntimeError(
            f"Dataset at {maps_path} lacks 'visibility_lut'. Regenerate with:\n"
            f"    python scripts/generate_dataset.py"
        )
    result = {}
    for key in ("heightmap", "terrain_idx", "berry_mask", "visibility_lut"):
        t = data[key]
        if isinstance(t, torch.Tensor):
            result[key] = t.numpy()
        else:
            result[key] = np.array(t)
    # Optional RGB (kept for trajectory viz only; not consumed by the env obs).
    if "rgb" in data:
        rgb = data["rgb"]
        result["rgb"] = (rgb.numpy() if isinstance(rgb, torch.Tensor) else np.array(rgb)).astype(np.uint8)

    result["heightmap"] = result["heightmap"].astype(np.float32)
    result["terrain_idx"] = result["terrain_idx"].astype(np.int8)
    result["berry_mask"] = result["berry_mask"].astype(bool)
    result["visibility_lut"] = result["visibility_lut"].astype(np.uint8)

    biomes = data.get("biomes", None)
    if biomes is None:
        biomes = ["unknown"] * result["terrain_idx"].shape[0]
    result["biomes"] = np.array([str(b) for b in biomes], dtype=object)

    if biome_filter is not None:
        allowed = set(str(b) for b in biome_filter)
        mask = np.array([b in allowed for b in result["biomes"]], dtype=bool)
        if not mask.any():
            raise ValueError(
                f"biome_filter {sorted(allowed)} matched 0 maps in {maps_path} "
                f"(available biomes: {sorted(set(result['biomes'].tolist()))})"
            )
        for key in ("heightmap", "terrain_idx", "berry_mask", "visibility_lut", "biomes"):
            result[key] = result[key][mask]
        if "rgb" in result:
            result["rgb"] = result["rgb"][mask]

    return result


TARGET_GAP = 3  # column offset: YES _ _ _ NO on the same row


def _sample_spawn_target_batch(
    terrain_idx: np.ndarray,
    map_indices: np.ndarray,
    rng: np.random.Generator,
    min_manhattan: int | np.ndarray = 0,
    max_manhattan: int | np.ndarray | None = None,
    water_idx: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample spawn and paired (YES, NO) targets for a batch of envs.

    YES is placed on a sampled land cell. NO is ``TARGET_GAP`` columns to the
    right of YES, on the same row. Both targets plus all intervening cells
    must be land. Spawn is sampled on land with Manhattan distance in
    ``[min_manhattan[i], max_manhattan[i]]`` (inclusive) from the targets'
    midpoint. ``max_manhattan=None`` means no upper bound (default).

    Returns: spawn_r, spawn_c, yes_r, yes_c, no_r, no_c — all shape [B].
    """
    B = len(map_indices)
    H, W = terrain_idx.shape[1], terrain_idx.shape[2]
    spawn_r = np.zeros(B, dtype=np.int32)
    spawn_c = np.zeros(B, dtype=np.int32)
    yes_r = np.zeros(B, dtype=np.int32)
    yes_c = np.zeros(B, dtype=np.int32)
    no_r = np.zeros(B, dtype=np.int32)
    no_c = np.zeros(B, dtype=np.int32)

    # Broadcast scalar distance to per-env if needed
    if isinstance(min_manhattan, (int, float, np.integer)):
        min_manhattan_arr = np.full(B, int(min_manhattan), dtype=np.int32)
    else:
        min_manhattan_arr = np.asarray(min_manhattan, dtype=np.int32)
    if max_manhattan is None:
        max_manhattan_arr = np.full(B, H + W, dtype=np.int32)
    elif isinstance(max_manhattan, (int, float, np.integer)):
        max_manhattan_arr = np.full(B, int(max_manhattan), dtype=np.int32)
    else:
        max_manhattan_arr = np.asarray(max_manhattan, dtype=np.int32)

    for i in range(B):
        tidx = terrain_idx[map_indices[i]]
        min_m = int(min_manhattan_arr[i])
        max_m = int(max_manhattan_arr[i])
        # Valid YES candidates: land, and column + TARGET_GAP in-bounds and all
        # cells from c to c+TARGET_GAP on the same row are land.
        land_mask = tidx > water_idx
        valid_yes = np.zeros_like(land_mask)
        if W > TARGET_GAP:
            # Rolling AND over (c, c+1, ..., c+TARGET_GAP)
            acc = land_mask[:, :W - TARGET_GAP].copy()
            for k in range(1, TARGET_GAP + 1):
                acc &= land_mask[:, k:W - TARGET_GAP + k]
            valid_yes[:, :W - TARGET_GAP] = acc
        yes_candidates = np.argwhere(valid_yes)
        all_land = np.argwhere(land_mask)

        if len(yes_candidates) == 0 or len(all_land) == 0:
            # Pathological map — place everything mid.
            mid = H // 2
            spawn_r[i] = spawn_c[i] = mid
            yes_r[i] = no_r[i] = mid
            yes_c[i] = max(0, mid - TARGET_GAP // 2)
            no_c[i] = min(W - 1, yes_c[i] + TARGET_GAP)
            continue

        placed = False
        for _ in range(500):
            yi = rng.integers(len(yes_candidates))
            yr, yc = int(yes_candidates[yi, 0]), int(yes_candidates[yi, 1])
            nr, nc = yr, yc + TARGET_GAP
            mid_r, mid_c = yr, yc + TARGET_GAP // 2
            si = rng.integers(len(all_land))
            sr, sc = int(all_land[si, 0]), int(all_land[si, 1])
            dm = abs(sr - mid_r) + abs(sc - mid_c)
            if min_m <= dm <= max_m:
                spawn_r[i], spawn_c[i] = sr, sc
                yes_r[i], yes_c[i] = yr, yc
                no_r[i], no_c[i] = nr, nc
                placed = True
                break

        if not placed:
            # Fallback: first valid YES + far-away spawn.
            yr, yc = int(yes_candidates[0, 0]), int(yes_candidates[0, 1])
            yes_r[i], yes_c[i] = yr, yc
            no_r[i], no_c[i] = yr, yc + TARGET_GAP
            spawn_r[i], spawn_c[i] = int(all_land[-1, 0]), int(all_land[-1, 1])

    return spawn_r, spawn_c, yes_r, yes_c, no_r, no_c


def _compute_tile_idx_batch(
    terrain_idx: np.ndarray,
    berry_mask: np.ndarray,
    map_idx: np.ndarray,
    pos_r: np.ndarray,
    pos_c: np.ndarray,
    yes_r: np.ndarray,
    yes_c: np.ndarray,
    no_r: np.ndarray,
    no_c: np.ndarray,
    vis_per_terrain: np.ndarray,
    vis_lut_packed: np.ndarray | None,
    disk_stack: np.ndarray | None,
    occlude: bool = True,
) -> np.ndarray:
    """Compute the per-cell single-channel minimap.

    Priority when multiple entities occupy one cell:
        TARGET_YES > TARGET_NO > BERRY > DEADLY > terrain > UNSEEN.

    Returns
    -------
    tile_idx : int8 [B, 45, 45]
        0 = UNSEEN/OOB, 1..9 = base terrain, 10 = TILE_BERRY,
        11 = TILE_TARGET_YES, 12 = TILE_TARGET_NO, 13 = TILE_DEADLY.
    """
    B = len(pos_r)
    R = MINIMAP_RADIUS
    D = MINIMAP_DIAMETER
    H, W = terrain_idx.shape[1], terrain_idx.shape[2]

    pos_r_c = np.clip(pos_r, 0, H - 1)
    pos_c_c = np.clip(pos_c, 0, W - 1)
    t_idx_here = terrain_idx[map_idx, pos_r_c, pos_c_c]
    t_idx_here = np.clip(t_idx_here, 0, len(vis_per_terrain) - 1).astype(np.int32)
    vis_r_b = vis_per_terrain[t_idx_here]

    if occlude and vis_lut_packed is not None and disk_stack is not None:
        packed = vis_lut_packed[map_idx, pos_r_c, pos_c_c]
        full = np.unpackbits(packed, axis=1, bitorder="little")
        full = full[:, : D * D].reshape(B, D, D).astype(bool)
        vis_masks = full & disk_stack[vis_r_b]
    else:
        yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
        dist_sq = yy * yy + xx * xx
        vis_masks = dist_sq[None] <= (vis_r_b[:, None, None] ** 2)

    di = np.arange(-R, R + 1, dtype=pos_r.dtype)
    rows = pos_r[:, None, None] + di[None, :, None]
    cols = pos_c[:, None, None] + di[None, None, :]
    rows_b = np.broadcast_to(rows, (B, D, D))
    cols_b = np.broadcast_to(cols, (B, D, D))
    in_bounds = (rows_b >= 0) & (rows_b < H) & (cols_b >= 0) & (cols_b < W)
    rows_cl = np.clip(rows_b, 0, H - 1)
    cols_cl = np.clip(cols_b, 0, W - 1)
    mi_b = np.broadcast_to(map_idx[:, None, None], (B, D, D))

    t_raw = terrain_idx[mi_b, rows_cl, cols_cl]             # [B, D, D] int8, -1=deadly
    b_raw = berry_mask[mi_b, rows_cl, cols_cl]              # [B, D, D] bool
    valid = vis_masks & in_bounds                           # [B, D, D]

    # Base: unseen -> 0, terrain -> 1..9, deadly -> 13.
    base = (t_raw.astype(np.int16) + 1)                     # -1 -> 0, others shifted
    base = np.where(valid, base, 0).astype(np.int16)
    deadly = valid & (t_raw == -1)
    base = np.where(deadly, TILE_DEADLY, base)

    # Berry override: visible berry tiles (excluding deadly).
    is_berry_cell = valid & b_raw & ~deadly
    base = np.where(is_berry_cell, TILE_BERRY, base)

    # Target override: write NO first, YES overrides if they collide
    # (shouldn't in practice — targets sit TARGET_GAP apart on land).
    b_idx = np.arange(B)
    for tr, tc, tile_val in (
        (no_r, no_c, TILE_TARGET_NO),
        (yes_r, yes_c, TILE_TARGET_YES),
    ):
        ty = tr - pos_r + R
        tx = tc - pos_c + R
        ty_c = np.clip(ty, 0, D - 1)
        tx_c = np.clip(tx, 0, D - 1)
        in_patch = (ty >= 0) & (ty < D) & (tx >= 0) & (tx < D)
        visible = in_patch & vis_masks[b_idx, ty_c, tx_c]
        if visible.any():
            env_idx = np.where(visible)[0]
            base[env_idx, ty[env_idx], tx[env_idx]] = tile_val

    return base.astype(np.int8)


@jax.jit
def _compute_tile_idx_jax(
    terrain_idx_jax: jnp.ndarray,
    berry_mask_jax: jnp.ndarray,
    vis_lut_packed_jax: jnp.ndarray,
    disk_stack_jax: jnp.ndarray,
    vis_per_terrain_jax: jnp.ndarray,
    map_idx: jnp.ndarray,
    pos_r: jnp.ndarray,
    pos_c: jnp.ndarray,
    yes_r: jnp.ndarray,
    yes_c: jnp.ndarray,
    no_r: jnp.ndarray,
    no_c: jnp.ndarray,
) -> jnp.ndarray:
    """GPU port of ``_compute_tile_idx_batch`` (occlusion LUT fast path).

    Returns ``tile_idx`` — see the numpy version for semantics.
    """
    B = pos_r.shape[0]
    R = MINIMAP_RADIUS
    D = MINIMAP_DIAMETER
    H = terrain_idx_jax.shape[1]
    W = terrain_idx_jax.shape[2]

    pos_r_c = jnp.clip(pos_r, 0, H - 1)
    pos_c_c = jnp.clip(pos_c, 0, W - 1)
    t_idx_here = terrain_idx_jax[map_idx, pos_r_c, pos_c_c]
    t_idx_here = jnp.clip(t_idx_here, 0, vis_per_terrain_jax.shape[0] - 1).astype(jnp.int32)
    vis_r_b = vis_per_terrain_jax[t_idx_here]

    packed = vis_lut_packed_jax[map_idx, pos_r_c, pos_c_c]
    full = jnp.unpackbits(packed, axis=1, bitorder="little")
    full = full[:, : D * D].reshape(B, D, D).astype(jnp.bool_)
    vis_masks = full & disk_stack_jax[vis_r_b]

    di = jnp.arange(-R, R + 1, dtype=pos_r.dtype)
    rows = pos_r[:, None, None] + di[None, :, None]
    cols = pos_c[:, None, None] + di[None, None, :]
    rows_b = jnp.broadcast_to(rows, (B, D, D))
    cols_b = jnp.broadcast_to(cols, (B, D, D))
    in_bounds = (rows_b >= 0) & (rows_b < H) & (cols_b >= 0) & (cols_b < W)
    rows_cl = jnp.clip(rows_b, 0, H - 1)
    cols_cl = jnp.clip(cols_b, 0, W - 1)
    mi_b = jnp.broadcast_to(map_idx[:, None, None], (B, D, D))

    t_raw = terrain_idx_jax[mi_b, rows_cl, cols_cl]
    b_raw = berry_mask_jax[mi_b, rows_cl, cols_cl]
    valid = vis_masks & in_bounds

    # Base: unseen -> 0, terrain -> 1..9, deadly -> 13.
    base = (t_raw.astype(jnp.int16) + 1)
    base = jnp.where(valid, base, 0)
    deadly = valid & (t_raw == -1)
    base = jnp.where(deadly, jnp.int16(TILE_DEADLY), base)

    # Berry override (excluding deadly).
    is_berry_cell = valid & b_raw & ~deadly
    base = jnp.where(is_berry_cell, jnp.int16(TILE_BERRY), base)

    # Target override: NO first, YES overrides on collision.
    b_idx = jnp.arange(B)
    for tr, tc, tile_val in (
        (no_r, no_c, TILE_TARGET_NO),
        (yes_r, yes_c, TILE_TARGET_YES),
    ):
        ty = tr - pos_r + R
        tx = tc - pos_c + R
        ty_c = jnp.clip(ty, 0, D - 1)
        tx_c = jnp.clip(tx, 0, D - 1)
        in_patch = (ty >= 0) & (ty < D) & (tx >= 0) & (tx < D)
        visible = in_patch & vis_masks[b_idx, ty_c, tx_c]
        prev = base[b_idx, ty_c, tx_c]
        new_vals = jnp.where(visible, jnp.int16(tile_val), prev)
        base = base.at[b_idx, ty_c, tx_c].set(new_vals)

    return base.astype(jnp.int8)


def _compute_occlusion_mask(
    heightmap: np.ndarray,
    center_r: int,
    center_c: int,
    vis_radius: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Compute visibility mask with Bresenham raycasting.

    Returns: bool [D, D] where True = visible.
    """
    R = MINIMAP_RADIUS
    D = MINIMAP_DIAMETER
    visible = np.zeros((D, D), dtype=bool)
    visible[R, R] = True

    center_h = 0.0
    if 0 <= center_r < H and 0 <= center_c < W:
        center_h = float(heightmap[center_r, center_c])

    # Cast rays to perimeter cells
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
    """Cast a single Bresenham ray from center to (end_y, end_x)."""
    y0, x0 = R, R
    y1, x1 = end_y, end_x
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    blocked = False

    # Skip center
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


def _build_circular_masks(max_radius: int) -> dict[int, np.ndarray]:
    """Return ``{r: [D, D] bool disk of radius r}`` for r in 1..max_radius."""
    R = MINIMAP_RADIUS
    D = MINIMAP_DIAMETER
    yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
    dist_sq = yy * yy + xx * xx
    out: dict[int, np.ndarray] = {}
    for r in range(1, max_radius + 1):
        out[r] = dist_sq <= r * r
    return out


def _build_cost_graph(
    terrain_idx: np.ndarray,
    berry_mask: np.ndarray,
    hp_drain_arr: np.ndarray,
) -> csr_matrix:
    """4-connected HP-drain graph for a single 128x128 map.

    Edge cost entering a cell = hp_drain[terrain], or 0 for berry tiles.
    Deadly cells (terrain_idx == -1) are disconnected.
    The returned CSR is asymmetric — edge (u -> v) has cost = cost of entering v.
    """
    H, W = terrain_idx.shape
    cell_cost = np.full((H, W), np.inf, dtype=np.float64)
    valid = terrain_idx >= 0
    cell_cost[valid] = hp_drain_arr[terrain_idx[valid]]
    cell_cost[berry_mask & valid] = 0.0

    r_h, c_h = np.mgrid[0:H, 0:W - 1]
    src_h = (r_h * W + c_h).ravel()
    dst_h = (r_h * W + c_h + 1).ravel()
    r_v, c_v = np.mgrid[0:H - 1, 0:W]
    src_v = (r_v * W + c_v).ravel()
    dst_v = ((r_v + 1) * W + c_v).ravel()

    all_src = np.concatenate([src_h, dst_h, src_v, dst_v])
    all_dst = np.concatenate([dst_h, src_h, dst_v, src_v])
    all_cost = np.concatenate([
        cell_cost[r_h, c_h + 1].ravel(),
        cell_cost[r_h, c_h].ravel(),
        cell_cost[r_v + 1, c_v].ravel(),
        cell_cost[r_v, c_v].ravel(),
    ])
    finite = np.isfinite(all_cost)
    return csr_matrix(
        (all_cost[finite], (all_src[finite], all_dst[finite])),
        shape=(H * W, H * W),
    )


class CognilandEnv:
    """Batched Cogniland environment using pure numpy arrays."""

    def __init__(self, config: Any, maps_path: str, num_envs: int):
        self._config = config
        self._num_envs = num_envs

        env_cfg_for_filter = config.env if hasattr(config, "env") else config.get("env", {})
        biome_filter = None
        if hasattr(env_cfg_for_filter, "biome_filter"):
            bf = env_cfg_for_filter.biome_filter
            biome_filter = list(bf) if bf is not None else None
        elif isinstance(env_cfg_for_filter, dict):
            biome_filter = env_cfg_for_filter.get("biome_filter", None)

        maps = _load_maps(maps_path, biome_filter=biome_filter)
        # Keep rgb (if present) only for trajectory viz; env obs is tile-idx.
        self._rgb = maps.get("rgb", None)
        self._heightmap = maps["heightmap"]            # [N, 128, 128]
        self._terrain_idx = maps["terrain_idx"]        # [N, 128, 128]
        self._berry_mask = maps["berry_mask"]          # [N, 128, 128]
        self._vis_lut_packed = maps["visibility_lut"]  # [N, 128, 128, 254] uint8
        self._biomes = maps["biomes"]                  # object array [N] of biome names
        self._num_maps = self._terrain_idx.shape[0]
        self._map_size = self._terrain_idx.shape[1]

        # Precomputed circular disks, keyed by vis_radius. Used to AND with
        # the unpacked occlusion mask to restrict to the agent's current
        # terrain's vis radius (the LUT itself is at max radius).
        self._circular_masks = _build_circular_masks(MINIMAP_RADIUS)

        # Tile effects
        self._effects = TileEffects()

        # Config params
        env_cfg = config.env if hasattr(config, "env") else config.get("env", {})
        if hasattr(env_cfg, "max_steps"):
            self._max_steps = env_cfg.max_steps
        elif isinstance(env_cfg, dict):
            self._max_steps = env_cfg.get("max_steps", 1000)
        else:
            self._max_steps = 1000

        if hasattr(env_cfg, "min_spawn_target_manhattan"):
            self._min_manhattan = env_cfg.min_spawn_target_manhattan
        elif isinstance(env_cfg, dict):
            self._min_manhattan = env_cfg.get("min_spawn_target_manhattan", 60)
        else:
            self._min_manhattan = 60

        # Optional curriculum on the spawn-target distance.
        #
        # Two forms are supported, in precedence order:
        #
        # 1. ``spawn_distance_schedule: {start: [lo, hi], end: [lo, hi],
        #    anneal_frames: N}`` — the trainer calls
        #    ``set_spawn_distance_range(lo, hi)`` every segment, interpolating
        #    linearly between ``start`` and ``end`` over ``anneal_frames``
        #    total training frames, then clamping at ``end`` thereafter. Use
        #    this to train on easy spawns first and widen to the full range
        #    over time.
        #
        # 2. ``spawn_distance_range: [lo, hi]`` — static band, per-episode
        #    uniform sampling of the minimum spawn-target Manhattan distance
        #    from ``[lo, hi]``. Overrides ``min_spawn_target_manhattan``.
        #
        # With neither set, a scalar ``min_spawn_target_manhattan`` is used
        # (no band — spawn is ``>= min_manhattan``, no upper cap).
        raw_range = None
        if hasattr(env_cfg, "spawn_distance_range"):
            raw_range = env_cfg.spawn_distance_range
        elif isinstance(env_cfg, dict):
            raw_range = env_cfg.get("spawn_distance_range", None)
        if raw_range is None:
            self._distance_range = None
        else:
            lo, hi = int(raw_range[0]), int(raw_range[1])
            if hi < lo:
                raise ValueError(f"spawn_distance_range hi({hi}) < lo({lo})")
            self._distance_range = (lo, hi)

        # Schedule parse. ``start``/``end`` are [lo, hi] pairs; when present
        # the trainer calls ``set_spawn_distance_range`` before each segment
        # and the schedule overrides any static ``spawn_distance_range``.
        raw_sched = None
        if hasattr(env_cfg, "spawn_distance_schedule"):
            raw_sched = env_cfg.spawn_distance_schedule
        elif isinstance(env_cfg, dict):
            raw_sched = env_cfg.get("spawn_distance_schedule", None)
        if raw_sched is None:
            self._distance_schedule = None
        else:
            start = tuple(int(x) for x in raw_sched["start"])
            end = tuple(int(x) for x in raw_sched["end"])
            anneal = int(raw_sched["anneal_frames"])
            if anneal <= 0:
                raise ValueError("spawn_distance_schedule.anneal_frames must be positive")
            self._distance_schedule = {"start": start, "end": end, "anneal_frames": anneal}
            # Initialise the live band at the schedule's start so any reset
            # before the trainer calls ``set_spawn_distance_range`` uses the
            # easy curriculum setting rather than a stale prior.
            self._distance_range = start

        # Terrain vis radius
        if hasattr(env_cfg, "terrain_vis_radius"):
            tvr = env_cfg.terrain_vis_radius
            if hasattr(tvr, "__iter__") and not isinstance(tvr, dict):
                # OmegaConf DictConfig
                self._terrain_vis_radius = dict(tvr)
            else:
                self._terrain_vis_radius = dict(tvr) if isinstance(tvr, dict) else DEFAULT_TERRAIN_VIS.copy()
        elif isinstance(env_cfg, dict) and "terrain_vis_radius" in env_cfg:
            self._terrain_vis_radius = env_cfg["terrain_vis_radius"]
        else:
            self._terrain_vis_radius = DEFAULT_TERRAIN_VIS.copy()

        # Occlusion setting
        if hasattr(env_cfg, "occlude"):
            self._occlude = bool(env_cfg.occlude)
        elif isinstance(env_cfg, dict):
            self._occlude = env_cfg.get("occlude", True)
        else:
            self._occlude = True

        # Vectorised minimap helpers — precomputed once so _compute_tile_idx_batch
        # is a single-shot fancy-indexing call per step.
        self._vis_per_terrain = np.array(
            [self._terrain_vis_radius.get(name, 7) for name in TERRAIN_NAMES],
            dtype=np.int32,
        )
        max_r = max(self._circular_masks.keys())
        stack = np.zeros((max_r + 1, MINIMAP_DIAMETER, MINIMAP_DIAMETER), dtype=bool)
        for r, m in self._circular_masks.items():
            stack[r] = m
        # r=0 should never be selected, but keep it an all-false disk for safety.
        self._disk_stack = stack

        # RNG
        seed = config.seed if hasattr(config, "seed") else config.get("seed", 42)
        self._rng = np.random.default_rng(seed)

        # Map assignment counter
        self._map_counter = 0

        # When False, step() does not auto-reset finished envs — used by the
        # trajectory logger so pos_r/pos_c retain the final episode position.
        self._auto_reset_enabled = True

        # State arrays — allocated in reset()
        self.pos_r: np.ndarray | None = None
        self.pos_c: np.ndarray | None = None
        self.hp: np.ndarray | None = None
        self.wood: np.ndarray | None = None
        self.tool: np.ndarray | None = None  # 0=none, 1=raft, 2=rope, 3=shoes
        self.consec_grass: np.ndarray | None = None
        self.steps: np.ndarray | None = None
        self.map_idx: np.ndarray | None = None
        self.spawn_r: np.ndarray | None = None
        self.spawn_c: np.ndarray | None = None
        # Paired targets: YES is the left of two cells on the same row; NO is
        # TARGET_GAP columns to the right of YES.
        self.yes_r: np.ndarray | None = None
        self.yes_c: np.ndarray | None = None
        self.no_r: np.ndarray | None = None
        self.no_c: np.ndarray | None = None
        # Midpoint cell (used for compass + PBRS shaping).
        self.mid_r: np.ndarray | None = None
        self.mid_c: np.ndarray | None = None
        # Which tool (1=raft, 2=rope, 3=shoes) was newly crafted on the current
        # step, 0 otherwise. Consumed by tasks.py for craft-bonus dispatch.
        self.crafted_this_step: np.ndarray | None = None
        self.done: np.ndarray | None = None

        # Per-episode cost-to-go map (Dijkstra from target), one per env
        self.ctg: np.ndarray | None = None  # [B, H, W] float32
        self.ctg_spawn: np.ndarray | None = None  # [B] float32 — ctg at spawn

        # Episode tracking
        self._episode_returns: np.ndarray | None = None
        self._episode_lengths: np.ndarray | None = None

        # Drain lookup arrays for vectorized computation
        self._hp_drain_arr = np.array(
            [self._effects.hp_drain.get(name, 1) for name in TERRAIN_NAMES],
            dtype=np.float32,
        )

        # Full drain LUT: [terrain, tool_id, shoes_active] -> float32 drain.
        # Used by the vectorised step() to replace per-env drain_for() calls.
        # Priority follows drain_for(): raft > rope > shoes > default. The last
        # axis is 1 only when tool==shoes AND terrain==grassland AND the
        # post-step consec_grass counter is >= shoes_k.
        fx = self._effects
        lut = np.zeros((len(TERRAIN_NAMES), 4, 2), dtype=np.float32)
        for ti, name in enumerate(TERRAIN_NAMES):
            base = float(fx.hp_drain.get(name, 1))
            for tool_id in range(4):
                for shoes_active in (0, 1):
                    if tool_id == 1 and name in fx.raft_drain:
                        d = float(fx.raft_drain[name])
                    elif tool_id == 2 and name in fx.rope_drain:
                        d = float(fx.rope_drain[name])
                    elif (tool_id == 3 and name == "grassland"
                          and shoes_active == 1):
                        d = float(fx.shoes_drain_grassland)
                    else:
                        d = base
                    lut[ti, tool_id, shoes_active] = d
        self._drain_lut = lut
        self._grass_idx = TERRAIN_NAMES.index("grassland")
        self._forest_idx = TERRAIN_NAMES.index("forest")

        # Cache of per-map HP-drain graphs (built lazily on first use)
        self._graph_cache: dict[int, csr_matrix] = {}

        # GPU-resident copies of the map arrays used by the jitted minimap
        # kernel. Uploaded once here so every call reuses them (closure over
        # device arrays inside the jit cache). The non-occlusion path stays
        # on CPU.
        if self._occlude:
            self._berry_mask_jax = jnp.asarray(self._berry_mask)
            self._vis_lut_packed_jax = jnp.asarray(self._vis_lut_packed)
            self._disk_stack_jax = jnp.asarray(self._disk_stack)
            self._vis_per_terrain_jax = jnp.asarray(self._vis_per_terrain)
            self._terrain_idx_jax = jnp.asarray(self._terrain_idx)
        else:
            self._berry_mask_jax = None
            self._vis_lut_packed_jax = None
            self._disk_stack_jax = None
            self._vis_per_terrain_jax = None
            self._terrain_idx_jax = None

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def action_space(self) -> int:
        return NUM_ACTIONS

    def observation_space(self) -> dict:
        return {
            "minimap": (MINIMAP_DIAMETER, MINIMAP_DIAMETER),  # int8, 0..13
            "scalars": (6,),
        }

    def _assign_maps(self, count: int) -> np.ndarray:
        """Assign map indices cycling through the pool."""
        indices = np.arange(count) + self._map_counter
        self._map_counter += count
        return (indices % self._num_maps).astype(np.int32)

    def _sample_distance_constraint(self, count: int):
        """Return (min_m, max_m) per-env distance constraints for a new batch.

        With ``spawn_distance_range: [lo, hi]`` set, each env samples a
        uniform integer ``d ∈ [lo, hi]`` and uses it as BOTH min and max
        (with ±5 tolerance), giving a roughly uniform distribution of spawn
        distances across the curriculum. Otherwise returns the scalar
        ``min_spawn_target_manhattan`` and no max.
        """
        if self._distance_range is None:
            return int(self._min_manhattan), None
        lo, hi = self._distance_range
        d = self._rng.integers(lo, hi + 1, size=count, dtype=np.int32)
        tol = 5
        return (
            np.maximum(d - tol, 0).astype(np.int32),
            (d + tol).astype(np.int32),
        )

    def set_spawn_distance_range(self, lo: int, hi: int) -> None:
        """Update the live spawn-distance band. Used by the trainer's
        curriculum schedule (see Trainer._apply_spawn_distance_schedule)."""
        lo, hi = int(lo), int(hi)
        if hi < lo:
            raise ValueError(f"spawn_distance_range hi({hi}) < lo({lo})")
        self._distance_range = (lo, hi)

    @property
    def spawn_distance_schedule(self):
        """Read-only accessor used by the trainer to decide whether to drive
        a curriculum. Returns ``None`` if no schedule was configured."""
        return self._distance_schedule

    def _tool_name(self, tool_id: int) -> str | None:
        return {0: None, 1: "raft", 2: "rope", 3: "shoes"}.get(tool_id, None)

    def _tool_set(self, tool_id: int) -> frozenset:
        name = self._tool_name(tool_id)
        if name is None:
            return frozenset()
        return frozenset({name})

    def _map_graph(self, mi: int) -> csr_matrix:
        """Return the cached HP-drain graph for map ``mi`` (build on first access)."""
        g = self._graph_cache.get(int(mi))
        if g is None:
            g = _build_cost_graph(
                self._terrain_idx[mi], self._berry_mask[mi], self._hp_drain_arr,
            )
            self._graph_cache[int(mi)] = g
        return g

    def _compute_ctg(self, mi: int, tr: int, tc: int) -> np.ndarray:
        """Dijkstra cost-to-go from every cell to (tr, tc) on map ``mi``.

        Uses the no-tool HP drain table (ignores raft/rope/shoes). Returns a
        float32 [H, W] array; unreachable cells are +inf.
        """
        graph = self._map_graph(mi)
        target_flat = int(tr) * self._map_size + int(tc)
        dist = scipy_dijkstra(graph.T, directed=True, indices=target_flat)
        return dist.reshape(self._map_size, self._map_size).astype(np.float32)

    def reset(
        self,
        seed: int | None = None,
        map_indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Reset all environments. Returns observation dict.

        Args:
            seed: optional seed for spawn/target sampling RNG.
            map_indices: optional [B] array of explicit map indices. If None,
                maps are assigned cycling through the pool.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        B = self._num_envs

        if map_indices is not None:
            mi = np.asarray(map_indices, dtype=np.int32).reshape(-1)
            if mi.shape[0] != B:
                raise ValueError(
                    f"map_indices length {mi.shape[0]} != num_envs {B}"
                )
            self.map_idx = mi
        else:
            self.map_idx = self._assign_maps(B)
        min_m, max_m = self._sample_distance_constraint(B)
        (
            self.spawn_r, self.spawn_c,
            self.yes_r, self.yes_c,
            self.no_r, self.no_c,
        ) = _sample_spawn_target_batch(
            self._terrain_idx, self.map_idx, self._rng,
            min_manhattan=min_m, max_manhattan=max_m,
        )
        self.mid_r = self.yes_r.copy()
        self.mid_c = self.yes_c + (TARGET_GAP // 2)
        self.pos_r = self.spawn_r.copy()
        self.pos_c = self.spawn_c.copy()
        self.hp = np.full(B, float(self._effects.init_hp), dtype=np.float32)
        self.wood = np.zeros(B, dtype=np.int32)
        self.tool = np.zeros(B, dtype=np.int32)
        self.consec_grass = np.zeros(B, dtype=np.int32)
        self.steps = np.zeros(B, dtype=np.int32)
        self.done = np.zeros(B, dtype=bool)
        self.crafted_this_step = np.zeros(B, dtype=np.int32)

        # Precompute Dijkstra cost-to-go maps per env (one per episode),
        # measured from the midpoint between YES and NO targets.
        self.ctg = np.empty((B, self._map_size, self._map_size), dtype=np.float32)
        for b in range(B):
            self.ctg[b] = self._compute_ctg(
                int(self.map_idx[b]), int(self.mid_r[b]), int(self.mid_c[b]),
            )
        self.ctg_spawn = self.ctg[np.arange(B), self.spawn_r, self.spawn_c].copy()

        self._episode_returns = np.zeros(B, dtype=np.float32)
        self._episode_lengths = np.zeros(B, dtype=np.int32)

        return self._get_obs()

    def _reset_envs(self, mask: np.ndarray) -> None:
        """Reset specific environments (auto-reset on done)."""
        if not mask.any():
            return
        count = int(mask.sum())
        indices = np.where(mask)[0]

        new_map_idx = self._assign_maps(count)
        self.map_idx[indices] = new_map_idx

        min_m, max_m = self._sample_distance_constraint(count)
        sr, sc, yr, yc, nr, nc = _sample_spawn_target_batch(
            self._terrain_idx, new_map_idx, self._rng,
            min_manhattan=min_m, max_manhattan=max_m,
        )
        self.spawn_r[indices] = sr
        self.spawn_c[indices] = sc
        self.yes_r[indices] = yr
        self.yes_c[indices] = yc
        self.no_r[indices] = nr
        self.no_c[indices] = nc
        self.mid_r[indices] = yr
        self.mid_c[indices] = yc + (TARGET_GAP // 2)
        self.pos_r[indices] = sr
        self.pos_c[indices] = sc

        self.hp[indices] = float(self._effects.init_hp)
        self.wood[indices] = 0
        self.tool[indices] = 0
        self.consec_grass[indices] = 0
        self.steps[indices] = 0
        self.done[indices] = False
        self.crafted_this_step[indices] = 0

        # Recompute cost-to-go (from midpoint) for envs that just reset.
        for j, b in enumerate(indices):
            self.ctg[b] = self._compute_ctg(
                int(new_map_idx[j]), int(yr[j]), int(yc[j] + TARGET_GAP // 2),
            )
            self.ctg_spawn[b] = self.ctg[b, sr[j], sc[j]]

        self._episode_returns[indices] = 0.0
        self._episode_lengths[indices] = 0

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
        """Step all environments.

        Args:
            actions: int array [B], values 0-7

        Returns:
            (obs_dict, rewards, dones, info)
        """
        B = self._num_envs
        actions = np.asarray(actions, dtype=np.int32)

        rewards = np.zeros(B, dtype=np.float32)

        # Snapshot cost-to-go and HP at current position BEFORE the step is
        # applied, so the reward function can compute PBRS progress
        # = ctg_prev - ctg_curr and the HP-delta shaping term (hp_curr - hp_prev).
        ctg_prev = self.ctg[np.arange(B), self.pos_r, self.pos_c].copy()
        hp_prev = self.hp.copy()

        # Reset per-step craft flag (set below when a craft action succeeds).
        self.crafted_this_step.fill(0)

        # Track which envs just finished (for episode return reporting)
        returned_episode = np.zeros(B, dtype=bool)
        returned_episode_returns = np.zeros(B, dtype=np.float32)
        returned_episode_lengths = np.zeros(B, dtype=np.int32)

        # Process each env — vectorize the common path, loop for special cases
        # We process movement, forage, and craft separately

        is_move = actions < 4
        is_forage = actions == 4
        is_craft = actions >= 5

        H = self._map_size
        W = self._map_size
        fx = self._effects
        hp_max = float(fx.hp_max)
        wood_max = int(fx.wood_max)
        berry_heal = float(fx.berry_heal)
        forest_wood = int(fx.forest_wood)
        craft_cost = int(fx.craft_cost)
        shoes_k = int(fx.shoes_k)
        grass_idx = self._grass_idx
        forest_idx = self._forest_idx
        drain_lut = self._drain_lut

        # --- Movement actions (0-3) ---
        move_mask = is_move & ~self.done
        if move_mask.any():
            idx = np.where(move_mask)[0]
            deltas = MOVE_DELTAS[actions[idx]]
            cur_r = self.pos_r[idx]
            cur_c = self.pos_c[idx]
            new_r = cur_r + deltas[:, 0]
            new_c = cur_c + deltas[:, 1]

            in_bounds = (new_r >= 0) & (new_r < H) & (new_c >= 0) & (new_c < W)
            new_r_safe = np.where(in_bounds, new_r, cur_r)
            new_c_safe = np.where(in_bounds, new_c, cur_c)

            mi = self.map_idx[idx]
            t_idx = self._terrain_idx[mi, new_r_safe, new_c_safe]
            deadly = in_bounds & (t_idx < 0)
            valid_move = in_bounds & ~deadly                        # entered walkable cell
            t_idx_safe = np.where(valid_move, t_idx, 0).astype(np.int32)

            is_berry = valid_move & self._berry_mask[mi, new_r_safe, new_c_safe]
            terrain_is_grass = valid_move & (t_idx_safe == grass_idx) & ~is_berry

            prev_consec = self.consec_grass[idx]
            new_consec = np.where(terrain_is_grass, prev_consec + 1, 0)
            tool_id = self.tool[idx]
            shoes_active = (tool_id == 3) & terrain_is_grass & (new_consec >= shoes_k)
            drain = drain_lut[t_idx_safe, tool_id, shoes_active.astype(np.int32)]
            drain = np.where(valid_move & ~is_berry, drain, 0.0)

            # Position: update on any in-bounds attempt (deadly moves also land).
            final_r = np.where(in_bounds, new_r, cur_r)
            final_c = np.where(in_bounds, new_c, cur_c)
            self.pos_r[idx] = final_r
            self.pos_c[idx] = final_c

            # HP: apply drain on valid_move; deadly clears to 0; oob leaves unchanged.
            new_hp = self.hp[idx] - drain
            new_hp = np.where(deadly, 0.0, new_hp)
            new_hp = np.where(new_hp < 0, 0.0, new_hp)
            self.hp[idx] = new_hp

            # Consecutive-grass counter only updates on valid moves (old code
            # left it unchanged on both oob and deadly attempts).
            self.consec_grass[idx] = np.where(valid_move, new_consec, prev_consec)

            # Done: deadly tile, hp<=0 after a valid drain, or reached YES/NO.
            hp_dead = valid_move & (new_hp <= 0)
            reached_yes = valid_move & (final_r == self.yes_r[idx]) & (final_c == self.yes_c[idx])
            reached_no  = valid_move & (final_r == self.no_r[idx])  & (final_c == self.no_c[idx])
            new_done = deadly | hp_dead | reached_yes | reached_no
            self.done[idx] = self.done[idx] | new_done

            self.steps[idx] += 1

        # --- Forage action (4) ---
        forage_mask = is_forage & ~self.done
        if forage_mask.any():
            idx = np.where(forage_mask)[0]
            cur_r = self.pos_r[idx]
            cur_c = self.pos_c[idx]
            mi = self.map_idx[idx]
            t_idx = self._terrain_idx[mi, cur_r, cur_c]
            deadly = t_idx < 0
            valid = ~deadly
            t_idx_safe = np.where(valid, t_idx, 0).astype(np.int32)

            is_berry = valid & self._berry_mask[mi, cur_r, cur_c]
            is_forest = valid & ~is_berry & (t_idx_safe == forest_idx)
            terrain_is_grass = valid & ~is_berry & (t_idx_safe == grass_idx)

            # Berry: heal & reset consec_grass; no drain.
            heal = np.where(is_berry, berry_heal, 0.0)
            # Non-berry non-deadly: maybe add wood (forest), then apply drain.
            prev_consec = self.consec_grass[idx]
            new_consec = np.where(terrain_is_grass, prev_consec + 1, 0)
            tool_id = self.tool[idx]
            shoes_active = (tool_id == 3) & terrain_is_grass & (new_consec >= shoes_k)
            drain = drain_lut[t_idx_safe, tool_id, shoes_active.astype(np.int32)]
            drain = np.where(valid & ~is_berry, drain, 0.0)

            new_hp = self.hp[idx] + heal - drain
            new_hp = np.minimum(new_hp, hp_max)
            new_hp = np.where(new_hp < 0, 0.0, new_hp)
            self.hp[idx] = new_hp

            add_wood = np.where(is_forest, forest_wood, 0).astype(np.int32)
            new_wood = np.minimum(self.wood[idx] + add_wood, wood_max)
            self.wood[idx] = new_wood

            # consec_grass: 0 on berry; unchanged on deadly; new_consec otherwise.
            self.consec_grass[idx] = np.where(
                is_berry, 0,
                np.where(deadly, prev_consec, new_consec),
            )

            # Done only if drain killed us (berry can't kill; deadly forage = no-op).
            hp_dead = valid & ~is_berry & (new_hp <= 0)
            self.done[idx] = self.done[idx] | hp_dead
            self.steps[idx] += 1

        # --- Craft actions (5-7) ---
        craft_mask = is_craft & ~self.done
        if craft_mask.any():
            idx = np.where(craft_mask)[0]
            # Action 5,6,7 -> tool_id 1,2,3
            new_tool_id = (actions[idx] - 4).astype(np.int32)
            can_craft = (self.tool[idx] == 0) & (self.wood[idx] >= craft_cost)
            self.wood[idx] = np.where(can_craft, self.wood[idx] - craft_cost, self.wood[idx])
            updated_tool = np.where(can_craft, new_tool_id, self.tool[idx])
            self.tool[idx] = updated_tool
            self.crafted_this_step[idx] = np.where(can_craft, new_tool_id, 0).astype(np.int32)

            # Apply a standing-tile drain using the (possibly newly updated) tool.
            cur_r = self.pos_r[idx]
            cur_c = self.pos_c[idx]
            mi = self.map_idx[idx]
            t_idx = self._terrain_idx[mi, cur_r, cur_c]
            valid = t_idx >= 0
            t_idx_safe = np.where(valid, t_idx, 0).astype(np.int32)
            is_berry = valid & self._berry_mask[mi, cur_r, cur_c]
            terrain_is_grass = valid & ~is_berry & (t_idx_safe == grass_idx)

            prev_consec = self.consec_grass[idx]
            new_consec = np.where(terrain_is_grass, prev_consec + 1, 0)
            shoes_active = (updated_tool == 3) & terrain_is_grass & (new_consec >= shoes_k)
            drain = drain_lut[t_idx_safe, updated_tool, shoes_active.astype(np.int32)]
            # Old code: if valid -> apply drain (incl. berry which sets drain=0 and new_consec=0).
            drain = np.where(valid & ~is_berry, drain, 0.0)
            # On berry the old code also resets consec_grass to 0 (grass mask below handles that).

            new_hp = self.hp[idx] - drain
            new_hp = np.where(new_hp < 0, 0.0, new_hp)
            self.hp[idx] = new_hp

            # consec_grass: 0 on berry or non-grass; prev+1 on grass; unchanged on deadly.
            self.consec_grass[idx] = np.where(
                valid, np.where(is_berry, 0, new_consec), prev_consec
            )

            hp_dead = valid & (new_hp <= 0)
            self.done[idx] = self.done[idx] | hp_dead
            self.steps[idx] += 1

        # --- Check timeout ---
        timeout_mask = (self.steps >= self._max_steps) & ~self.done
        self.done[timeout_mask] = True

        # --- Compute rewards (placeholder — actual reward is in tasks.py) ---
        # The raw env returns 0 reward. Task-specific reward is computed by
        # MultiTaskEnvWrapper via tasks.py.

        # --- Collect episode stats for finished envs ---
        just_done = self.done.copy()
        if just_done.any():
            returned_episode[just_done] = True
            returned_episode_returns[just_done] = self._episode_returns[just_done]
            returned_episode_lengths[just_done] = self.steps[just_done]

        # Cost-to-go at the new position (after the step, before any auto-reset).
        # Agents that ended on a deadly cell or otherwise unreachable index will
        # see +inf here; tasks.py filters these out.
        ctg_curr = self.ctg[np.arange(B), self.pos_r, self.pos_c].copy()

        reached_yes = (
            (self.pos_r == self.yes_r) & (self.pos_c == self.yes_c) & self.done
        )
        reached_no = (
            (self.pos_r == self.no_r) & (self.pos_c == self.no_c) & self.done
        )
        info = {
            "returned_episode_returns": returned_episode_returns,
            "returned_episode_lengths": returned_episode_lengths,
            "returned_episode": returned_episode,
            # Extra info for reward computation
            "reached": reached_yes | reached_no,
            "reached_yes": reached_yes,
            "reached_no": reached_no,
            "alive": self.hp > 0,
            "dist_to_target": np.sqrt(
                (self.pos_r.astype(np.float32) - self.mid_r.astype(np.float32)) ** 2 +
                (self.pos_c.astype(np.float32) - self.mid_c.astype(np.float32)) ** 2
            ),
            "initial_dist": np.sqrt(
                (self.spawn_r.astype(np.float32) - self.mid_r.astype(np.float32)) ** 2 +
                (self.spawn_c.astype(np.float32) - self.mid_c.astype(np.float32)) ** 2
            ),
            # Cost-to-go potentials for PBRS shaping (one-shot Dijkstra from the
            # YES/NO midpoint, computed per episode). ``ctg_spawn`` is the
            # initial potential.
            "ctg_prev": ctg_prev,
            "ctg_curr": ctg_curr,
            "ctg_spawn": self.ctg_spawn.copy(),
            # HP snapshots for the hp-delta shaping term (hp_coef * Δhp).
            "hp_prev": hp_prev,
            "hp_curr": self.hp.copy(),
            # Biome label per env (string) — used by tasks 1-3 to score
            # classification questions. Not exposed in the obs.
            "biome": self._biomes[self.map_idx].copy(),
            # Tool id crafted on this step (0=none, 1=raft, 2=rope, 3=shoes) —
            # used by tasks 4-6 to fire the craft bonus once per episode.
            "crafted": self.crafted_this_step.copy(),
        }

        dones = self.done.copy()

        # Auto-reset done envs (skip when the logger has disabled it)
        if self._auto_reset_enabled:
            self._reset_envs(just_done)

        obs = self._get_obs()

        return obs, rewards, dones, info

    def _get_obs(self) -> dict:
        """Build observation dict for all envs.

        When ``self._occlude`` is True the minimap is computed on GPU via
        ``_compute_tile_idx_jax`` and returned as a jnp array; ``scalars``
        stays as a numpy array. When ``occlude=False`` the numpy fallback runs.
        """
        B = self._num_envs

        if self._occlude:
            minimap = _compute_tile_idx_jax(
                self._terrain_idx_jax,
                self._berry_mask_jax,
                self._vis_lut_packed_jax,
                self._disk_stack_jax,
                self._vis_per_terrain_jax,
                jnp.asarray(self.map_idx),
                jnp.asarray(self.pos_r),
                jnp.asarray(self.pos_c),
                jnp.asarray(self.yes_r),
                jnp.asarray(self.yes_c),
                jnp.asarray(self.no_r),
                jnp.asarray(self.no_c),
            )
        else:
            minimap = _compute_tile_idx_batch(
                self._terrain_idx, self._berry_mask,
                self.map_idx, self.pos_r, self.pos_c,
                self.yes_r, self.yes_c,
                self.no_r, self.no_c,
                self._vis_per_terrain,
                vis_lut_packed=self._vis_lut_packed,
                disk_stack=self._disk_stack,
                occlude=self._occlude,
            )

        # Compute scalars
        scalars = np.zeros((B, 6), dtype=np.float32)

        # Compass: unit vector from agent to the YES/NO midpoint.
        dr = self.mid_r.astype(np.float32) - self.pos_r.astype(np.float32)
        dc = self.mid_c.astype(np.float32) - self.pos_c.astype(np.float32)
        dist = np.sqrt(dr * dr + dc * dc)
        dist = np.maximum(dist, 1e-6)
        scalars[:, 0] = dc / dist  # compass_x (column direction)
        scalars[:, 1] = dr / dist  # compass_y (row direction)

        # Current tile class, normalized to [0, 1]. 10 classes:
        #   0..8 = ocean, deep_water, water, beach, sandy, grassland, forest, rocky, mountains
        #   9    = berry (overlay on forest/beach — overrides the base terrain)
        mi_all = self.map_idx
        t_here = self._terrain_idx[mi_all, self.pos_r, self.pos_c]
        tile_cls = np.maximum(t_here, 0)
        berry_here = self._berry_mask[mi_all, self.pos_r, self.pos_c]
        tile_cls = np.where(berry_here, 9, tile_cls).astype(np.float32)
        scalars[:, 2] = tile_cls / 9.0

        scalars[:, 3] = self.hp / float(self._effects.hp_max)
        scalars[:, 4] = self.wood.astype(np.float32) / float(self._effects.wood_max)
        scalars[:, 5] = self.tool.astype(np.float32) / 3.0

        return {
            "minimap": minimap,
            "scalars": scalars,
        }
