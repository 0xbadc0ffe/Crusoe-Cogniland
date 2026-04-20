"""Batched Cogniland environment — pure numpy, no PyTorch/JAX inside.

Runs B parallel games simultaneously. Each game has:
  - An agent with HP, wood, tool, position
  - A 128x128 map with terrain, berries, heightmap, RGB
  - Spawn and target positions
  - 8 actions: 4 cardinal moves, forage, craft_raft, craft_rope, craft_shoes

Observations:
  - minimap: float32 [B, 5, 45, 45] — 3 RGB channels + visibility mask + target indicator
  - scalars: float32 [B, 6] — compass, terrain, hp, wood, tool
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
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

# Default terrain visibility radii
DEFAULT_TERRAIN_VIS = {
    "ocean": 16, "deep_water": 12, "water": 10,
    "beach": 7, "sandy": 7, "grassland": 7,
    "forest": 5, "rocky": 10, "mountains": 22,
}

# Minimap config
MINIMAP_RADIUS = 22
MINIMAP_DIAMETER = 2 * MINIMAP_RADIUS + 1  # 45

# Height tolerance for occlusion
CLEAR_TOLERANCE = 0.15


def _load_maps(maps_path: str) -> dict[str, np.ndarray]:
    """Load map dataset and convert everything to numpy."""
    data = torch.load(maps_path, map_location="cpu", weights_only=False)
    if "visibility_lut" not in data:
        raise RuntimeError(
            f"Dataset at {maps_path} lacks 'visibility_lut'. Regenerate with:\n"
            f"    python scripts/generate_dataset.py"
        )
    result = {}
    for key in ("rgb", "heightmap", "terrain_idx", "berry_mask", "visibility_lut"):
        t = data[key]
        if isinstance(t, torch.Tensor):
            result[key] = t.numpy()
        else:
            result[key] = np.array(t)
    # Ensure correct dtypes
    result["rgb"] = result["rgb"].astype(np.uint8)
    result["heightmap"] = result["heightmap"].astype(np.float32)
    result["terrain_idx"] = result["terrain_idx"].astype(np.int8)
    result["berry_mask"] = result["berry_mask"].astype(bool)
    result["visibility_lut"] = result["visibility_lut"].astype(np.uint8)
    # Biome labels (string per map); default to "unknown" if dataset predates them.
    biomes = data.get("biomes", None)
    if biomes is None:
        biomes = ["unknown"] * result["rgb"].shape[0]
    result["biomes"] = np.array([str(b) for b in biomes], dtype=object)
    return result


TARGET_GAP = 3  # column offset: YES _ _ _ NO on the same row


def _sample_spawn_target_batch(
    terrain_idx: np.ndarray,
    map_indices: np.ndarray,
    rng: np.random.Generator,
    min_manhattan: int = 0,
    water_idx: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample spawn and paired (YES, NO) targets for a batch of envs.

    YES is placed on a sampled land cell. NO is ``TARGET_GAP`` columns to the
    right of YES, on the same row. Both targets plus all intervening cells
    must be land. Spawn is sampled on land with Manhattan distance
    ``>= min_manhattan`` from the targets' midpoint.

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

    for i in range(B):
        tidx = terrain_idx[map_indices[i]]
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
            if abs(sr - mid_r) + abs(sc - mid_c) >= min_manhattan:
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


def _compute_minimap_batch(
    rgb: np.ndarray,
    heightmap: np.ndarray,
    terrain_idx: np.ndarray,
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
    """Compute minimap observations for a batch.

    Returns: float32 [B, 5, 45, 45] where channels are:
        0-2: RGB patch (true map colors; unseen cells are 0)
        3:   visibility mask (1.0 visible, 0.0 occluded / out-of-bounds)
        4:   target indicator (YES target: 1.0, NO target: 0.5, 0.0 if not visible or out of patch)

    Fully vectorised over the batch. When ``occlude=True`` and
    ``vis_lut_packed`` + ``disk_stack`` are provided, occlusion is a single
    fancy-index into the LUT + AND with ``disk_stack[vis_r]``.
    """
    B = len(pos_r)
    R = MINIMAP_RADIUS
    D = MINIMAP_DIAMETER
    H, W = rgb.shape[1], rgb.shape[2]

    # --- Per-env vis radius from current terrain ----------------------------
    pos_r_c = np.clip(pos_r, 0, H - 1)
    pos_c_c = np.clip(pos_c, 0, W - 1)
    t_idx = terrain_idx[map_idx, pos_r_c, pos_c_c]
    t_idx = np.clip(t_idx, 0, len(vis_per_terrain) - 1).astype(np.int32)
    vis_r_b = vis_per_terrain[t_idx]  # [B]

    # --- Visibility masks [B, D, D] -----------------------------------------
    if occlude and vis_lut_packed is not None and disk_stack is not None:
        # LUT fast path. ``vis_lut_packed[mi, pr, pc]`` with [B] indices
        # returns [B, 254]. Batched unpack + AND with per-env disk.
        packed = vis_lut_packed[map_idx, pos_r, pos_c]                 # [B, 254]
        full = np.unpackbits(packed, axis=1, bitorder="little")
        full = full[:, : D * D].reshape(B, D, D).astype(bool)
        vis_masks = full & disk_stack[vis_r_b]                         # [B, D, D]
    elif occlude:
        # Fallback: live Bresenham raycast per env (should not be hit in
        # practice — ``_load_maps`` now requires the LUT).
        vis_masks = np.zeros((B, D, D), dtype=bool)
        for b in range(B):
            vis_masks[b] = _compute_occlusion_mask(
                heightmap[map_idx[b]], int(pos_r[b]), int(pos_c[b]),
                int(vis_r_b[b]), H, W,
            )
    else:
        # No occlusion: simple disk per env (test fast path).
        yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
        dist_sq = yy * yy + xx * xx                                    # [D, D]
        vis_masks = dist_sq[None] <= (vis_r_b[:, None, None] ** 2)     # [B, D, D]

    # --- RGB patch extraction: single fancy-index call ----------------------
    di = np.arange(-R, R + 1, dtype=pos_r.dtype)
    rows = pos_r[:, None, None] + di[None, :, None]                    # [B, D, 1]
    cols = pos_c[:, None, None] + di[None, None, :]                    # [B, 1, D]
    rows_b = np.broadcast_to(rows, (B, D, D))
    cols_b = np.broadcast_to(cols, (B, D, D))

    in_bounds = (rows_b >= 0) & (rows_b < H) & (cols_b >= 0) & (cols_b < W)
    rows_c = np.clip(rows_b, 0, H - 1)
    cols_c = np.clip(cols_b, 0, W - 1)
    mi_b = np.broadcast_to(map_idx[:, None, None], (B, D, D))

    patches = rgb[mi_b, rows_c, cols_c]                                # [B, D, D, 3]
    valid = vis_masks & in_bounds                                      # [B, D, D]
    patches = np.where(valid[..., None], patches, 0)                   # zero masked cells

    # --- Assemble output ----------------------------------------------------
    result = np.empty((B, 5, D, D), dtype=np.float32)
    # Channels 0-2: RGB
    result[:, :3] = patches.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    # Channel 3: visibility mask
    result[:, 3] = vis_masks.astype(np.float32)
    # Channel 4: target indicator — NO=0.5, YES=1.0 on a single channel.
    result[:, 4] = 0.0
    for tr, tc, val in ((no_r, no_c, 0.5), (yes_r, yes_c, 1.0)):
        ty = tr - pos_r + R
        tx = tc - pos_c + R
        ty_c = np.clip(ty, 0, D - 1)
        tx_c = np.clip(tx, 0, D - 1)
        in_patch = (ty >= 0) & (ty < D) & (tx >= 0) & (tx < D)
        visible = in_patch & vis_masks[np.arange(B), ty_c, tx_c]
        if visible.any():
            env_idx = np.where(visible)[0]
            result[env_idx, 4, ty[env_idx], tx[env_idx]] = val

    return result


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

        # Load maps
        maps = _load_maps(maps_path)
        self._rgb = maps["rgb"]           # [N, 128, 128, 3]
        self._heightmap = maps["heightmap"]  # [N, 128, 128]
        self._terrain_idx = maps["terrain_idx"]  # [N, 128, 128]
        self._berry_mask = maps["berry_mask"]  # [N, 128, 128]
        self._vis_lut_packed = maps["visibility_lut"]  # [N, 128, 128, 254] uint8
        self._biomes = maps["biomes"]      # object array [N] of biome name strings
        self._num_maps = self._rgb.shape[0]
        self._map_size = self._rgb.shape[1]

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

        # Vectorised minimap helpers — precomputed once so _compute_minimap_batch
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

        # Cache of per-map HP-drain graphs (built lazily on first use)
        self._graph_cache: dict[int, csr_matrix] = {}

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def action_space(self) -> int:
        return NUM_ACTIONS

    def observation_space(self) -> dict:
        return {
            "minimap": (5, MINIMAP_DIAMETER, MINIMAP_DIAMETER),
            "scalars": (6,),
        }

    def _assign_maps(self, count: int) -> np.ndarray:
        """Assign map indices cycling through the pool."""
        indices = np.arange(count) + self._map_counter
        self._map_counter += count
        return (indices % self._num_maps).astype(np.int32)

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
        (
            self.spawn_r, self.spawn_c,
            self.yes_r, self.yes_c,
            self.no_r, self.no_c,
        ) = _sample_spawn_target_batch(
            self._terrain_idx, self.map_idx, self._rng,
            min_manhattan=self._min_manhattan,
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

        sr, sc, yr, yc, nr, nc = _sample_spawn_target_batch(
            self._terrain_idx, new_map_idx, self._rng,
            min_manhattan=self._min_manhattan,
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

        # Snapshot cost-to-go at current position BEFORE the step is applied,
        # so the reward function can compute PBRS progress = ctg_prev - ctg_curr.
        ctg_prev = self.ctg[np.arange(B), self.pos_r, self.pos_c].copy()

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

        # --- Movement actions (0-3) ---
        move_mask = is_move & ~self.done
        if move_mask.any():
            deltas = MOVE_DELTAS[actions[move_mask]]  # [M, 2]
            idx = np.where(move_mask)[0]

            new_r = self.pos_r[idx] + deltas[:, 0]
            new_c = self.pos_c[idx] + deltas[:, 1]

            # Bounds check
            in_bounds = (
                (new_r >= 0) & (new_r < self._map_size) &
                (new_c >= 0) & (new_c < self._map_size)
            )

            # For in-bounds moves, check terrain
            for j, env_i in enumerate(idx):
                if not in_bounds[j]:
                    # Out of bounds: no-op, still costs a step
                    self.steps[env_i] += 1
                    continue

                nr, nc = int(new_r[j]), int(new_c[j])
                mi = self.map_idx[env_i]
                t_idx = int(self._terrain_idx[mi, nr, nc])

                if t_idx < 0:
                    # Deadly terrain — instant death
                    self.pos_r[env_i] = nr
                    self.pos_c[env_i] = nc
                    self.hp[env_i] = 0.0
                    self.steps[env_i] += 1
                    self.done[env_i] = True
                    continue

                terrain_name = TERRAIN_NAMES[t_idx]
                new_consec = (
                    int(self.consec_grass[env_i]) + 1
                    if terrain_name == "grassland"
                    else 0
                )
                tools = self._tool_set(int(self.tool[env_i]))
                # Berry tiles are a free step (0 drain) — overrides terrain drain.
                if self._berry_mask[mi, nr, nc]:
                    drain = 0.0
                else:
                    drain = drain_for(terrain_name, tools, new_consec, self._effects)

                self.hp[env_i] -= drain
                self.pos_r[env_i] = nr
                self.pos_c[env_i] = nc
                self.consec_grass[env_i] = new_consec
                self.steps[env_i] += 1

                if self.hp[env_i] <= 0:
                    self.hp[env_i] = 0.0
                    self.done[env_i] = True
                elif (
                    (self.pos_r[env_i] == self.yes_r[env_i] and
                     self.pos_c[env_i] == self.yes_c[env_i])
                    or
                    (self.pos_r[env_i] == self.no_r[env_i] and
                     self.pos_c[env_i] == self.no_c[env_i])
                ):
                    self.done[env_i] = True

        # --- Forage action (4) ---
        forage_mask = is_forage & ~self.done
        if forage_mask.any():
            for env_i in np.where(forage_mask)[0]:
                r, c = int(self.pos_r[env_i]), int(self.pos_c[env_i])
                mi = self.map_idx[env_i]
                t_idx = int(self._terrain_idx[mi, r, c])
                if t_idx < 0:
                    self.steps[env_i] += 1
                    continue

                terrain_name = TERRAIN_NAMES[t_idx]
                is_berry = bool(self._berry_mask[mi, r, c])

                if is_berry:
                    self.hp[env_i] = min(
                        float(self._effects.hp_max),
                        float(self.hp[env_i]) + self._effects.berry_heal,
                    )
                    self.consec_grass[env_i] = 0
                    self.steps[env_i] += 1
                    continue

                if terrain_name == "forest":
                    self.wood[env_i] = min(
                        int(self.wood[env_i]) + self._effects.forest_wood,
                        self._effects.wood_max,
                    )

                new_consec = (
                    int(self.consec_grass[env_i]) + 1
                    if terrain_name == "grassland"
                    else 0
                )
                tools = self._tool_set(int(self.tool[env_i]))
                drain = drain_for(terrain_name, tools, new_consec, self._effects)
                self.hp[env_i] -= drain
                self.consec_grass[env_i] = new_consec
                self.steps[env_i] += 1

                if self.hp[env_i] <= 0:
                    self.hp[env_i] = 0.0
                    self.done[env_i] = True

        # --- Craft actions (5-7) ---
        craft_mask = is_craft & ~self.done
        if craft_mask.any():
            for env_i in np.where(craft_mask)[0]:
                action = int(actions[env_i])
                tool_name = CRAFT_TOOLS[action]
                tool_id = {"raft": 1, "rope": 2, "shoes": 3}[tool_name]

                if (self.tool[env_i] == 0 and
                        self.wood[env_i] >= self._effects.craft_cost):
                    self.wood[env_i] -= self._effects.craft_cost
                    self.tool[env_i] = tool_id
                    self.crafted_this_step[env_i] = tool_id

                r, c = int(self.pos_r[env_i]), int(self.pos_c[env_i])
                mi = self.map_idx[env_i]
                t_idx = int(self._terrain_idx[mi, r, c])
                if t_idx >= 0:
                    terrain_name = TERRAIN_NAMES[t_idx]
                    if self._berry_mask[mi, r, c]:
                        drain = 0.0
                        new_consec = 0
                    else:
                        new_consec = (
                            int(self.consec_grass[env_i]) + 1
                            if terrain_name == "grassland"
                            else 0
                        )
                        tools = self._tool_set(int(self.tool[env_i]))
                        drain = drain_for(terrain_name, tools, new_consec, self._effects)
                    self.hp[env_i] -= drain
                    self.consec_grass[env_i] = new_consec

                self.steps[env_i] += 1

                if self.hp[env_i] <= 0:
                    self.hp[env_i] = 0.0
                    self.done[env_i] = True

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
            "hp": self.hp.copy(),
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

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Build observation dict for all envs."""
        B = self._num_envs

        # Compute minimap
        minimap = _compute_minimap_batch(
            self._rgb, self._heightmap, self._terrain_idx,
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
        for i in range(B):
            mi = self.map_idx[i]
            r, c = int(self.pos_r[i]), int(self.pos_c[i])
            if self._berry_mask[mi, r, c]:
                tile_cls = 9
            else:
                tile_cls = max(0, int(self._terrain_idx[mi, r, c]))
            scalars[i, 2] = tile_cls / 9.0

        scalars[:, 3] = self.hp / float(self._effects.hp_max)
        scalars[:, 4] = self.wood.astype(np.float32) / float(self._effects.wood_max)
        scalars[:, 5] = self.tool.astype(np.float32) / 3.0

        return {
            "minimap": minimap,
            "scalars": scalars,
        }
