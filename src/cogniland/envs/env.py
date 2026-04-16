"""Batched Cogniland environment — pure numpy, no PyTorch/JAX inside.

Runs B parallel games simultaneously. Each game has:
  - An agent with HP, wood, tool, position
  - A 128x128 map with terrain, berries, heightmap, RGB
  - Spawn and target positions
  - 8 actions: 4 cardinal moves, forage, craft_raft, craft_rope, craft_shoes

Observations:
  - minimap: float32 [B, 3, 45, 45] — RGB patch with occlusion
  - scalars: float32 [B, 6] — compass, terrain, hp, wood, tool
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

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
    result = {}
    for key in ("rgb", "heightmap", "terrain_idx", "berry_mask"):
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
    return result


def _sample_spawn_target_batch(
    terrain_idx: np.ndarray,
    map_indices: np.ndarray,
    rng: np.random.Generator,
    min_manhattan: int = 60,
    water_idx: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample spawn/target pairs for a batch of environments.

    Returns: spawn_r, spawn_c, target_r, target_c — all shape [B].
    """
    B = len(map_indices)
    spawn_r = np.zeros(B, dtype=np.int32)
    spawn_c = np.zeros(B, dtype=np.int32)
    target_r = np.zeros(B, dtype=np.int32)
    target_c = np.zeros(B, dtype=np.int32)

    for i in range(B):
        tidx = terrain_idx[map_indices[i]]
        land = np.argwhere(tidx > water_idx)
        if len(land) < 2:
            mid = tidx.shape[0] // 2
            spawn_r[i] = spawn_c[i] = target_r[i] = target_c[i] = mid
            continue
        for _ in range(500):
            si = rng.integers(len(land))
            ti = rng.integers(len(land))
            s = land[si]
            t = land[ti]
            if abs(int(s[0]) - int(t[0])) + abs(int(s[1]) - int(t[1])) >= min_manhattan:
                spawn_r[i], spawn_c[i] = int(s[0]), int(s[1])
                target_r[i], target_c[i] = int(t[0]), int(t[1])
                break
        else:
            spawn_r[i], spawn_c[i] = int(land[0, 0]), int(land[0, 1])
            target_r[i], target_c[i] = int(land[-1, 0]), int(land[-1, 1])

    return spawn_r, spawn_c, target_r, target_c


def _compute_minimap_batch(
    rgb: np.ndarray,
    heightmap: np.ndarray,
    terrain_idx: np.ndarray,
    map_idx: np.ndarray,
    pos_r: np.ndarray,
    pos_c: np.ndarray,
    target_r: np.ndarray,
    target_c: np.ndarray,
    terrain_vis_radius: dict[str, int],
    occlude: bool = True,
) -> np.ndarray:
    """Compute minimap observations for a batch.

    Returns: float32 [B, 3, 45, 45] — normalized RGB minimap.
    """
    B = len(pos_r)
    R = MINIMAP_RADIUS
    D = MINIMAP_DIAMETER
    H, W = rgb.shape[1], rgb.shape[2]

    result = np.zeros((B, 3, D, D), dtype=np.float32)

    # Vis radius per terrain index
    vis_per_terrain = np.array(
        [terrain_vis_radius.get(name, 7) for name in TERRAIN_NAMES],
        dtype=np.int32,
    )

    for b in range(B):
        mi = map_idx[b]
        pr, pc = int(pos_r[b]), int(pos_c[b])

        # Get visibility radius from current terrain
        t_idx = int(terrain_idx[mi, pr, pc]) if 0 <= pr < H and 0 <= pc < W else 0
        if t_idx < 0:
            t_idx = 0
        vis_r = int(vis_per_terrain[min(t_idx, len(vis_per_terrain) - 1)])

        # Build RGB patch
        patch = np.zeros((D, D, 3), dtype=np.uint8)

        if occlude:
            # Compute occlusion mask via raycasting
            vis_mask = _compute_occlusion_mask(
                heightmap[mi], pr, pc, vis_r, H, W
            )
        else:
            # Simple circular mask
            vis_mask = np.zeros((D, D), dtype=bool)
            yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
            vis_mask[yy * yy + xx * xx <= vis_r * vis_r] = True

        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                wr, wc = pr + dy, pc + dx
                py, px = dy + R, dx + R
                if not vis_mask[py, px]:
                    continue
                if 0 <= wr < H and 0 <= wc < W:
                    patch[py, px] = rgb[mi, wr, wc]

        # Target marker if visible
        ty = target_r[b] - pr + R
        tx = target_c[b] - pc + R
        if 0 <= ty < D and 0 <= tx < D and vis_mask[ty, tx]:
            # Green cross
            for d in range(-1, 2):
                for oy, ox in [(d, 0), (0, d)]:
                    ny, nx = ty + oy, tx + ox
                    if 0 <= ny < D and 0 <= nx < D:
                        patch[ny, nx] = (60, 255, 80)

        # Player dot at center
        patch[R, R] = (255, 60, 60)

        # Transpose to [3, D, D] and normalize
        result[b] = patch.transpose(2, 0, 1).astype(np.float32) / 255.0

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

    TODO: This is the expensive per-env loop. For large batch sizes, consider
    vectorizing or using a simplified circular mask as a fast path.

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
        self._num_maps = self._rgb.shape[0]
        self._map_size = self._rgb.shape[1]

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

        # RNG
        seed = config.seed if hasattr(config, "seed") else config.get("seed", 42)
        self._rng = np.random.default_rng(seed)

        # Map assignment counter
        self._map_counter = 0

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
        self.target_r: np.ndarray | None = None
        self.target_c: np.ndarray | None = None
        self.done: np.ndarray | None = None

        # Episode tracking
        self._episode_returns: np.ndarray | None = None
        self._episode_lengths: np.ndarray | None = None

        # Drain lookup arrays for vectorized computation
        self._hp_drain_arr = np.array(
            [self._effects.hp_drain.get(name, 1) for name in TERRAIN_NAMES],
            dtype=np.float32,
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def action_space(self) -> int:
        return NUM_ACTIONS

    def observation_space(self) -> dict:
        return {
            "minimap": (3, MINIMAP_DIAMETER, MINIMAP_DIAMETER),
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

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset all environments. Returns observation dict."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        B = self._num_envs

        self.map_idx = self._assign_maps(B)
        self.spawn_r, self.spawn_c, self.target_r, self.target_c = (
            _sample_spawn_target_batch(
                self._terrain_idx, self.map_idx, self._rng,
                min_manhattan=self._min_manhattan,
            )
        )
        self.pos_r = self.spawn_r.copy()
        self.pos_c = self.spawn_c.copy()
        self.hp = np.full(B, float(self._effects.init_hp), dtype=np.float32)
        self.wood = np.zeros(B, dtype=np.int32)
        self.tool = np.zeros(B, dtype=np.int32)
        self.consec_grass = np.zeros(B, dtype=np.int32)
        self.steps = np.zeros(B, dtype=np.int32)
        self.done = np.zeros(B, dtype=bool)

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

        sr, sc, tr, tc = _sample_spawn_target_batch(
            self._terrain_idx, new_map_idx, self._rng,
            min_manhattan=self._min_manhattan,
        )
        self.spawn_r[indices] = sr
        self.spawn_c[indices] = sc
        self.target_r[indices] = tr
        self.target_c[indices] = tc
        self.pos_r[indices] = sr
        self.pos_c[indices] = sc

        self.hp[indices] = float(self._effects.init_hp)
        self.wood[indices] = 0
        self.tool[indices] = 0
        self.consec_grass[indices] = 0
        self.steps[indices] = 0
        self.done[indices] = False

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
                drain = drain_for(terrain_name, tools, new_consec, self._effects)

                self.hp[env_i] -= drain
                self.pos_r[env_i] = nr
                self.pos_c[env_i] = nc
                self.consec_grass[env_i] = new_consec
                self.steps[env_i] += 1

                if self.hp[env_i] <= 0:
                    self.hp[env_i] = 0.0
                    self.done[env_i] = True
                elif (self.pos_r[env_i] == self.target_r[env_i] and
                      self.pos_c[env_i] == self.target_c[env_i]):
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

                if self._berry_mask[mi, r, c]:
                    # Berry forage: heal, no drain
                    self.hp[env_i] = min(
                        float(self._effects.hp_max),
                        float(self.hp[env_i]) + self._effects.berry_heal,
                    )
                    self.steps[env_i] += 1
                elif terrain_name == "forest":
                    # Forest forage: +wood, costs drain
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
                else:
                    # No-op forage, still costs a step
                    self.steps[env_i] += 1

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
                # Regardless of success, costs a step
                self.steps[env_i] += 1

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

        info = {
            "returned_episode_returns": returned_episode_returns,
            "returned_episode_lengths": returned_episode_lengths,
            "returned_episode": returned_episode,
            # Extra info for reward computation
            "reached": (
                (self.pos_r == self.target_r) &
                (self.pos_c == self.target_c) &
                self.done
            ),
            "alive": self.hp > 0,
            "dist_to_target": np.sqrt(
                (self.pos_r.astype(np.float32) - self.target_r.astype(np.float32)) ** 2 +
                (self.pos_c.astype(np.float32) - self.target_c.astype(np.float32)) ** 2
            ),
            "initial_dist": np.sqrt(
                (self.spawn_r.astype(np.float32) - self.target_r.astype(np.float32)) ** 2 +
                (self.spawn_c.astype(np.float32) - self.target_c.astype(np.float32)) ** 2
            ),
        }

        dones = self.done.copy()

        # Auto-reset done envs
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
            self.target_r, self.target_c,
            self._terrain_vis_radius,
            occlude=self._occlude,
        )

        # Compute scalars
        scalars = np.zeros((B, 6), dtype=np.float32)

        # Compass: unit vector from agent to target
        dr = self.target_r.astype(np.float32) - self.pos_r.astype(np.float32)
        dc = self.target_c.astype(np.float32) - self.pos_c.astype(np.float32)
        dist = np.sqrt(dr * dr + dc * dc)
        dist = np.maximum(dist, 1e-6)
        scalars[:, 0] = dc / dist  # compass_x (column direction)
        scalars[:, 1] = dr / dist  # compass_y (row direction)

        # Current terrain index (normalized)
        for i in range(B):
            mi = self.map_idx[i]
            r, c = int(self.pos_r[i]), int(self.pos_c[i])
            t_idx = int(self._terrain_idx[mi, r, c])
            scalars[i, 2] = max(0, t_idx) / 8.0

        scalars[:, 3] = self.hp / float(self._effects.hp_max)
        scalars[:, 4] = self.wood.astype(np.float32) / float(self._effects.wood_max)
        scalars[:, 5] = self.tool.astype(np.float32) / 3.0

        return {
            "minimap": minimap,
            "scalars": scalars,
        }
