"""Islands class — thin wrapper around the pure-function core.

Owns a pool of world_maps (generated once at init) and delegates all step
logic to core.py.  Supports Level Replay: each environment in the batch
may operate on a different map, and maps are re-sampled on episode reset.
"""

from __future__ import annotations

import math
import random


import numpy as np
import torch

from cogniland.env.constants import TERRAIN_COSTS, TERRAIN_THRESHOLDS
from cogniland.env.core import compute_minimap_batch, compute_terrain_levels, env_step
from cogniland.env.pathfinding import batch_dijkstra_from_sources
from cogniland.env.types import CurriculumStage, EnvConfig, EnvState, StepResult


def generate_island(config: EnvConfig) -> torch.Tensor:
    """Generate a single island heightmap on CPU.

    Uses the bundled SimplexNoise library.  This is the only part of the
    pipeline that runs nested Python loops — but it only happens once at init,
    so it is not a training bottleneck.
    """
    from cogniland.simplexnoise.noise import SimplexNoise, normalize

    size = config.size
    scale = size * config.scale
    sn = SimplexNoise(num_octaves=config.octaves, persistence=config.persistence, dimensions=2)

    world = torch.zeros(size, size)
    for i in range(size):
        for j in range(size):
            world[i, j] = normalize(sn.fractal(i, j, hgrid=scale, lacunarity=config.lacunarity))

    # Sink mode
    if config.sink_mode == 1:
        world = world ** 3
    elif config.sink_mode == 2:
        world = (2 * world) ** 2

    world = world / torch.max(world)

    # Filtering (island shape)
    if config.filtering:
        center = size // 2
        circle_grad = torch.zeros(size, size)

        for y in range(size):
            for x in range(size):
                dx = abs(x - center)
                dy = abs(y - center)
                if config.filtering == "circle":
                    dist = math.sqrt(dx * dx + dy * dy)
                elif config.filtering == "diamond":
                    dist = dx + dy
                elif config.filtering == "square":
                    dist = max(dx ** 2, dy ** 2)
                else:
                    raise ValueError(f"Unknown filtering: {config.filtering}")
                circle_grad[y, x] = dist

        circle_grad = circle_grad / torch.max(circle_grad)
        circle_grad = -(circle_grad - 0.5) * 2.0

        for y in range(size):
            for x in range(size):
                if circle_grad[y, x] > 0:
                    circle_grad[y, x] *= 20

        circle_grad = circle_grad / torch.max(circle_grad)

        world_noise = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                world_noise[i, j] = world[i, j] * circle_grad[i, j]
                if world_noise[i, j] > 0:
                    world_noise[i, j] *= 20

        world_noise = world_noise / torch.max(world_noise)
        world = world_noise

    return world


def colorize(world_map: torch.Tensor, config: EnvConfig) -> torch.Tensor:
    """Convert heightmap to [H, W, 3] uint8 color tensor for visualisation."""
    from cogniland.env.constants import palette, TERRAIN_LEVELS

    threshold = 0.02 if config.sink_mode == 1 else (0.1 if config.sink_mode == 2 else 0.2)

    # Derive color order from canonical TERRAIN_LEVELS
    color_order = [(TERRAIN_LEVELS[i]["color"], TERRAIN_LEVELS[i]["threshold"]) for i in range(9)]

    color_world = torch.zeros(*world_map.shape, 3)
    for i in range(world_map.shape[0]):
        for j in range(world_map.shape[1]):
            val = world_map[i, j].item()
            for name, t in color_order:
                if val < threshold + t:
                    color_world[i, j] = torch.tensor(palette[name], dtype=torch.float32)
                    break

    return color_world


class Islands:
    """Batched island navigation environment with Level Replay.

    In procedural mode (map_name == ""), pre-generates a pool of N maps at init.
    Each environment in the batch may be on a different map. On episode reset,
    a new map is randomly sampled from the pool and spawn/target are re-sampled.
    """

    def __init__(
        self,
        config: EnvConfig | None = None,
        world_maps: torch.Tensor | None = None,
        **kwargs,
    ):
        if config is None:
            config = EnvConfig(**kwargs)
        self.config = config
        self._device = config.resolved_device()

        # Seed all RNGs — SimplexNoise uses Python's random module internally
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

        if world_maps is not None:
            # Pre-generated maps provided externally (from MapDataset) — skip generation
            self.world_maps = world_maps.to(self._device)
            self._fixed_spawn: tuple[int, int] | None = None
            self._fixed_target: tuple[int, int] | None = None
        elif config.map_name:
            # Custom map mode — single map, no pool
            from cogniland.env import custom_maps as cm
            single_map = cm.get_map(config.map_name).to(self._device)
            self.world_maps = single_map.unsqueeze(0)   # [1, H, W]
            self._fixed_spawn = cm.get_spawn(config.map_name)
            self._fixed_target = cm.get_target(config.map_name)
        else:
            # Procedural pool — generate N maps with different seeds
            pool_size = config.map_pool_size
            print(f"Generating {pool_size} procedural maps ({config.size}×{config.size}) ...")
            maps = []
            for i in range(pool_size):
                seed_i = config.seed + i
                torch.manual_seed(seed_i)
                random.seed(seed_i)
                np.random.seed(seed_i)
                maps.append(generate_island(config))
            self.world_maps = torch.stack(maps).to(self._device)  # [N, H, W]
            print(f"Map pool ready: {self.world_maps.shape}")
            self._fixed_spawn = None
            self._fixed_target = None

        # Backward-compat: world_map points to the first map by default.
        # Eval code and trajectory rendering that uses self.world_map will
        # get map 0; per-env indexing uses self.world_maps.
        self.world_map = self.world_maps[0]

        # Per-run position overrides (config.spawn_r/c, target_r/c)
        if config.spawn_r >= 0:
            self._fixed_spawn = (config.spawn_r, config.spawn_c)
        if config.target_r >= 0:
            self._fixed_target = (config.target_r, config.target_c)

        # Per-env map assignment (set during reset)
        self._env_map_idx: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        batch_size: int,
        seed: int | None = None,
        curriculum_stage: CurriculumStage = CurriculumStage.NORMAL,
    ) -> tuple[EnvState, torch.Tensor]:
        """Reset: sample maps, spawn + target on land, return (initial_state, target_positions)."""
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        N = self.world_maps.shape[0]
        size = self.config.size

        # Assign each env a random map from the pool
        self._env_map_idx = torch.randint(0, N, (batch_size,), device=self._device)

        land_threshold = TERRAIN_THRESHOLDS[2].item()  # must be above water

        # Sample or use fixed spawn/target positions (per env, on its assigned map)
        if self._fixed_spawn is not None:
            r, c = self._fixed_spawn
            spawn_pos = torch.tensor([[r, c]], device=self._device).expand(batch_size, 2).clone()
        else:
            spawn_pos = self._sample_land_positions_batched(
                self._env_map_idx, land_threshold, curriculum_stage
            )

        if self._fixed_target is not None:
            r, c = self._fixed_target
            target_pos = torch.tensor([[r, c]], device=self._device).expand(batch_size, 2).clone()
        else:
            target_pos = self._sample_land_positions_batched(
                self._env_map_idx, land_threshold, curriculum_stage
            )

        # Build per-env world maps for batched ops: [B, H, W]
        per_env_maps = self.world_maps[self._env_map_idx]  # [B, H, W]

        terrain_idx = compute_terrain_levels(per_env_maps, spawn_pos)
        minimap = compute_minimap_batch(
            per_env_maps, spawn_pos,
            self.config.minimap_max_ray, terrain_idx,
            self.config.minimap_occlude,
            self.config.minimap_clear_tolerance,
            target_pos=target_pos,
        )
        compass_raw = (spawn_pos - target_pos).float()
        compass = compass_raw / torch.norm(compass_raw, dim=1, keepdim=True).clamp(min=1e-8)

        # Precompute optimal time cost spawn→target via Dijkstra (episode constant)
        dist_maps = batch_dijkstra_from_sources(
            per_env_maps.cpu(), TERRAIN_COSTS, spawn_pos.cpu()
        )
        dijkstra_cost = torch.tensor([
            dist_maps[i][target_pos[i, 0].item(), target_pos[i, 1].item()]
            for i in range(batch_size)
        ], dtype=torch.float32, device=self._device)
        if not torch.all(torch.isfinite(dijkstra_cost)):
            raise ValueError("Dijkstra returned inf cost — disconnected map at reset()")

        state = EnvState(
            position=spawn_pos,
            minimap=minimap,
            compass=compass,
            terrain_idx=terrain_idx,
            resources=torch.full((batch_size,), self.config.init_resources, device=self._device),
            hp=torch.full((batch_size,), self.config.init_hp, device=self._device),
            cost=torch.zeros(batch_size, device=self._device),
            dijkstra_cost=dijkstra_cost,
        )
        return state, target_pos

    def _sample_land_positions_batched(
        self,
        map_indices: torch.Tensor,
        land_threshold: float,
        curriculum_stage: CurriculumStage = CurriculumStage.NORMAL,
    ) -> torch.Tensor:
        """Sample one land position per env, where each env uses its own map.

        In EASY mode, positions are constrained to a circle of radius
        ``config.curriculum_easy_radius`` around the map center.
        """
        B = map_indices.shape[0]
        size = self.config.size
        positions = torch.zeros(B, 2, dtype=torch.long, device=self._device)

        if curriculum_stage == CurriculumStage.EASY:
            center = size // 2
            radius = self.config.curriculum_easy_radius
            r_lo = max(0, center - radius)
            r_hi = min(size - 1, center + radius)
            c_lo = max(0, center - radius)
            c_hi = min(size - 1, center + radius)

        for b in range(B):
            m = self.world_maps[map_indices[b]]  # [H, W]
            while True:
                if curriculum_stage == CurriculumStage.EASY:
                    r = random.randint(r_lo, r_hi)
                    c = random.randint(c_lo, c_hi)
                    if (r - center) ** 2 + (c - center) ** 2 > radius * radius:
                        continue
                    p = torch.tensor([r, c], dtype=torch.long, device=self._device)
                else:
                    p = torch.randint(0, size, (2,), device=self._device)
                if m[p[0], p[1]].item() > land_threshold:
                    positions[b] = p
                    break
        return positions

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, state: EnvState, action: torch.Tensor, target_pos: torch.Tensor) -> StepResult:
        """Single batched step — delegates to core.env_step.

        Passes per-env maps [B, H, W] so each env steps on its own map.
        """
        per_env_maps = self.world_maps[self._env_map_idx]  # [B, H, W]
        return env_step(state, action, per_env_maps, target_pos, self.config)

    # ------------------------------------------------------------------
    # Auto-reset helper
    # ------------------------------------------------------------------

    def reset_done(
        self,
        state: EnvState,
        target_pos: torch.Tensor,
        done: torch.Tensor,
        curriculum_stage: CurriculumStage = CurriculumStage.NORMAL,
    ) -> tuple[EnvState, torch.Tensor]:
        """Re-sample only the environments where done[i]==True.

        Each done env gets a new random map from the pool plus
        fresh spawn/target positions on that map.
        """
        if not done.any():
            return state, target_pos

        n_done = int(done.sum().item())
        N = self.world_maps.shape[0]
        land_threshold = TERRAIN_THRESHOLDS[2].item()

        # Assign new random map for each done env
        new_map_idx = torch.randint(0, N, (n_done,), device=self._device)
        self._env_map_idx[done] = new_map_idx

        if self._fixed_spawn is not None:
            r, c = self._fixed_spawn
            new_spawn = torch.tensor([[r, c]], device=self._device).expand(n_done, 2).clone()
        else:
            new_spawn = self._sample_land_positions_batched(
                new_map_idx, land_threshold, curriculum_stage
            )

        if self._fixed_target is not None:
            r, c = self._fixed_target
            new_target = torch.tensor([[r, c]], device=self._device).expand(n_done, 2).clone()
        else:
            new_target = self._sample_land_positions_batched(
                new_map_idx, land_threshold, curriculum_stage
            )

        # Per-env maps for the done environments
        done_maps = self.world_maps[new_map_idx]   # [n_done, H, W]

        # Build replacement state fields
        new_terrain = compute_terrain_levels(done_maps, new_spawn)
        new_minimap = compute_minimap_batch(
            done_maps, new_spawn,
            self.config.minimap_max_ray, new_terrain,
            self.config.minimap_occlude,
            self.config.minimap_clear_tolerance,
            target_pos=new_target,
        )
        new_compass_raw = (new_spawn - new_target).float()
        new_compass = new_compass_raw / torch.norm(new_compass_raw, dim=1, keepdim=True).clamp(min=1e-8)

        # Precompute optimal time cost for done envs
        done_dist_maps = batch_dijkstra_from_sources(
            done_maps.cpu(), TERRAIN_COSTS, new_spawn.cpu()
        )
        new_dijkstra_cost_np = torch.tensor([
            done_dist_maps[i][new_target[i, 0].item(), new_target[i, 1].item()]
            for i in range(n_done)
        ], dtype=torch.float32, device=self._device)
        if not torch.all(torch.isfinite(new_dijkstra_cost_np)):
            raise ValueError("Dijkstra returned inf cost — disconnected map at reset_done()")

        # Replace done environments in each tensor
        position = state.position.clone()
        position[done] = new_spawn

        minimap = state.minimap.clone()
        minimap[done] = new_minimap

        compass = state.compass.clone()
        compass[done] = new_compass

        terrain_idx = state.terrain_idx.clone()
        terrain_idx[done] = new_terrain

        resources = state.resources.clone()
        resources[done] = self.config.init_resources

        hp = state.hp.clone()
        hp[done] = self.config.init_hp

        cost = state.cost.clone()
        cost[done] = 0.0

        dijkstra_cost = state.dijkstra_cost.clone()
        dijkstra_cost[done] = new_dijkstra_cost_np

        new_state = EnvState(
            position=position, minimap=minimap, compass=compass,
            terrain_idx=terrain_idx,
            resources=resources, hp=hp, cost=cost,
            dijkstra_cost=dijkstra_cost,
        )

        new_targets = target_pos.clone()
        new_targets[done] = new_target

        return new_state, new_targets
