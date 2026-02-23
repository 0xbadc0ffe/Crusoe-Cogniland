"""Islands class — thin wrapper around the pure-function core.

Owns the world_map (generated once) and delegates all step logic to core.py.
"""

from __future__ import annotations

import math
import random
import sys
import os

import numpy as np
import torch

from cogniland.env.constants import TERRAIN_THRESHOLDS
from cogniland.env.core import compute_minimap_batch, compute_terrain_levels, env_step
from cogniland.env.types import EnvConfig, EnvState, StepResult


def generate_island(config: EnvConfig) -> torch.Tensor:
    """Generate a single island heightmap on CPU.

    Uses the bundled SimplexNoise library.  This is the only part of the
    pipeline that runs nested Python loops — but it only happens once at init,
    so it is not a training bottleneck.
    """
    # Make sure lib/ is importable
    lib_dir = os.path.join(os.path.dirname(__file__), "..", "..", "lib")
    if lib_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(lib_dir))

    from simplexnoise.noise import SimplexNoise, normalize

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
    """Batched island navigation environment.

    Owns the world_map and delegates step logic to pure functions in core.py.
    """

    def __init__(self, config: EnvConfig | None = None, **kwargs):
        if config is None:
            config = EnvConfig(**kwargs)
        self.config = config
        self._device = config.resolved_device()

        # Generate island (CPU) then move to device
        # Seed all RNGs — SimplexNoise uses Python's random module internally
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        self.world_map = generate_island(config).to(self._device)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, batch_size: int, seed: int | None = None) -> tuple[EnvState, torch.Tensor]:
        """Reset: sample spawn + target on land, return (initial_state, target_positions)."""
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        land_threshold = TERRAIN_THRESHOLDS[2].item()  # must be above water
        size = self.config.size

        # Sample valid land positions for spawns and targets
        spawn_pos = self._sample_land_positions(batch_size, land_threshold)
        target_pos = self._sample_land_positions(batch_size, land_threshold)

        minimap = compute_minimap_batch(
            self.world_map, spawn_pos,
            self.config.minimap_ray, self.config.minimap_occlude,
            self.config.minimap_min_clear_lv,
        )
        compass = (spawn_pos - target_pos).float()
        terrain_lev = compute_terrain_levels(self.world_map, spawn_pos)

        state = EnvState(
            position=spawn_pos,
            minimap=minimap,
            compass=compass,
            terrain_lev=terrain_lev,
            terrain_clock=torch.zeros(batch_size, device=self._device),
            resources=torch.full((batch_size,), self.config.init_resources, device=self._device),
            hp=torch.full((batch_size,), self.config.init_hp, device=self._device),
            cost=torch.zeros(batch_size, device=self._device),
        )
        return state, target_pos

    def _sample_land_positions(self, n: int, land_threshold: float) -> torch.Tensor:
        """Sample n positions that are above `land_threshold` on the world_map."""
        size = self.config.size
        positions = []
        while len(positions) < n:
            p = torch.randint(0, size, (2,), device=self._device)
            if self.world_map[p[0], p[1]].item() > land_threshold:
                positions.append(p)
        return torch.stack(positions, dim=0)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, state: EnvState, action: torch.Tensor, target_pos: torch.Tensor) -> StepResult:
        """Single batched step — delegates to core.env_step."""
        return env_step(state, action, self.world_map, target_pos, self.config)

    # ------------------------------------------------------------------
    # Auto-reset helper
    # ------------------------------------------------------------------

    def reset_done(
        self, state: EnvState, target_pos: torch.Tensor, done: torch.Tensor
    ) -> tuple[EnvState, torch.Tensor]:
        """Re-sample only the environments where done[i]==True."""
        if not done.any():
            return state, target_pos

        n_done = done.sum().item()
        land_threshold = TERRAIN_THRESHOLDS[2].item()

        new_spawn = self._sample_land_positions(n_done, land_threshold)
        new_target = self._sample_land_positions(n_done, land_threshold)

        # Build replacement state fields
        new_minimap = compute_minimap_batch(
            self.world_map, new_spawn,
            self.config.minimap_ray, self.config.minimap_occlude,
            self.config.minimap_min_clear_lv,
        )
        new_compass = (new_spawn - new_target).float()
        new_terrain = compute_terrain_levels(self.world_map, new_spawn)

        # Replace done environments in each tensor
        position = state.position.clone()
        position[done] = new_spawn

        minimap = state.minimap.clone()
        minimap[done] = new_minimap

        compass = state.compass.clone()
        compass[done] = new_compass

        terrain_lev = state.terrain_lev.clone()
        terrain_lev[done] = new_terrain

        terrain_clock = state.terrain_clock.clone()
        terrain_clock[done] = 0.0

        resources = state.resources.clone()
        resources[done] = self.config.init_resources

        hp = state.hp.clone()
        hp[done] = self.config.init_hp

        cost = state.cost.clone()
        cost[done] = 0.0

        new_state = EnvState(
            position=position, minimap=minimap, compass=compass,
            terrain_lev=terrain_lev, terrain_clock=terrain_clock,
            resources=resources, hp=hp, cost=cost,
        )

        new_targets = target_pos.clone()
        new_targets[done] = new_target

        return new_state, new_targets
