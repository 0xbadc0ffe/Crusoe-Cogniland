"""Immutable state types for the environment.

NamedTuples are chosen for JAX compatibility (they are pytrees).
`state._replace(hp=new_hp)` creates a new state without mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch


class EnvState(NamedTuple):
    """Full batched environment state."""

    position: torch.Tensor        # [B, 2] long
    minimap: torch.Tensor         # [B, 2, 2*max_ray+1, 2*max_ray+1] float (ch0=heightmap, ch1=visibility mask)
    compass: torch.Tensor         # [B, 2] float — unit direction vector (position − target) / dist
    terrain_lev: torch.Tensor     # [B] float
    terrain_clock: torch.Tensor   # [B] float
    resources: torch.Tensor       # [B] float
    hp: torch.Tensor              # [B] float
    cost: torch.Tensor            # [B] float


class StepResult(NamedTuple):
    """Result of a single environment step."""

    state: EnvState
    reward: torch.Tensor   # [B]
    done: torch.Tensor     # [B] bool
    info: dict


@dataclass(frozen=True)
class EnvConfig:
    """Immutable environment configuration."""

    # Island generation
    size: int = 250
    scale: float = 0.33
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    seed: int = 42
    detailed_ocean: bool = True
    filtering: str = "square"   # "circle", "square", "diamond"
    sink_mode: int = 1          # 0=none, 1, 2

    # Agent
    init_hp: float = 100.0
    max_hp: float = 100.0
    init_resources: float = 30.0
    max_resources: float = 100.0
    hard_mode: bool = False

    # Terrain effects (previously hardcoded in core.py)
    passive_heal_rate: float = 1.0
    land_to_water_resource_cost: float = 10.0  # boat construction cost (resources)
    land_to_water_hp_per_missing_res: float = 5.0  # HP penalty per missing resource
    land_resource_drain: float = 1.0            # beach, sandy, grassland — drain per step
    no_res_hp_multiplier: float = 2.0           # HP lost per missing resource unit
    forest_hp_gain: float = 5.0
    forest_resource_gain: float = 2.0
    sea_resource_costs: tuple = (3.0, 2.0, 1.5)    # ocean, deep_water, water
    mountain_resource_costs: tuple = (1.5, 3.0)      # rocky, mountains
    hard_mode_resource_drain: float = 0.25
    hard_mode_hp_gain: float = 1.0
    hard_mode_hp_loss: float = 0.5

    # Minimap
    minimap_ray: int = 15
    minimap_max_ray: int = 21        # CNN spatial dim = 2*max_ray+1 = 43
    minimap_occlude: bool = False
    minimap_clear_tolerance: float = 0.1

    # Episode limits
    max_steps: int = 1000

    # Reward coefficients
    reward_dist_coef: float = 0.35
    reward_reach_bonus: float = 12.0
    reward_death_penalty: float = -8.0
    reward_time_penalty: float = -0.1
    reward_hp_coef: float = 0.05
    reward_hp_thresh: float = 35.0
    reward_resource_coef: float = 0.02
    reward_resource_thresh: float = 25.0

    # Custom map support
    map_name: str = ""
    spawn_r: int = -1
    spawn_c: int = -1
    target_r: int = -1
    target_c: int = -1

    # Map pool (Level Replay)
    map_pool_size: int = 16

    # Device
    device: str = "auto"

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @classmethod
    def from_hydra(cls, cfg) -> "EnvConfig":
        """Build EnvConfig from a Hydra DictConfig (cfg.env + cfg.device)."""
        env = cfg.env
        return cls(
            size=env.size, scale=env.scale, octaves=env.octaves,
            persistence=env.persistence, lacunarity=env.lacunarity,
            seed=env.seed, detailed_ocean=env.detailed_ocean,
            filtering=env.filtering, sink_mode=env.sink_mode,
            init_hp=env.init_hp, max_hp=env.max_hp,
            init_resources=env.init_resources,
            max_resources=env.get("max_resources", 100.0),
            hard_mode=env.hard_mode,
            # Terrain effects — .get() so older config files still work
            passive_heal_rate=env.get("passive_heal_rate", 1.0),
            land_to_water_resource_cost=env.get("land_to_water_resource_cost", 10.0),
            land_to_water_hp_per_missing_res=env.get("land_to_water_hp_per_missing_res", 5.0),
            land_resource_drain=env.get("land_resource_drain", 1.0),
            no_res_hp_multiplier=env.get("no_res_hp_multiplier", 8.0),
            forest_hp_gain=env.get("forest_hp_gain", 10.0),
            forest_resource_gain=env.get("forest_resource_gain", 2.0),
            sea_resource_costs=tuple(env.get("sea_resource_costs", [1.0, 0.75, 0.3])),
            mountain_resource_costs=tuple(env.get("mountain_resource_costs", [0.5, 1.0])),
            hard_mode_resource_drain=env.get("hard_mode_resource_drain", 0.25),
            hard_mode_hp_gain=env.get("hard_mode_hp_gain", 1.0),
            hard_mode_hp_loss=env.get("hard_mode_hp_loss", 0.5),
            minimap_ray=env.minimap_ray,
            minimap_max_ray=env.get("minimap_max_ray", 10),
            minimap_occlude=env.minimap_occlude,
            minimap_clear_tolerance=env.get("minimap_clear_tolerance", env.get("minimap_min_clear_lv", 0.1)),
            max_steps=env.max_steps,
            reward_dist_coef=env.reward_dist_coef,
            reward_reach_bonus=env.reward_reach_bonus,
            reward_death_penalty=env.reward_death_penalty,
            reward_time_penalty=env.reward_time_penalty,
            reward_hp_coef=env.reward_hp_coef,
            reward_hp_thresh=env.reward_hp_thresh,
            reward_resource_coef=env.get("reward_resource_coef", 0.02),
            reward_resource_thresh=env.get("reward_resource_thresh", 25.0),
            map_name=env.get("map_name", ""),
            spawn_r=env.get("spawn_r", -1),
            spawn_c=env.get("spawn_c", -1),
            target_r=env.get("target_r", -1),
            target_c=env.get("target_c", -1),
            device=cfg.device,
            map_pool_size=env.get("map_pool_size", 16),
        )
