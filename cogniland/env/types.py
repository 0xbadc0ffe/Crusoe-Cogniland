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
    compass: torch.Tensor         # [B, 2] float
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
    size: int = 20
    scale: float = 0.33
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    seed: int = 42
    detailed_ocean: bool = True
    filtering: str = "square"   # "circle", "square", "diamond"
    sink_mode: int = 1          # 0=none, 1, 2

    # Agent
    init_hp: float = 75.0
    max_hp: float = 100.0
    init_resources: float = 0.0
    max_resources: float = 100.0
    max_sea_movement_without_resources: int = 3
    hard_mode: bool = False

    # Terrain effects (previously hardcoded in core.py)
    passive_heal_rate: float = 1.0
    land_to_water_penalty: float = 3.0
    forest_hp_gain: float = 4.0
    forest_resource_gain: float = 1.0
    sea_resource_costs: tuple = (0.75, 0.50, 0.25)   # ocean, deep_water, water
    sea_hp_costs: tuple = (25.0, 25.0, 10.0)          # ocean, deep_water, water
    mountain_resource_costs: tuple = (0.25, 0.75)      # rocky, mountains
    mountain_hp_costs: tuple = (5.0, 20.0)             # rocky, mountains
    hard_mode_resource_drain: float = 0.25
    hard_mode_hp_gain: float = 1.0
    hard_mode_hp_loss: float = 0.5

    # Minimap
    minimap_ray: int = 5
    minimap_max_ray: int = 3        # fixed CNN spatial dim = 2*max_ray+1 = 7
    minimap_occlude: bool = False
    minimap_min_clear_lv: float = 0.25

    # Episode limits
    max_steps: int = 150

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
            max_sea_movement_without_resources=env.max_sea_movement_without_resources,
            hard_mode=env.hard_mode,
            # Terrain effects — .get() so older config files still work
            passive_heal_rate=env.get("passive_heal_rate", 1.0),
            land_to_water_penalty=env.get("land_to_water_penalty", 3.0),
            forest_hp_gain=env.get("forest_hp_gain", 4.0),
            forest_resource_gain=env.get("forest_resource_gain", 1.0),
            sea_resource_costs=tuple(env.get("sea_resource_costs", [0.75, 0.50, 0.25])),
            sea_hp_costs=tuple(env.get("sea_hp_costs", [25.0, 25.0, 10.0])),
            mountain_resource_costs=tuple(env.get("mountain_resource_costs", [0.25, 0.75])),
            mountain_hp_costs=tuple(env.get("mountain_hp_costs", [5.0, 20.0])),
            hard_mode_resource_drain=env.get("hard_mode_resource_drain", 0.25),
            hard_mode_hp_gain=env.get("hard_mode_hp_gain", 1.0),
            hard_mode_hp_loss=env.get("hard_mode_hp_loss", 0.5),
            minimap_ray=env.minimap_ray,
            minimap_max_ray=env.get("minimap_max_ray", 10),
            minimap_occlude=env.minimap_occlude,
            minimap_min_clear_lv=env.minimap_min_clear_lv,
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
        )
