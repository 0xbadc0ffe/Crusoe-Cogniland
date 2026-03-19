"""Immutable state types for the environment.

NamedTuples are chosen for JAX compatibility (they are pytrees).
`state._replace(hp=new_hp)` creates a new state without mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import torch


class CurriculumStage(str, Enum):
    """Training curriculum stage controlling spawn/target sampling."""
    EASY   = "easy"    # Spawn + target constrained to radius-50 circle around map center
    NORMAL = "normal"  # Spawn + target sampled uniformly over land cells


class EnvState(NamedTuple):
    """Full batched environment state."""

    position: torch.Tensor        # [B, 2] long
    minimap: torch.Tensor         # [B, 3, 2*max_ray+1, 2*max_ray+1] float (ch0=heightmap, ch1=target indicator, ch2=visibility mask)
    compass: torch.Tensor         # [B, 2] float — unit direction vector (position − target) / dist
    terrain_idx: torch.Tensor     # [B] float
    resources: torch.Tensor       # [B] float
    hp: torch.Tensor              # [B] float
    cost: torch.Tensor            # [B] float — accumulated terrain time cost this episode
    dijkstra_cost: torch.Tensor   # [B] float — optimal time cost spawn→target (constant per episode)


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
    filtering: str = "square"   # "circle", "square", "diamond"
    sink_mode: int = 1          # 0=none, 1, 2

    # Agent
    init_hp: float = 100.0
    max_hp: float = 100.0
    init_resources: float = 50.0
    max_resources: float = 100.0

    # Terrain effects
    land_to_water_resource_cost: float = 20.0  # boat construction cost (resources)
    land_resource_drain: float = 1.5            # beach, sandy, grassland — drain per step
    no_res_hp_multiplier: float = 2.0           # HP lost per missing resource unit
    forest_hp_gain: float = 8.0
    forest_resource_gain: float = 5.0
    sea_resource_costs: tuple = (0.7, 0.5, 0.3)    # ocean, deep_water, water
    mountain_resource_costs: tuple = (2.0, 5.0)      # rocky, mountains

    # Minimap
    minimap_max_ray: int = 22        # CNN spatial dim = 2*max_ray+1 = 45
    minimap_occlude: bool = False
    minimap_clear_tolerance: float = 0.1

    # Episode limits
    max_steps: int = 1000

    # Reward coefficients
    lambda_p: float = 0.1           # progress reward weight
    lambda_t: float = 60.0          # time-efficiency bonus weight
    lambda_d: float = 0.6           # death penalty coefficient (fraction of reach bonus)
    reward_reach_bonus: float = 100.0

    # Custom map support
    map_name: str = ""
    spawn_r: int = -1
    spawn_c: int = -1
    target_r: int = -1
    target_c: int = -1

    # Map pool (Level Replay)
    map_pool_size: int = 16

    # Dataset and curriculum
    dataset_path: str = ""
    curriculum_switch_steps: int = 0   # 0 = disabled; switch EASY→NORMAL at this global_step
    curriculum_easy_radius: int = 50   # radius (cells) around center for EASY spawn/target

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
            seed=env.seed,
            filtering=env.filtering, sink_mode=env.sink_mode,
            init_hp=env.init_hp, max_hp=env.max_hp,
            init_resources=env.init_resources,
            max_resources=env.get("max_resources", 100.0),
            land_to_water_resource_cost=env.get("land_to_water_resource_cost", 20.0),
            land_resource_drain=env.get("land_resource_drain", 1.5),
            no_res_hp_multiplier=env.get("no_res_hp_multiplier", 2.0),
            forest_hp_gain=env.get("forest_hp_gain", 8.0),
            forest_resource_gain=env.get("forest_resource_gain", 5.0),
            sea_resource_costs=tuple(env.get("sea_resource_costs", [0.7, 0.5, 0.3])),
            mountain_resource_costs=tuple(env.get("mountain_resource_costs", [2.0, 5.0])),
            minimap_max_ray=env.get("minimap_max_ray", 22),
            minimap_occlude=env.minimap_occlude,
            minimap_clear_tolerance=env.get("minimap_clear_tolerance", env.get("minimap_min_clear_lv", 0.1)),
            max_steps=env.max_steps,
            lambda_p=env.get("lambda_p", 0.1),
            lambda_t=env.get("lambda_t", 60.0),
            lambda_d=env.get("lambda_d", 0.6),
            reward_reach_bonus=env.reward_reach_bonus,
            map_name=env.get("map_name", ""),
            spawn_r=env.get("spawn_r", -1),
            spawn_c=env.get("spawn_c", -1),
            target_r=env.get("target_r", -1),
            target_c=env.get("target_c", -1),
            device=cfg.device,
            map_pool_size=env.get("map_pool_size", 16),
            dataset_path=env.get("dataset_path", ""),
            curriculum_switch_steps=env.get("curriculum_switch_steps", 0),
            curriculum_easy_radius=env.get("curriculum_easy_radius", 50),
        )
