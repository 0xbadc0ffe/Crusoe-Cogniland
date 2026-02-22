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
    minimap: torch.Tensor         # [B, 1, 2*ray+1, 2*ray+1] float (channel-first for CNN)
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
    init_hp: float = 75.0
    max_hp: float = 100.0
    init_resources: float = 0.0
    max_sea_movement_without_resources: int = 7
    hard_mode: bool = False

    # Minimap
    minimap_ray: int = 25
    minimap_occlude: bool = False
    minimap_min_clear_lv: float = 0.25

    # Episode limits
    max_steps: int = 1000

    # Reward coefficients
    reward_dist_coef: float = 1.0
    reward_reach_bonus: float = 10.0
    reward_death_penalty: float = -5.0
    reward_time_penalty: float = -0.01
    reward_hp_coef: float = 0.01
    reward_hp_thresh: float = 30.0

    # Device
    device: str = "auto"

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
