"""Pure functions for computing per-episode behavioral metrics from raw episode tensors.

Each function takes the tensors collected during the episode loop and returns a scalar
tensor of shape [n_episodes].  No side effects, no WandB calls.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_directness(
    initial_spawns: torch.Tensor,    # [N, 2] spawn positions
    final_positions: torch.Tensor,   # [N, 2] agent final positions
    total_moves: torch.Tensor,       # [N] number of steps taken
) -> torch.Tensor:
    """D = manhattan(spawn, final) / n_steps.

    Measures how directly the agent moved toward its final position,
    independently of terrain costs.  ~1 = nearly straight path,
    lower values = more detours.

    Range: (0, 1].
    """
    manhattan = (initial_spawns - final_positions).abs().sum(dim=1).float()
    return (manhattan / total_moves.clamp(min=1)).clamp(max=1.0)


def compute_risk_exposure(
    risk_sum: torch.Tensor,    # [N] accumulated per-step risk values
    risk_count: torch.Tensor,  # [N] number of steps accumulated
) -> torch.Tensor:
    """Mean per-step risk: drain_t / (resources_t + hp_t / 2).

    > 1.0 → average step drains more than the combined HP+resource buffer.
    < 1.0 → agent is comfortably provisioned on average.
    """
    return risk_sum / risk_count.clamp(min=1)


def compute_danger_fraction(
    danger_steps: torch.Tensor,  # [N]
    total_moves: torch.Tensor,   # [N]
) -> torch.Tensor:
    """Fraction of episode steps spent with HP below the danger threshold."""
    return danger_steps / total_moves.clamp(min=1)


def compute_exploration(
    observed: torch.Tensor,  # [N, H, W] bool
) -> torch.Tensor:
    """Fraction of map cells ever observed during the episode."""
    H, W = observed.shape[1], observed.shape[2]
    return observed.sum(dim=(1, 2)).float() / (H * W)


def compute_terrain_visit_fractions(
    terrain_visits: torch.Tensor,  # [N, 9]
) -> torch.Tensor:
    """Per-terrain visit fractions, normalised to sum to 1 per episode."""
    visit_totals = terrain_visits.sum(dim=1, keepdim=True).clamp(min=1)
    return terrain_visits / visit_totals  # [N, 9]


