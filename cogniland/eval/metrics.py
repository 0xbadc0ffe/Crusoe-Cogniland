"""Pure functions for computing per-episode behavioral metrics from raw episode tensors.

Each function takes the tensors collected during the episode loop and returns a scalar
tensor of shape [n_episodes].  No side effects, no WandB calls.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_path_adherence(
    visited_cells: torch.Tensor,     # [N, H, W] bool — unique cells visited by agent
    dijkstra_corridor: torch.Tensor, # [N, H, W] bool — dilated Dijkstra path
) -> torch.Tensor:
    """Fraction of agent's unique visited cells inside the Dijkstra corridor.

    A = |C_agent ∩ D_r(C_dijkstra)| / |C_agent|

    where D_r is a dilation of radius r around the optimal path cells.
    Range: [0, 1].  1 = agent stayed within the corridor, lower = more rerouting.
    """
    overlap = (visited_cells & dijkstra_corridor).sum(dim=(1, 2)).float()
    agent_total = visited_cells.sum(dim=(1, 2)).float().clamp(min=1)
    return overlap / agent_total


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


