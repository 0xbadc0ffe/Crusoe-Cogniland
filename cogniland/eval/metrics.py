"""Pure functions for computing per-episode behavioral metrics from raw episode tensors.

Each function takes the tensors collected during the episode loop and returns a scalar
tensor of shape [n_episodes].  No side effects, no WandB calls.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_directness(
    dijkstra_cost: torch.Tensor,  # [N] optimal traversal time (move_cost Dijkstra)
    agent_cost: torch.Tensor,     # [N] actual cumulative traversal time
) -> torch.Tensor:
    """Time-efficiency ratio: T = time* / t_agent.

    Range [0, 1].  1 = agent matched the shortest-time path; lower = detours,
    backtracking, or foraging reduced time efficiency.
    """
    return torch.clamp(dijkstra_cost / agent_cost.clamp(min=1e-6), 0.0, 1.0)


def compute_risk_exposure(
    drawdown_sq_sum: torch.Tensor,  # [N] accumulated ((u0 - u_t) / u0)^2
    risk_count: torch.Tensor,       # [N] number of steps accumulated
) -> torch.Tensor:
    """Ulcer-Index-style risk exposure: RMS of relative drawdowns.

    ρ = sqrt( (1/T) Σ ((u0 - u_t) / u0)^2 )

    where u_t = res_t + hp_t is the survival budget and u0 = init budget.
    Range [0, 1].  Low = healthy budget throughout; high = prolonged or acute depletion.
    """
    return (drawdown_sq_sum / risk_count.clamp(min=1)).sqrt()


def compute_danger_fraction(
    danger_steps: torch.Tensor,  # [N]
    total_moves: torch.Tensor,   # [N]
) -> torch.Tensor:
    """Fraction of episode steps spent with HP below the danger threshold."""
    return danger_steps / total_moves.clamp(min=1)


def compute_exploration(
    vis_counts: torch.Tensor,  # [N, H, W] int — per-cell visibility counts n(x,y)
    land_mask: torch.Tensor,   # [N, H, W] bool — True for land cells
) -> torch.Tensor:
    """Coverage: fraction of land cells observed at least once.

    C = |C_obs ∩ L| / |L|

    where L is the set of land cells and C_obs is the set of cells observed
    at least once during the episode.  Range [0, 1].
    """
    observed = vis_counts > 0                              # [N, H, W]
    land_observed = (observed & land_mask).view(vis_counts.shape[0], -1).sum(dim=1).float()
    land_total = land_mask.view(vis_counts.shape[0], -1).sum(dim=1).float().clamp(min=1)
    return land_observed / land_total


def compute_terrain_visit_fractions(
    terrain_visits: torch.Tensor,  # [N, 9]
) -> torch.Tensor:
    """Per-terrain visit fractions, normalised to sum to 1 per episode."""
    visit_totals = terrain_visits.sum(dim=1, keepdim=True).clamp(min=1)
    return terrain_visits / visit_totals  # [N, 9]


