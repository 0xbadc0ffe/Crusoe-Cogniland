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
) -> torch.Tensor:
    """Normalised entropy of the visibility distribution.

    E = -1/log(H*W) * Σ p(x,y) log p(x,y)   over observed cells

    where p(x,y) = n(x,y) / Σ n.  Range [0, 1].
    High = broad visual attention; low = concentrated on a narrow region.
    """
    N, H, W = vis_counts.shape
    flat = vis_counts.view(N, -1).float()          # [N, H*W]
    total = flat.sum(dim=1, keepdim=True).clamp(min=1)  # [N, 1]
    p = flat / total                                # [N, H*W]
    log_p = torch.where(p > 0, p.log(), torch.zeros_like(p))
    entropy = -(p * log_p).sum(dim=1)              # [N]
    log_hw = np.log(H * W)
    return entropy / log_hw


def compute_terrain_visit_fractions(
    terrain_visits: torch.Tensor,  # [N, 9]
) -> torch.Tensor:
    """Per-terrain visit fractions, normalised to sum to 1 per episode."""
    visit_totals = terrain_visits.sum(dim=1, keepdim=True).clamp(min=1)
    return terrain_visits / visit_totals  # [N, 9]


