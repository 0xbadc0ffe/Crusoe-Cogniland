"""Pure functions for computing per-episode behavioral metrics from raw episode tensors.

Each function takes the tensors collected during the episode loop and returns a scalar
tensor of shape [n_episodes].  No side effects, no WandB calls.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_directness(
    final_cost: torch.Tensor,           # [N] accumulated terrain cost
    dijkstra_to_final: torch.Tensor,    # [N] optimal spawn→final_position cost
) -> torch.Tensor:
    """D = C_agent / (C_agent - C_dijkstra_partial), capped at 100.

    100 = agent moved as efficiently as possible to wherever it ended up.
    ~2  = agent used roughly twice the optimal terrain cost.
    """
    return torch.where(
        final_cost > dijkstra_to_final + 1e-6,
        (final_cost / (final_cost - dijkstra_to_final)).clamp(max=100.0),
        torch.full_like(final_cost, 100.0),
    )


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


def read_dijkstra_to_final(
    dist_maps: list[np.ndarray],    # one [H, W] dist map per episode
    final_positions: torch.Tensor,  # [N, 2]
    device: str,
) -> torch.Tensor:
    """Read spawn→final_position distance from pre-computed Dijkstra dist maps."""
    final_pos_cpu = final_positions.cpu()
    return torch.tensor([
        dist_maps[i][final_pos_cpu[i, 0].item(), final_pos_cpu[i, 1].item()]
        for i in range(len(dist_maps))
    ], dtype=torch.float32, device=device)
