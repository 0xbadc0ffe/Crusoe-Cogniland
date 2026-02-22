"""Behavioral metrics for policy identification (conservative vs greedy)."""

from __future__ import annotations

import torch

from cogniland.env.types import EnvState


def compute_behavioral_metrics(
    terrain_visits: torch.Tensor,
    total_moves: torch.Tensor,
    final_state: EnvState,
) -> dict[str, float]:
    """Compute behavioral metrics averaged over evaluation episodes.

    Args:
        terrain_visits: [num_episodes, 9] count of visits to each terrain type
        total_moves: [num_episodes] moves per episode
        final_state: final EnvState from evaluation

    Returns:
        Dict of behavioral/ prefixed metrics.
    """
    moves_safe = total_moves.clamp(min=1).unsqueeze(1)  # [N, 1]
    visit_pct = terrain_visits / moves_safe  # [N, 9]

    # Water percentage (terrains 0-2)
    water_pct = visit_pct[:, 0:3].sum(dim=1).mean().item()

    # Forest percentage (terrain 6)
    forest_pct = visit_pct[:, 6].mean().item()

    # Mountain percentage (terrains 7-8)
    mountain_pct = visit_pct[:, 7:9].sum(dim=1).mean().item()

    # Mean resources held at end
    mean_resource_held = final_state.resources.mean().item()

    # HP management score: mean HP / max_HP
    hp_score = (final_state.hp / 100.0).mean().item()

    # Directness ratio: compass distance / moves (higher = more direct)
    compass_dist = torch.norm(final_state.compass.float(), dim=1)
    directness = (compass_dist / total_moves.clamp(min=1)).mean().item()

    # Land percentage (terrains 3-5)
    land_pct = visit_pct[:, 3:6].sum(dim=1).mean().item()

    return {
        "behavioral/water_pct": water_pct,
        "behavioral/land_pct": land_pct,
        "behavioral/forest_pct": forest_pct,
        "behavioral/mountain_pct": mountain_pct,
        "behavioral/resource_held": mean_resource_held,
        "behavioral/hp_score": hp_score,
        "behavioral/directness": directness,
    }
