"""Pure-function reward computation.

All coefficients come from EnvConfig so they can be tuned via Hydra / WandB sweeps.
"""

from __future__ import annotations

import torch

from cogniland.env.types import EnvConfig, EnvState


def compute_reward(
    state: EnvState,
    alive: torch.Tensor,
    reached: torch.Tensor,
    dist_to_target: torch.Tensor,
    prev_dist: torch.Tensor,
    config: EnvConfig,
) -> torch.Tensor:
    """Compute per-environment reward.  Pure function.

    Components:
        r_progress — dense: encourages moving toward target (Manhattan distance)
        r_time     — dense: small per-step penalty to discourage idling
        r_success  — sparse: reach bonus + time-efficiency bonus
        r_death    — sparse: proportional penalty for dying
    """
    device = state.hp.device

    r_progress = config.lambda_p * (prev_dist - dist_to_target)
    r_time = -config.time_penalty

    # Time-efficiency ratio: optimal time / actual time, clamped to [0, 1]
    time_ratio = torch.clamp(state.dijkstra_cost / (state.cost + 1e-6), 0.0, 1.0)

    r_success = torch.where(
        reached,
        torch.tensor(config.reward_reach_bonus, device=device) + config.lambda_t * time_ratio,
        torch.zeros(1, device=device),
    )
    r_death = torch.where(
        ~alive,
        torch.tensor(-config.lambda_d * config.reward_reach_bonus, device=device),
        torch.zeros(1, device=device),
    )

    return r_progress + r_time + r_success + r_death
