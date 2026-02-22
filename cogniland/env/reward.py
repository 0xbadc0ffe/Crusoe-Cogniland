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
        r_dist  — dense: encourages moving toward target
        r_reach — sparse: large bonus for reaching target
        r_death — sparse: penalty for dying
        r_time  — dense: small per-step penalty to encourage efficiency
        r_hp    — mild: penalise dangerously low HP
    """
    r_dist = (prev_dist - dist_to_target) * config.reward_dist_coef
    r_reach = torch.where(reached, torch.tensor(config.reward_reach_bonus, device=state.hp.device), torch.tensor(0.0, device=state.hp.device))
    r_death = torch.where(~alive, torch.tensor(config.reward_death_penalty, device=state.hp.device), torch.tensor(0.0, device=state.hp.device))
    r_time = config.reward_time_penalty
    r_hp = config.reward_hp_coef * torch.clamp(config.reward_hp_thresh - state.hp, min=0.0)

    return r_dist + r_reach + r_death + r_time - r_hp
