"""Compass-following baseline — always picks the move that minimises
Manhattan distance to the target based on the compass observation."""

from __future__ import annotations

import torch
import torch.nn as nn

from cogniland.env.constants import NUM_ACTIONS


class CompassAgent(nn.Module):
    """Greedy compass follower.

    Reads compass_y (scalars[:,0]) and compass_x (scalars[:,1]) which encode
    ``position - target``.  Picks the action that reduces ``|compass_y| +
    |compass_x|`` the most.  Ties broken by preferring vertical movement.
    """

    def __init__(self, action_dim: int = NUM_ACTIONS):
        super().__init__()
        self.action_dim = action_dim
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        # ACTION_DELTAS: up=[-1,0], down=[1,0], right=[0,1], left=[0,-1], stay=[0,0]
        self.register_buffer(
            "_deltas",
            torch.tensor([[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]], dtype=torch.float32),
        )

    def get_action_and_value(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = obs["scalars"].shape[0]
        device = obs["scalars"].device

        if action is None:
            compass = obs["scalars"][:, :2]  # [B, 2]  (dy_to_target, dx_to_target)

            # Manhattan distance after each possible action:
            # new_compass = compass + delta  (since compass = pos - target, moving by delta adds delta)
            # manhattan = |new_compass_y| + |new_compass_x|
            new_compass = compass.unsqueeze(1) + self._deltas.unsqueeze(0)  # [B, 5, 2]
            manhattan = new_compass.abs().sum(dim=2)  # [B, 5]
            action = manhattan.argmin(dim=1)  # [B]

        log_prob = torch.zeros(batch_size, device=device)
        entropy = torch.zeros(batch_size, device=device)
        value = torch.zeros(batch_size, device=device)
        return action, log_prob, entropy, value

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.zeros(obs["scalars"].shape[0], device=obs["scalars"].device)
