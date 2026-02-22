"""PPO ActorCritic with CNN minimap encoder + MLP scalar encoder.

Follows CleanRL conventions: orthogonal init, get_value / get_action_and_value.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """PPO agent with separate CNN (minimap) and MLP (scalars) encoders."""

    def __init__(
        self,
        scalar_dim: int = 6,
        minimap_channels: int = 1,
        minimap_size: int = 51,  # 2*25+1
        hidden_dim: int = 128,
        action_dim: int = 5,
    ):
        super().__init__()

        # --- CNN encoder for minimap ---
        self.cnn = nn.Sequential(
            _layer_init(nn.Conv2d(minimap_channels, 16, kernel_size=5, stride=2)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=2)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, minimap_channels, minimap_size, minimap_size)
            cnn_out_dim = self.cnn(dummy).shape[1]

        # --- MLP encoder for scalars ---
        self.scalar_encoder = nn.Sequential(
            _layer_init(nn.Linear(scalar_dim, hidden_dim)),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )

        combined_dim = cnn_out_dim + hidden_dim

        # --- Shared trunk ---
        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(combined_dim, hidden_dim)),
            nn.ReLU(),
        )

        # --- Actor head ---
        self.actor = _layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

        # --- Critic head ---
        self.critic = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        cnn_features = self.cnn(obs["minimap"])
        scalar_features = self.scalar_encoder(obs["scalars"])
        combined = torch.cat([cnn_features, scalar_features], dim=1)
        return self.trunk(combined)

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(self._encode(obs)).squeeze(-1)

    def get_action_and_value(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob, entropy, value)."""
        features = self._encode(obs)
        logits = self.actor(features)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)

        return action, log_prob, entropy, value
