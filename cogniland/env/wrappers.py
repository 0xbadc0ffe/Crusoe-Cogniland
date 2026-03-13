"""Batched training wrapper around the Islands environment."""

from __future__ import annotations

import torch

from cogniland.env.constants import TERRAIN_VISIBILITY
from cogniland.env.islands import Islands
from cogniland.env.types import EnvConfig, EnvState


class BatchedIslandEnv:
    """Batched wrapper used by the training loop.

    Handles auto-reset of done environments and provides observations
    as a dict with ``"scalars"`` and ``"minimap"`` keys.
    """

    def __init__(self, config: EnvConfig, num_envs: int):
        self.config = config
        self.num_envs = num_envs
        self.env = Islands(config)
        self.state: EnvState | None = None
        self.target_pos: torch.Tensor | None = None
        self.step_count: torch.Tensor | None = None
        self._device = config.resolved_device()

        # Track episode stats
        self.episode_rewards: torch.Tensor | None = None
        self.episode_lengths: torch.Tensor | None = None

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        self.state, self.target_pos = self.env.reset(self.num_envs, seed=seed)
        self.step_count = torch.zeros(self.num_envs, device=self._device)
        self.episode_rewards = torch.zeros(self.num_envs, device=self._device)
        self.episode_lengths = torch.zeros(self.num_envs, device=self._device)
        return self.get_obs()

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """Returns (obs, reward, done, info)."""
        result = self.env.step(self.state, action, self.target_pos)
        self.state = result.state
        self.step_count += 1

        # Track episode stats
        self.episode_rewards += result.reward
        self.episode_lengths += 1

        # Truncation check
        truncated = self.step_count >= self.config.max_steps
        done = result.done | truncated

        # Collect episode stats before reset
        info = dict(result.info)
        if done.any():
            info["final_rewards"] = self.episode_rewards[done].clone()
            info["final_lengths"] = self.episode_lengths[done].clone()
            info["final_reached"] = result.info["reached"][done].clone()

            # Reset episode tracking for done envs
            self.episode_rewards[done] = 0.0
            self.episode_lengths[done] = 0.0

        # Auto-reset done environments
        if done.any():
            self.state, self.target_pos = self.env.reset_done(self.state, self.target_pos, done)
            self.step_count[done] = 0

        return self.get_obs(), result.reward, done, info

    def get_obs(self) -> dict[str, torch.Tensor]:
        """Build observation dict from current state.

        Returns:
            ``"scalars"``: [B, 7] — compass_dir(2) unit vector, terrain_lev, terrain_clock, resources, hp, visibility_range
            ``"minimap"``: [B, 2, H, W]
        """
        s = self.state
        vis_range = TERRAIN_VISIBILITY.to(s.terrain_lev.device)[s.terrain_lev.long()].float()
        vis_norm = vis_range / self.config.minimap_max_ray  # normalize to [0, 1]
        scalars = torch.stack([
            s.compass[:, 0],
            s.compass[:, 1],
            s.terrain_lev / 8.0,
            s.terrain_clock / 10.0,
            s.resources / self.config.max_resources,
            s.hp / self.config.max_hp,
            vis_norm,
        ], dim=1)  # [B, 7]
        return {"scalars": scalars, "minimap": s.minimap}
