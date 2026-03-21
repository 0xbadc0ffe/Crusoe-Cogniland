"""Batched training wrapper around the Islands environment."""

from __future__ import annotations

import torch

from cogniland.env.islands import Islands
from cogniland.env.types import CompiledTerrainData, CurriculumStage, EnvConfig, EnvState


class BatchedIslandEnv:
    """Batched wrapper used by the training loop.

    Handles auto-reset of done environments and provides observations
    as a dict with ``"scalars"`` and ``"minimap"`` keys.
    """

    def __init__(
        self,
        config: EnvConfig,
        num_envs: int,
        world_maps: torch.Tensor | None = None,
        map_pool_size: int = 16,
        curriculum_easy_radius: int = 40,
    ):
        self.config = config
        self.num_envs = num_envs
        self.env = Islands(
            config,
            world_maps=world_maps,
            map_pool_size=map_pool_size,
            curriculum_easy_radius=curriculum_easy_radius,
        )
        self.compiled = self.env.compiled
        self.state: EnvState | None = None
        self.target_pos: torch.Tensor | None = None
        self.step_count: torch.Tensor | None = None
        self._device = config.resolved_device()
        self._curriculum_stage = CurriculumStage.NORMAL

        # Track episode stats
        self.episode_rewards: torch.Tensor | None = None
        self.episode_lengths: torch.Tensor | None = None

    def set_curriculum_stage(self, stage: CurriculumStage) -> None:
        self._curriculum_stage = stage

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        self.state, self.target_pos = self.env.reset(
            self.num_envs, seed=seed, curriculum_stage=self._curriculum_stage
        )
        self.step_count = torch.zeros(self.num_envs, device=self._device)
        self.episode_rewards = torch.zeros(self.num_envs, device=self._device)
        self.episode_lengths = torch.zeros(self.num_envs, device=self._device)
        return self.get_obs()

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """Returns (obs, reward, done, info)."""
        result = self.env.step(self.state, action, self.target_pos)
        self.state = result.state
        self.step_count += 1

        reward = result.reward

        # Track episode stats
        self.episode_rewards += reward
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
            self.state, self.target_pos = self.env.reset_done(
                self.state, self.target_pos, done,
                curriculum_stage=self._curriculum_stage,
            )
            self.step_count[done] = 0

        return self.get_obs(), reward, done, info


    def get_obs(self) -> dict[str, torch.Tensor]:
        """Build observation dict from current state.

        Returns:
            ``"scalars"``: [B, 5] — compass_dir(2) unit vector, terrain_idx, resources, hp
            ``"minimap"``: [B, 2, H, W]
        """
        s = self.state
        num_terrains = self.compiled.num_terrains
        scalars = torch.stack([
            s.compass[:, 0],
            s.compass[:, 1],
            s.terrain_idx / max(num_terrains - 1, 1),
            s.resources / self.config.max_resources,
            s.hp / self.config.max_hp,
        ], dim=1)  # [B, 5]
        return {"scalars": scalars, "minimap": s.minimap}
