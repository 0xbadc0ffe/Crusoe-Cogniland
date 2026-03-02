"""Training wrappers around the Islands environment.

BatchedIslandEnv  — efficient batched wrapper for PPO rollout collection.
IslandNavEnv      — single-instance Gymnasium wrapper for compatibility.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from cogniland.env.constants import NUM_ACTIONS, TERRAIN_VISIBILITY
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
            ``"scalars"``: [B, 7] — compass(2), terrain_lev, terrain_clock, resources, hp, visibility_range
            ``"minimap"``: [B, 2, H, W]
        """
        s = self.state
        vis_range = TERRAIN_VISIBILITY.to(s.terrain_lev.device)[s.terrain_lev.long()].float()
        vis_norm = vis_range / self.config.minimap_max_ray  # normalize to [0, 1]
        compass_norm = s.compass / self.config.size          # normalize to [-1, 1]
        scalars = torch.stack([
            compass_norm[:, 0],
            compass_norm[:, 1],
            s.terrain_lev,
            s.terrain_clock,
            s.resources,
            s.hp,
            vis_norm,
        ], dim=1)  # [B, 7]
        return {"scalars": scalars, "minimap": s.minimap}


class IslandNavEnv(gym.Env):
    """Single-instance Gymnasium wrapper for standard RL library compatibility."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = EnvConfig(**kwargs)
        self.config = config
        self.env = Islands(config)
        self.render_mode = render_mode

        ray = config.minimap_max_ray
        diameter = 2 * ray + 1

        self.observation_space = gym.spaces.Dict({
            "scalars": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "minimap": gym.spaces.Box(0.0, 1.0, shape=(2, diameter, diameter), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        self._state: EnvState | None = None
        self._target: torch.Tensor | None = None
        self._steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._state, self._target = self.env.reset(1, seed=seed)
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        act_tensor = torch.tensor([action], device=self._state.position.device)
        result = self.env.step(self._state, act_tensor, self._target)
        self._state = result.state
        self._steps += 1

        terminated = result.done[0].item()
        truncated = self._steps >= self.config.max_steps
        reward = result.reward[0].item()

        info = {k: v[0].item() if isinstance(v, torch.Tensor) else v for k, v in result.info.items()}
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> dict[str, np.ndarray]:
        s = self._state
        vis_range = TERRAIN_VISIBILITY[s.terrain_lev[0].long().item()].float()
        vis_norm = vis_range / self.config.minimap_max_ray
        compass_norm = s.compass / self.config.size  # normalize to [-1, 1]
        scalars = torch.stack([
            compass_norm[0, 0], compass_norm[0, 1],
            s.terrain_lev[0], s.terrain_clock[0],
            s.resources[0], s.hp[0],
            vis_norm,
        ]).cpu().numpy().astype(np.float32)
        minimap = s.minimap[0].cpu().numpy().astype(np.float32)
        return {"scalars": scalars, "minimap": minimap}
