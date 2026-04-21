"""Multi-task wrapper: delegates step/reset, applies task reward, injects task embedding."""

from __future__ import annotations

from typing import Any

import numpy as np

from cogniland.envs.env import CognilandEnv
from cogniland.envs.tasks import (
    _TASK_BIOME_QUESTION,
    _TASK_CRAFT_TOOL,
    compute_task_reward,
)


class MultiTaskEnvWrapper:
    """Wraps CognilandEnv with task reward + one-hot task embedding."""

    def __init__(
        self,
        env: CognilandEnv,
        config: Any,
        num_tasks: int = 1,
        task_embedding_dim: int = 7,
    ):
        self.env = env
        self._config = config
        self._num_tasks = num_tasks
        self._task_embedding_dim = task_embedding_dim

        if num_tasks > task_embedding_dim:
            raise ValueError(
                f"one-hot task embedding requires task_embedding_dim "
                f"({task_embedding_dim}) >= num_tasks ({num_tasks})"
            )
        self._task_embeddings = np.eye(task_embedding_dim, dtype=np.float32)[:num_tasks]

        self.task_ids = np.zeros(env.num_envs, dtype=np.int32)

        # Running per-episode sum over task-computed rewards. Base env reports 0.
        self._episode_returns = np.zeros(env.num_envs, dtype=np.float32)

        # Tool crafted at any point during the current episode (for tasks 4-6).
        self._episode_crafted_tool = np.zeros(env.num_envs, dtype=np.int32)

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def observation_space(self) -> dict:
        return self.env.observation_space()

    def action_space(self) -> int:
        return self.env.action_space()

    def set_tasks(self, task_ids: np.ndarray) -> None:
        self.task_ids = np.asarray(task_ids, dtype=np.int32)

    def set_spawn_distance_range(self, lo: int, hi: int) -> None:
        """Passthrough to the base env — used by the trainer's curriculum."""
        self.env.set_spawn_distance_range(lo, hi)

    @property
    def spawn_distance_schedule(self):
        return self.env.spawn_distance_schedule

    def get_task_embeddings(self, task_ids: np.ndarray) -> np.ndarray:
        return self._task_embeddings[task_ids]

    def reset(
        self,
        seed: int | None = None,
        map_indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        obs = self.env.reset(seed=seed, map_indices=map_indices)
        self._episode_returns.fill(0.0)
        self._episode_crafted_tool.fill(0)
        obs["task_embedding"] = self.get_task_embeddings(self.task_ids)
        return obs

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
        obs, base_rewards, dones, info = self.env.step(actions)

        rewards = compute_task_reward(
            self.task_ids, base_rewards, dones, info, self._config,
        )

        crafted_this_step = info.get("crafted")
        if crafted_this_step is not None:
            self._episode_crafted_tool = np.maximum(
                self._episode_crafted_tool, crafted_this_step.astype(np.int32)
            )

        info["task_success"] = self._compute_task_success(info)

        self._episode_returns += rewards
        returned = info.get("returned_episode")
        if returned is not None and returned.any():
            info["returned_episode_returns"] = np.where(
                returned, self._episode_returns, 0.0
            ).astype(np.float32)
        if dones.any():
            self._episode_returns[dones] = 0.0
            self._episode_crafted_tool[dones] = 0

        obs["task_embedding"] = self.get_task_embeddings(self.task_ids)
        info["task_rewards"] = rewards

        return obs, rewards, dones, info

    def _compute_task_success(self, info: dict[str, Any]) -> np.ndarray:
        B = len(self.task_ids)
        success = np.zeros(B, dtype=np.float32)

        reached_yes = info.get("reached_yes")
        reached_no = info.get("reached_no")
        biome = info.get("biome")

        if reached_yes is not None and reached_no is not None:
            reached = reached_yes | reached_no
            success[(self.task_ids == 0) & reached] = 1.0

            if biome is not None:
                for t_id, target_biome in _TASK_BIOME_QUESTION.items():
                    is_match = biome == target_biome
                    correct = (reached_yes & is_match) | (reached_no & ~is_match)
                    success[(self.task_ids == t_id) & correct] = 1.0

        for t_id, tool_id in _TASK_CRAFT_TOOL.items():
            got_tool = self._episode_crafted_tool == tool_id
            success[(self.task_ids == t_id) & got_tool] = 1.0

        return success
