"""Multi-task wrapper for the base environment.

Wraps the base environment, delegates step/reset, applies task-specific
rewards, and exposes task embeddings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cogniland.envs.env import CognilandEnv
from cogniland.envs.task_sampler import TaskSampler
from cogniland.envs.tasks import (
    _TASK_BIOME_QUESTION,
    _TASK_CRAFT_TOOL,
    compute_task_reward,
)


class MultiTaskEnvWrapper:
    """Wraps CognilandEnv with multi-task reward computation and embeddings.

    Attributes:
        env: The underlying CognilandEnv
        task_ids: int array [B] — current task per env
    """

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

        # Task embeddings: random orthogonal vectors (fixed at init)
        rng = np.random.default_rng(12345)
        raw = rng.standard_normal((num_tasks, task_embedding_dim)).astype(np.float32)
        # Orthogonalize via QR if possible
        if num_tasks <= task_embedding_dim:
            q, _ = np.linalg.qr(raw.T)
            self._task_embeddings = q.T[:num_tasks]
        else:
            # More tasks than dims — just normalize
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            self._task_embeddings = raw / np.maximum(norms, 1e-8)

        # Current task assignment
        self.task_ids = np.zeros(env.num_envs, dtype=np.int32)

        # Running per-env episode-return sum over the task-computed rewards.
        # The base env's `_episode_returns` is never accumulated (base rewards
        # are zero), so we track it here and overwrite `info` on episode end.
        self._episode_returns = np.zeros(env.num_envs, dtype=np.float32)

        # Which tool (1=raft, 2=rope, 3=shoes) was crafted at any point during
        # the current episode. The base env's ``crafted`` info is non-zero
        # only on the single step the craft occurred, so tasks 4-6 success
        # needs this persistent flag. Resets on episode end.
        self._episode_crafted_tool = np.zeros(env.num_envs, dtype=np.int32)

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def observation_space(self) -> dict:
        return self.env.observation_space()

    def action_space(self) -> int:
        return self.env.action_space()

    def set_tasks(self, task_ids: np.ndarray) -> None:
        """Set task IDs for all envs."""
        self.task_ids = np.asarray(task_ids, dtype=np.int32)

    def get_task_embeddings(self, task_ids: np.ndarray) -> np.ndarray:
        """Look up task embeddings.

        Args:
            task_ids: int array [B]

        Returns:
            float32 array [B, task_embedding_dim]
        """
        return self._task_embeddings[task_ids]

    def reset(
        self,
        seed: int | None = None,
        map_indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Reset all envs and return observations."""
        obs = self.env.reset(seed=seed, map_indices=map_indices)
        self._episode_returns.fill(0.0)
        self._episode_crafted_tool.fill(0)
        # Add task embedding to observations
        obs["task_embedding"] = self.get_task_embeddings(self.task_ids)
        return obs

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
        """Step all envs and apply task-specific rewards.

        Returns:
            (obs_dict, rewards, dones, info)
        """
        obs, base_rewards, dones, info = self.env.step(actions)

        # Compute task-specific rewards
        rewards = compute_task_reward(
            self.task_ids, base_rewards, dones, info, self._config,
        )

        # Track the tool crafted at any point during the episode.
        crafted_this_step = info.get("crafted")
        if crafted_this_step is not None:
            self._episode_crafted_tool = np.maximum(
                self._episode_crafted_tool, crafted_this_step.astype(np.int32)
            )

        # Per-episode task success flag (binary, per-env). Only meaningful on
        # steps where ``returned_episode`` is True; consumers must mask.
        info["task_success"] = self._compute_task_success(info)

        # Accumulate the task-computed reward into our running return tracker,
        # then overwrite info["returned_episode_returns"] for envs that just
        # finished (the base env reports zeros there).
        self._episode_returns += rewards
        returned = info.get("returned_episode")
        if returned is not None and returned.any():
            info["returned_episode_returns"] = np.where(
                returned, self._episode_returns, 0.0
            ).astype(np.float32)
        if dones.any():
            self._episode_returns[dones] = 0.0
            self._episode_crafted_tool[dones] = 0

        # Add task embedding to observations
        obs["task_embedding"] = self.get_task_embeddings(self.task_ids)

        # Update info with task rewards for episode tracking
        info["task_rewards"] = rewards

        return obs, rewards, dones, info

    # ------------------------------------------------------------------ #
    def _compute_task_success(self, info: dict[str, Any]) -> np.ndarray:
        """Per-env binary success flag based on task definition.

          - Task 0:    reached YES or NO target
          - Tasks 1-3: reached the target matching the biome question
          - Tasks 4-6: the task-specific tool was crafted during the episode
        """
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
