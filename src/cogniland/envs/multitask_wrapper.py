"""Multi-task wrapper for the base environment.

Wraps the base environment, delegates step/reset, applies task-specific
rewards, and exposes task embeddings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cogniland.envs.env import CognilandEnv
from cogniland.envs.task_sampler import TaskSampler
from cogniland.envs.tasks import compute_task_reward


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

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset all envs and return observations."""
        obs = self.env.reset(seed=seed)
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

        # Add task embedding to observations
        obs["task_embedding"] = self.get_task_embeddings(self.task_ids)

        # Update info with task rewards for episode tracking
        info["task_rewards"] = rewards

        return obs, rewards, dones, info
