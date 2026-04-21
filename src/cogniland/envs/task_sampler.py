"""Task sampler for multi-task training.

Samples task ids from a user-provided list across parallel envs. Modes:
    round_robin: cycles through the task list deterministically
    random:      uniformly random pick from the task list
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class TaskSampler:
    """Samples task ids for a batch of environments from a fixed task list."""

    def __init__(
        self,
        task_ids: Sequence[int],
        num_envs: int,
        mode: str = "round_robin",
    ):
        task_ids = [int(t) for t in task_ids]
        if len(task_ids) < 1:
            raise ValueError(f"task_ids must be non-empty, got {task_ids!r}")
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if mode not in ("round_robin", "random"):
            raise ValueError(f"Unknown mode {mode!r}, expected 'round_robin' or 'random'")

        self.task_ids = np.asarray(task_ids, dtype=np.int32)
        self.num_envs = num_envs
        self.mode = mode
        self._rr_counter = 0

    @property
    def num_tasks(self) -> int:
        return int(self.task_ids.shape[0])

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample task ids for all envs.

        Args:
            rng: numpy random Generator (required for 'random' mode)

        Returns:
            int array of shape (num_envs,) with task ids drawn from ``task_ids``
        """
        if self.mode == "round_robin":
            positions = (np.arange(self.num_envs) + self._rr_counter) % self.num_tasks
            self._rr_counter += self.num_envs
            return self.task_ids[positions].astype(np.int32)
        if rng is None:
            rng = np.random.default_rng()
        positions = rng.integers(0, self.num_tasks, size=self.num_envs)
        return self.task_ids[positions].astype(np.int32)

    def fixed(self, task_id: int) -> np.ndarray:
        """Return an array where all envs have the same task.

        ``task_id`` need not be in the sampler's task list — eval loops pin a
        specific task per eval set, and that task might be outside the training
        pool.
        """
        return np.full(self.num_envs, int(task_id), dtype=np.int32)
