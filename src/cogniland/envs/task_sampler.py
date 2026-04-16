"""Task sampler for multi-task training.

Supports round_robin and random sampling of task indices across parallel envs.
"""

from __future__ import annotations

import numpy as np


class TaskSampler:
    """Samples task indices for a batch of environments.

    Modes:
        round_robin: cycles through tasks deterministically
        random: uniformly random task selection
    """

    def __init__(self, num_tasks: int, num_envs: int, mode: str = "round_robin"):
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be >= 1, got {num_tasks}")
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if mode not in ("round_robin", "random"):
            raise ValueError(f"Unknown mode {mode!r}, expected 'round_robin' or 'random'")

        self.num_tasks = num_tasks
        self.num_envs = num_envs
        self.mode = mode
        self._rr_counter = 0

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample task indices for all envs.

        Args:
            rng: numpy random Generator (required for 'random' mode)

        Returns:
            int array of shape (num_envs,) with task indices in [0, num_tasks)
        """
        if self.mode == "round_robin":
            # Each env gets the next task in sequence
            indices = (np.arange(self.num_envs) + self._rr_counter) % self.num_tasks
            self._rr_counter += self.num_envs
            return indices.astype(np.int32)
        else:
            # Random uniform
            if rng is None:
                rng = np.random.default_rng()
            return rng.integers(0, self.num_tasks, size=self.num_envs).astype(np.int32)

    def fixed(self, task_id: int) -> np.ndarray:
        """Return array where all envs have the same task.

        Args:
            task_id: task index to assign to all envs

        Returns:
            int array of shape (num_envs,) filled with task_id
        """
        if not (0 <= task_id < self.num_tasks):
            raise ValueError(
                f"task_id {task_id} out of range [0, {self.num_tasks})"
            )
        return np.full(self.num_envs, task_id, dtype=np.int32)
