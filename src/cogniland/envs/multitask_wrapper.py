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

        # Task embeddings: fixed one-hot vectors. Task i -> row i of the
        # identity matrix. Requires ``num_tasks <= task_embedding_dim``.
        if num_tasks > task_embedding_dim:
            raise ValueError(
                f"one-hot task embedding requires task_embedding_dim "
                f"({task_embedding_dim}) >= num_tasks ({num_tasks})"
            )
        self._task_embeddings = np.eye(task_embedding_dim, dtype=np.float32)[:num_tasks]

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

        # Curriculum state (forage bonus). See configs/env/cogniland.yaml
        # -> curriculum.forage_bonus. ``_prev_hp`` tracks the HP at the start
        # of each step (post auto-reset for envs that finished last step).
        self._init_hp = float(getattr(env, "_effects", None).init_hp) \
            if hasattr(env, "_effects") and env._effects is not None \
            else 100.0
        self._prev_hp = np.full(env.num_envs, self._init_hp, dtype=np.float32)
        self._curriculum_frac = 0.0

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

    def set_curriculum_progress(self, frac: float) -> None:
        """Advance the curriculum schedule.

        ``frac`` is expected to be ``total_trained_frames / anneal_frames``.
        Clamped to ``[0, 1]``; at ``frac >= 1`` the auxiliary reward is 0.
        """
        self._curriculum_frac = float(np.clip(frac, 0.0, 1.0))

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
        self._prev_hp.fill(self._init_hp)
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
        # Snapshot HP at the start of this step before the env applies drain.
        hp_before = self._prev_hp.copy()

        obs, base_rewards, dones, info = self.env.step(actions)

        # Post-step, pre-auto-reset HP. Falls back to ``hp_before`` if the env
        # did not expose ``info['hp']`` (keeps older envs working; dhp will
        # simply be 0 and the forage bonus won't fire).
        hp_after = info.get("hp")
        if hp_after is None:
            hp_after = hp_before

        # Compute task-specific rewards
        rewards = compute_task_reward(
            self.task_ids, base_rewards, dones, info, self._config,
            curriculum_frac=self._curriculum_frac,
            hp_before=hp_before, hp_after=hp_after,
        )

        # Advance prev_hp for the next step. Envs that just finished will
        # auto-reset inside the base env, so their next-step baseline is
        # ``init_hp`` rather than the pre-reset value (0 on death, etc.).
        self._prev_hp = np.where(dones, self._init_hp, hp_after).astype(np.float32)

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
