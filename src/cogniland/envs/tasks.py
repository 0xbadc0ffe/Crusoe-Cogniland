"""Task reward definitions for the multi-task game.

Each task is a function that computes per-step reward given env info.
Task 0 (reach target) is fully implemented; tasks 1-6 are stubs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_task_reward(
    task_ids: np.ndarray,
    rewards_base: np.ndarray,
    dones: np.ndarray,
    info: dict[str, Any],
    config: Any,
) -> np.ndarray:
    """Compute task-specific rewards for a batch of environments.

    Args:
        task_ids: int array [B] — current task index per env
        rewards_base: float array [B] — base reward from env (currently 0)
        dones: bool array [B] — which envs just finished
        info: dict from env.step() with 'reached', 'dist_to_target', etc.
        config: config object with reward params

    Returns:
        float array [B] — modified rewards
    """
    B = len(task_ids)
    rewards = np.zeros(B, dtype=np.float32)

    # Dispatch by task
    mask_0 = task_ids == 0
    if mask_0.any():
        rewards[mask_0] = _task_0_reward(mask_0, dones, info, config)

    # Tasks 1-6: stub — return 0
    # Future tasks will be added here

    return rewards


def _task_0_reward(
    mask: np.ndarray,
    dones: np.ndarray,
    info: dict[str, Any],
    config: Any,
) -> np.ndarray:
    """Task 0: Reach the target.

    Reward components:
      - Step penalty: -step_penalty per step
      - Success bonus: +reach_bonus on reaching target
      - Distance shaping: at episode end, +distance_shaping_coef * (1 - final_dist / initial_dist)
    """
    # Read config
    reward_cfg = config.reward if hasattr(config, "reward") else config.get("reward", {})
    if hasattr(reward_cfg, "reach_bonus"):
        reach_bonus = float(reward_cfg.reach_bonus)
        step_penalty = float(reward_cfg.step_penalty)
        dist_coef = float(reward_cfg.distance_shaping_coef)
    else:
        reach_bonus = reward_cfg.get("reach_bonus", 100.0)
        step_penalty = reward_cfg.get("step_penalty", 0.01)
        dist_coef = reward_cfg.get("distance_shaping_coef", 0.1)

    count = int(mask.sum())
    rewards = np.full(count, -step_penalty, dtype=np.float32)

    # Extract relevant info for masked envs
    reached = info["reached"][mask]
    done = dones[mask]
    dist = info["dist_to_target"][mask]
    init_dist = info["initial_dist"][mask]

    # Success bonus
    rewards[reached] += reach_bonus

    # Distance shaping at episode end (for all done envs, not just successful)
    done_mask = done & ~reached
    if done_mask.any():
        safe_init = np.maximum(init_dist[done_mask], 1e-6)
        progress = 1.0 - dist[done_mask] / safe_init
        rewards[done_mask] += dist_coef * progress

    return rewards
