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
      - Potential-based shaping: +shaping_coef * (ctg_prev - ctg_curr) every step,
        where ``ctg`` is the Dijkstra cost-to-go computed once per episode from
        the target on the HP-drain graph. This is PBRS (Ng et al. 1999) with
        potential Phi(s) = -ctg(s) and gamma=1: summing it along a successful
        trajectory telescopes to ``+shaping_coef * ctg_spawn``.
    """
    reward_cfg = config.reward if hasattr(config, "reward") else config.get("reward", {})
    if hasattr(reward_cfg, "reach_bonus"):
        reach_bonus = float(reward_cfg.reach_bonus)
        step_penalty = float(reward_cfg.step_penalty)
        shaping_coef = float(reward_cfg.shaping_coef)
    else:
        reach_bonus = reward_cfg.get("reach_bonus", 10.0)
        step_penalty = reward_cfg.get("step_penalty", 0.01)
        shaping_coef = reward_cfg.get("shaping_coef", 0.05)

    count = int(mask.sum())
    rewards = np.full(count, -step_penalty, dtype=np.float32)

    # Success bonus
    rewards[info["reached"][mask]] += reach_bonus

    # PBRS shaping: filter out non-finite values (e.g., stepping onto a deadly
    # tile leaves ctg_curr = +inf). On those transitions we contribute no shaping.
    ctg_prev = info["ctg_prev"][mask]
    ctg_curr = info["ctg_curr"][mask]
    finite = np.isfinite(ctg_prev) & np.isfinite(ctg_curr)
    progress = np.zeros(count, dtype=np.float32)
    progress[finite] = ctg_prev[finite] - ctg_curr[finite]
    rewards += shaping_coef * progress

    return rewards
