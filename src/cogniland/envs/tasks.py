"""Task reward definitions for the multi-task game.

All tasks share a common base reward ``r0``:
  - ``-step_penalty`` every step
  - ``+reach_bonus`` when the agent reaches either the YES or NO target
  - PBRS shaping ``+shaping_coef * (ctg_prev - ctg_curr)`` every step, where
    ``ctg`` is Dijkstra cost-to-go from the targets' midpoint

Tasks 1-3 add a classification question:
  - Task 1: "Is this biome archipelago?"
  - Task 2: "Is this biome grassland?"
  - Task 3: "Is this biome highland?"
  Agent picks by reaching YES or NO. ``+correct_answer_bonus`` when the
  reached target matches the biome answer. Biome is not in the obs — it must
  be inferred from what the agent sees.

Tasks 4-6 add a crafting bonus:
  - Task 4: craft a raft  → ``+craft_bonus`` (one-shot per episode)
  - Task 5: craft a rope  → ``+craft_bonus``
  - Task 6: craft shoes   → ``+craft_bonus``
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Task → target biome (for classification tasks 1-3)
_TASK_BIOME_QUESTION = {
    1: "archipelago",
    2: "grassland",
    3: "highland",
}

# Task → required tool id for craft bonus (for tasks 4-6)
_TASK_CRAFT_TOOL = {
    4: 1,  # raft
    5: 2,  # rope
    6: 3,  # shoes
}


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
        info: dict from env.step() with 'reached', 'ctg_prev', 'ctg_curr',
            'reached_yes', 'reached_no', 'biome', 'crafted', 'alive'
        config: config object with reward params

    Returns:
        float array [B] — modified rewards
    """
    B = len(task_ids)
    (
        reach_bonus, step_penalty, shaping_coef,
        death_penalty, correct_bonus, craft_bonus,
    ) = _read_reward_cfg(config)

    # Base r0 for every env (step penalty + shaping + reach bonus).
    rewards = np.full(B, -step_penalty, dtype=np.float32)
    rewards[info["reached"]] += reach_bonus

    ctg_prev = info["ctg_prev"]
    ctg_curr = info["ctg_curr"]
    finite = np.isfinite(ctg_prev) & np.isfinite(ctg_curr)
    progress = np.zeros(B, dtype=np.float32)
    progress[finite] = ctg_prev[finite] - ctg_curr[finite]
    rewards += shaping_coef * progress

    # Sparse death penalty — fires on the step an episode ends with hp<=0.
    alive = info.get("alive")
    if alive is not None and death_penalty != 0.0:
        died = dones & ~alive
        rewards[died] -= death_penalty

    # Tasks 1-3: classification bonus.
    reached_yes = info.get("reached_yes")
    reached_no = info.get("reached_no")
    biome = info.get("biome")
    if reached_yes is not None and reached_no is not None and biome is not None:
        for t_id, target_biome in _TASK_BIOME_QUESTION.items():
            mask = task_ids == t_id
            if not mask.any():
                continue
            # Correct answer: biome matches → YES, else → NO.
            is_match = biome == target_biome
            correct = (reached_yes & is_match) | (reached_no & ~is_match)
            rewards[mask & correct] += correct_bonus

    # Tasks 4-6: craft bonus (fires on the step the required tool is crafted).
    crafted = info.get("crafted")
    if crafted is not None:
        for t_id, tool_id in _TASK_CRAFT_TOOL.items():
            mask = (task_ids == t_id) & (crafted == tool_id)
            if mask.any():
                rewards[mask] += craft_bonus

    return rewards


def _read_reward_cfg(config: Any) -> tuple[float, float, float, float, float, float]:
    reward_cfg = config.reward if hasattr(config, "reward") else config.get("reward", {})
    if hasattr(reward_cfg, "reach_bonus"):
        reach_bonus = float(reward_cfg.reach_bonus)
        step_penalty = float(reward_cfg.step_penalty)
        shaping_coef = float(reward_cfg.shaping_coef)
        death_penalty = float(getattr(reward_cfg, "death_penalty", 0.0))
        correct_bonus = float(getattr(reward_cfg, "correct_answer_bonus", 10.0))
        craft_bonus = float(getattr(reward_cfg, "craft_bonus", 10.0))
    else:
        reach_bonus = reward_cfg.get("reach_bonus", 10.0)
        step_penalty = reward_cfg.get("step_penalty", 0.01)
        shaping_coef = reward_cfg.get("shaping_coef", 0.1)
        death_penalty = reward_cfg.get("death_penalty", 0.0)
        correct_bonus = reward_cfg.get("correct_answer_bonus", 10.0)
        craft_bonus = reward_cfg.get("craft_bonus", 10.0)
    return (
        reach_bonus, step_penalty, shaping_coef,
        death_penalty, correct_bonus, craft_bonus,
    )
