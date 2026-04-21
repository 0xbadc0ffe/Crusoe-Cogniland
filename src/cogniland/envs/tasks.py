"""Task reward definitions for the multi-task game.

All tasks share a common base reward:
    r = -step_penalty
      + reach_bonus * [reached YES or NO]
      + shaping_coef * (ctg_prev - ctg_curr)       # PBRS, Dijkstra cost-to-go
      - death_penalty * [died]                     # sparse, terminal

Tasks 1-3 (classification): +correct_answer_bonus when the reached target
matches the biome question.
Tasks 4-6 (craft): +craft_bonus on the step the required tool is crafted.
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
    """Compute per-env reward: base + classification/craft bonuses."""
    B = len(task_ids)
    reward_cfg = config.reward if hasattr(config, "reward") else config.get("reward", {})
    reach_bonus = float(reward_cfg.reach_bonus)
    step_penalty = float(reward_cfg.step_penalty)
    shaping_coef = float(reward_cfg.shaping_coef)
    death_penalty = float(getattr(reward_cfg, "death_penalty", 0.0))
    correct_bonus = float(getattr(reward_cfg, "correct_answer_bonus", 10.0))
    craft_bonus = float(getattr(reward_cfg, "craft_bonus", 10.0))
    forage_berry_bonus = float(getattr(reward_cfg, "forage_berry_bonus", 0.0))

    rewards = np.full(B, -step_penalty, dtype=np.float32)
    rewards[info["reached"]] += reach_bonus

    ctg_prev = info["ctg_prev"]
    ctg_curr = info["ctg_curr"]
    finite = np.isfinite(ctg_prev) & np.isfinite(ctg_curr)
    progress = np.zeros(B, dtype=np.float32)
    progress[finite] = ctg_prev[finite] - ctg_curr[finite]
    rewards += shaping_coef * progress

    alive = info.get("alive")
    if alive is not None and death_penalty != 0.0:
        died = dones & ~alive
        rewards[died] -= death_penalty

    # Markovian berry-forage bonus. Fires on the step the agent does action=4
    # on a berry tile. Gives a direct "berries are valuable" signal without
    # HP-dependent scaling or curriculum annealing.
    if forage_berry_bonus != 0.0:
        foraged = info.get("foraged_berry")
        if foraged is not None:
            rewards += forage_berry_bonus * np.asarray(foraged, dtype=np.float32)

    # Tasks 1-3: classification bonus
    reached_yes = info.get("reached_yes")
    reached_no = info.get("reached_no")
    biome = info.get("biome")
    if reached_yes is not None and reached_no is not None and biome is not None:
        for t_id, target_biome in _TASK_BIOME_QUESTION.items():
            mask = task_ids == t_id
            if not mask.any():
                continue
            is_match = biome == target_biome
            correct = (reached_yes & is_match) | (reached_no & ~is_match)
            rewards[mask & correct] += correct_bonus

    # Tasks 4-6: craft bonus
    crafted = info.get("crafted")
    if crafted is not None:
        for t_id, tool_id in _TASK_CRAFT_TOOL.items():
            mask = (task_ids == t_id) & (crafted == tool_id)
            if mask.any():
                rewards[mask] += craft_bonus

    return rewards
