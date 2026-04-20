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
    curriculum_frac: float = 1.0,
    hp_before: np.ndarray | None = None,
    hp_after: np.ndarray | None = None,
) -> np.ndarray:
    """Compute task-specific rewards for a batch of environments.

    Args:
        task_ids: int array [B] — current task index per env
        rewards_base: float array [B] — base reward from env (currently 0)
        dones: bool array [B] — which envs just finished
        info: dict from env.step() with 'reached', 'ctg_prev', 'ctg_curr',
            'reached_yes', 'reached_no', 'biome', 'crafted', 'alive'
        config: config object with reward params
        curriculum_frac: progress through the curriculum anneal, clamped to
            [0, 1]. ``1.0`` (default) disables the auxiliary forage bonus so
            callers that don't know about the curriculum get the unshaped
            reward.
        hp_before: float array [B] — HP at the start of the step (before the
            env applied drain / healed via forage). Required for the forage
            bonus; if ``None``, the bonus is skipped.
        hp_after: float array [B] — HP at the end of the step (post-step,
            pre-auto-reset). Required for the forage bonus; if ``None``, the
            bonus is skipped.

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

    # Curriculum: annealed forage bonus. Dense shaping to teach the
    #   forage -> survive -> reach chain. Fires only when HP increased
    #   this step (i.e. a berry was successfully foraged), weighted by a
    #   "missing HP" factor that is 0 at high HP and grows non-linearly at
    #   low HP. Anneals linearly to 0 so the final policy is trained
    #   against the unshaped reward.
    if hp_before is not None and hp_after is not None and curriculum_frac < 1.0:
        cur_cfg = _get_cfg_section(config, "curriculum")
        if cur_cfg is not None:
            fb = _get_cfg_section(cur_cfg, "forage_bonus")
            if fb is not None:
                initial_coef = float(_cfg_get(fb, "initial_coef", 0.0))
                coef = initial_coef * max(0.0, 1.0 - float(curriculum_frac))
                if coef > 0.0:
                    hp_thresh = float(_cfg_get(fb, "hp_thresh", 90.0))
                    exp = float(_cfg_get(fb, "missing_hp_exp", 2.0))
                    dhp = np.maximum(
                        0.0,
                        hp_after.astype(np.float32) - hp_before.astype(np.float32),
                    )
                    missing = np.maximum(0.0, hp_thresh - hp_before.astype(np.float32))
                    missing /= max(hp_thresh, 1e-6)
                    weight = missing ** exp
                    rewards += (coef * dhp * weight).astype(np.float32)

    return rewards


def _get_cfg_section(cfg: Any, key: str) -> Any:
    """Fetch a sub-section from an OmegaConf node or a plain dict-ish config."""
    if cfg is None:
        return None
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, None)
    return None


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Best-effort attribute/key lookup with default."""
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


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
