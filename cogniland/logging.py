"""WandB logging and behavioral metrics."""

from __future__ import annotations

from typing import Any

import torch

from cogniland.env.types import EnvState


# ---------------------------------------------------------------------------
# Behavioral metrics
# ---------------------------------------------------------------------------

def compute_behavioral_metrics(
    terrain_visits: torch.Tensor,
    total_moves: torch.Tensor,
    final_state: EnvState,
) -> dict[str, float]:
    """Compute behavioral metrics averaged over evaluation episodes.

    Args:
        terrain_visits: [num_episodes, 9] count of visits to each terrain type
        total_moves: [num_episodes] moves per episode
        final_state: final EnvState from evaluation

    Returns:
        Dict of behavioral/ prefixed metrics.
    """
    moves_safe = total_moves.clamp(min=1).unsqueeze(1)  # [N, 1]
    visit_pct = terrain_visits / moves_safe  # [N, 9]

    water_pct = visit_pct[:, 0:3].sum(dim=1).mean().item()
    forest_pct = visit_pct[:, 6].mean().item()
    mountain_pct = visit_pct[:, 7:9].sum(dim=1).mean().item()
    land_pct = visit_pct[:, 3:6].sum(dim=1).mean().item()
    mean_resource_held = final_state.resources.mean().item()
    hp_score = (final_state.hp / 100.0).mean().item()
    compass_dist = torch.norm(final_state.compass.float(), dim=1)
    directness = (compass_dist / total_moves.clamp(min=1)).mean().item()

    return {
        "behavioral/water_pct": water_pct,
        "behavioral/land_pct": land_pct,
        "behavioral/forest_pct": forest_pct,
        "behavioral/mountain_pct": mountain_pct,
        "behavioral/resource_held": mean_resource_held,
        "behavioral/hp_score": hp_score,
        "behavioral/directness": directness,
    }


# ---------------------------------------------------------------------------
# WandB logger
# ---------------------------------------------------------------------------

def _make_run_name(cfg) -> str:
    model = cfg.models.name
    env_mode = "hard" if cfg.env.get("hard_mode", False) else "easy"
    return f"{model}_{env_mode}"


def _flatten_cfg(cfg) -> dict:
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)


class WandBLogger:
    """Thin wrapper around wandb — no-ops when mode="disabled"."""

    def __init__(self, cfg):
        log_cfg = cfg.logging.wandb
        self.enabled = log_cfg.mode != "disabled"
        self._run = None

        if self.enabled:
            import wandb
            self._run = wandb.init(
                project=log_cfg.project,
                entity=log_cfg.get("entity", None),
                name=_make_run_name(cfg),
                mode=log_cfg.mode,
                config=_flatten_cfg(cfg),
                tags=[cfg.models.name, f"env_{cfg.env.get('hard_mode', False)}"],
                save_code=True,
            )

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self._run is not None:
            self._run.log(data, step=step)

    def log_trajectory_images(
        self, figures: list, captions: list[str], env_indices: list[int], step: int,
    ) -> None:
        if not self.enabled or self._run is None:
            return
        import wandb
        data = {}
        for fig, caption, env_idx in zip(figures, captions, env_indices):
            data[f"trajectories/env_{env_idx}"] = wandb.Image(fig, caption=caption)
        self._run.log(data, step=step)

    def log_eval_table(
        self,
        reached: torch.Tensor,
        total_rewards: torch.Tensor,
        total_moves: torch.Tensor,
        final_hp: torch.Tensor,
        trajectories: list[list[tuple[int, int]]],
        step: int,
    ) -> None:
        """Log a structured WandB Table with per-episode evaluation data."""
        if not self.enabled or self._run is None:
            return
        import wandb

        columns = ["episode", "outcome", "reward", "moves", "final_hp", "trajectory"]
        rows = []
        for i in range(len(reached)):
            outcome = "success" if reached[i].item() else "fail"
            traj_str = " → ".join(f"({r},{c})" for r, c in trajectories[i])
            rows.append([
                i,
                outcome,
                round(total_rewards[i].item(), 2),
                int(total_moves[i].item()),
                round(final_hp[i].item(), 2),
                traj_str,
            ])
        table = wandb.Table(columns=columns, data=rows)
        self._run.log({"eval/episode_table": table}, step=step)

    def log_behavioral_profile(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        import wandb
        labels, values = [], []
        for key, val in metrics.items():
            short = key.replace("behavioral/", "").replace("terrain_", "").replace("_", " ")
            labels.append(short)
            values.append(val)
        table = wandb.Table(data=[[l, v] for l, v in zip(labels, values)], columns=["metric", "value"])
        chart = wandb.plot.bar(table, "metric", "value", title="Policy Behavioral Profile")
        self._run.log({"behavior/profile": chart}, step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._run.finish()
