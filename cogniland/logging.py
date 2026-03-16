"""WandB logging for Cogniland training runs."""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Terrain names (9 types, indices 0–8)
# ---------------------------------------------------------------------------

TERRAIN_NAMES = [
    "ocean", "deep_water", "water",
    "beach", "sandy", "grassland",
    "forest", "rocky", "mountains",
]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def log_rollout_stats(logger, episode_stats: dict, step: int) -> None:
    """Log online episode stats collected during a training rollout.

    Args:
        logger: WandBLogger instance.
        episode_stats: dict with keys ``episode_rewards``, ``episode_lengths``,
            optionally ``episode_reached`` — all torch.Tensors.
        step: WandB x-axis step (usually update index).
    """
    if not episode_stats:
        return
    ep_rewards = episode_stats["episode_rewards"]
    ep_lengths = episode_stats["episode_lengths"]
    ep_reached = episode_stats.get("episode_reached")

    data: dict[str, Any] = {
        "train/env/episode_return_mean": ep_rewards.mean().item(),
        "train/env/episode_length_mean": ep_lengths.mean().item(),
    }
    if ep_reached is not None:
        data["train/env/success_rate"] = ep_reached.float().mean().item()

    logger.log(data, step=step)


# ---------------------------------------------------------------------------
# WandB logger
# ---------------------------------------------------------------------------

def _make_run_name(cfg) -> str:
    model = cfg.models.name
    env_mode = "hard" if cfg.env.get("hard_mode", False) else "easy"
    return f"{model}_{env_mode}"


def _make_group_name(cfg) -> str:
    model = cfg.models.name
    env_mode = "hard" if cfg.env.get("hard_mode", False) else "easy"
    lr = cfg.models.training.get("learning_rate", 3e-4)
    lr_str = f"lr{lr:.0e}".replace("-0", "-").replace("+0", "")
    return f"{model}_{env_mode}_{lr_str}"


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
                group=_make_group_name(cfg),
                mode=log_cfg.mode,
                config=_flatten_cfg(cfg),
                tags=[cfg.models.name, f"env_{cfg.env.get('hard_mode', False)}"],
                save_code=True,
            )
        # Per-namespace flat row lists; grow across eval steps
        self._terrain_history: dict[str, list] = {}
        self._eval_history: dict[str, dict[str, list]] = {}

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
        columns: list[str],
        rows: list[list],
        step: int,
        namespace: str = "val/det",
    ) -> None:
        """Log a structured WandB Table with per-episode evaluation data.

        Args:
            columns: column names.
            rows: list of row lists.
            step: WandB x-axis step.
            namespace: key prefix, e.g. ``"val/det"`` → key ``"val/det/tables/episodes"``.
        """
        if not self.enabled or self._run is None:
            return
        import wandb
        table = wandb.Table(columns=columns, data=rows)
        self._run.log({f"{namespace}/tables/episodes": table}, step=step)

    # Terrain names + colors in stacking order (index = terrain_idx column)
    _TERRAIN_ORDER = [
        ("ocean",      "#0523E1"),
        ("deep_water", "#1941E1"),
        ("water",      "#4169E1"),
        ("beach",      "#EED6AF"),
        ("sandy",      "#D2B48C"),
        ("grassland",  "#228B22"),
        ("forest",     "#006400"),
        ("rocky",      "#8B8989"),
        ("mountains",  "#FFFAFA"),
    ]

    def log_terrain_scalars(
        self,
        terrain_pcts: dict[str, float],
        step: int,
        namespace: str = "val_det",
    ) -> None:
        """Append terrain visit fractions to a growing WandB Table.

        Each call adds 9 rows (one per terrain) with columns
        ``[step, terrain, fraction, terrain_idx]``. The table grows across eval
        steps so a custom Vega stacked-area chart can be configured in the W&B UI
        using the JSON spec stored in ``docs/wandb_specs/terrain_distribution.json``.

        Args:
            terrain_pcts: dict mapping terrain name → mean visit fraction.
            step: WandB x-axis step.
            namespace: e.g. ``"val_det"`` → key ``"val_det/tables/terrain_distribution"``.
        """
        if not self.enabled or self._run is None:
            return
        import wandb

        if namespace not in self._terrain_history:
            self._terrain_history[namespace] = []
        for idx, (name, _) in enumerate(self._TERRAIN_ORDER):
            self._terrain_history[namespace].append(
                [step, name, round(terrain_pcts.get(name, 0.0), 6), idx]
            )

        table = wandb.Table(
            data=self._terrain_history[namespace],
            columns=["step", "terrain", "fraction", "terrain_idx"],
        )
        chart = wandb.plot_table(
            vega_spec_name="crusoe/terrain_distribution",
            data_table=table,
            fields={
                "step": "step",
                "terrain": "terrain",
                "fraction": "fraction",
                "terrain_idx": "terrain_idx",
            },
        )
        self._run.log({f"{namespace}/terrain_distribution": chart}, step=step)

    def log_eval_charts(
        self,
        per_ep_data: dict[str, list[float]],
        namespace: str,
        step: int,
    ) -> None:
        """Append aggregated eval metrics to growing WandB Tables (one per metric).

        Each call appends a ``[step, mean, lower_std, upper_std]`` row per metric.
        Tables grow across eval steps so custom Vega shaded-area charts can be
        configured in the W&B UI using the JSON spec stored in
        ``docs/wandb_specs/eval_metric.json``.

        Args:
            per_ep_data: ``{metric_name: [val_ep0, val_ep1, …]}`` from
                CognilandSummarizer.per_episode_metrics().
            namespace: e.g. ``"val_det"`` → keys ``"val_det/tables/eval_{metric}"``.
            step: WandB x-axis step.
        """
        if not self.enabled or self._run is None:
            return
        import wandb

        if namespace not in self._eval_history:
            self._eval_history[namespace] = {}
        ns_hist = self._eval_history[namespace]

        for metric_name, episode_values in per_ep_data.items():
            arr = np.array(episode_values, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std())

            label = metric_name.replace("_", " ").title()
            if metric_name not in ns_hist:
                ns_hist[metric_name] = []
            ns_hist[metric_name].append([step, mean, mean - std, mean + std, label])

            table = wandb.Table(
                data=ns_hist[metric_name],
                columns=["step", "mean", "lower_std", "upper_std", "label"],
            )
            chart = wandb.plot_table(
                vega_spec_name="crusoe/eval_mean_std",
                data_table=table,
                fields={
                    "step": "step",
                    "mean": "mean",
                    "lower_std": "lower_std",
                    "upper_std": "upper_std",
                    "label": "label",
                },
            )
            self._run.log({f"{namespace}/env/{metric_name}": chart}, step=step)

    def log_model_artifact(self, name: str, path: str, aliases: list[str] = ["latest"]) -> None:
        """Uploads a model checkpoint to WandB as an artifact."""
        if not self.enabled or self._run is None:
            return
        import wandb
        artifact_name = f"{name}_{self._run.id}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Checkpoint from run {self._run.name}"
        )
        artifact.add_file(path)
        self._run.log_artifact(artifact, aliases=aliases)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._run.finish()
