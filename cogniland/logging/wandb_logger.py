"""WandB logging wrapper.

Handles training metrics, eval metrics, trajectory images, and behavioral metrics.
"""

from __future__ import annotations

from typing import Any


def _make_run_name(cfg) -> str:
    """Generate a readable run name, e.g. 'ppo_hard'."""
    model = cfg.model.name
    env_mode = "hard" if cfg.env.get("hard_mode", False) else "easy"
    return f"{model}_{env_mode}"


class WandBLogger:
    """Thin wrapper around wandb.init / .log / .finish.

    When ``mode="disabled"`` everything is a no-op, so training still works
    without a WandB account.
    """

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
                tags=[cfg.model.name, f"env_{cfg.env.get('hard_mode', False)}"],
                save_code=True,
            )

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self._run is not None:
            self._run.log(data, step=step)

    def log_trajectory_images(
        self,
        figures: list,
        captions: list[str],
        outcomes: list[str],
        steps_list: list[int],
        step: int | None = None,
    ) -> None:
        """Log trajectory images as individual full-size WandB images.

        Each image is logged under trajectories/<label> so they appear in
        their own 'trajectories' section in the WandB dashboard, separate
        from eval/ metrics.  Individual keys display as large panels.
        """
        if not self.enabled or self._run is None:
            return
        import wandb

        data = {}
        for i, (fig, caption, outcome, n_steps) in enumerate(
            zip(figures, captions, outcomes, steps_list)
        ):
            key = f"trajectories/ep_{i}"
            data[key] = wandb.Image(fig, caption=f"{outcome} | {n_steps} moves")
        self._run.log(data, step=step)

    def log_behavioral_profile(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log a behavioral profile bar chart for quick policy analysis.

        Shows terrain preferences, resource/HP management, and navigation
        strategy in a single chart. Logged under behavior/ section.
        """
        if not self.enabled or self._run is None:
            return
        import wandb

        labels = []
        values = []
        for key, val in metrics.items():
            short = key.replace("behavioral/", "").replace("terrain_", "").replace("_", " ")
            labels.append(short)
            values.append(val)

        table = wandb.Table(
            data=[[l, v] for l, v in zip(labels, values)],
            columns=["metric", "value"],
        )
        chart = wandb.plot.bar(table, "metric", "value", title="Policy Behavioral Profile")
        self._run.log({"behavior/profile": chart}, step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._run.finish()


def _flatten_cfg(cfg) -> dict:
    """Flatten an OmegaConf DictConfig into a plain dict for wandb.config."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
