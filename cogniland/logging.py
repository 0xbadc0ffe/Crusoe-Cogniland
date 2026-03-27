"""WandB logging for Cogniland training runs."""

from __future__ import annotations

import json
from typing import Any


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
    env_mode = "sweep_reward"
    return f"{model}_{env_mode}"


def _make_group_name(cfg) -> str:
    model = cfg.models.name
    env_mode = "sweep_reward"
    lr = cfg.models.training.get("learning_rate", 3e-4)
    lr_str = f"lr{lr:.0e}".replace("-0", "-").replace("+0", "")
    return f"{model}_{env_mode}_{lr_str}"


def _make_run_config(cfg) -> dict:
    """Slim config for WandB — all hyperparams needed for reproducibility and run comparison.

    Grouped into sections that appear as collapsible prefixes in the WandB UI.
    The full config is also archived as a JSON string in ``run.config["_full_config"]``
    so it is accessible via the API without polluting the column list.
    """
    env = cfg.env
    rw  = env.get("reward", {})
    mg  = env.get("map_generation", env)  # fallback to flat keys for old configs
    tr  = cfg.models.get("training", {})
    ds  = tr.get("dataset", {})

    slim = {
        # ── Reward shaping ────────────────────────────────────────────────
        "reward/reach_bonus":  rw.get("reach_bonus"),
        "reward/lambda_p":     rw.get("lambda_p", env.get("lambda_p")),
        "reward/lambda_rho":   rw.get("lambda_rho"),
        "reward/lambda_t":     rw.get("lambda_t"),
        "reward/lambda_d":     rw.get("lambda_d"),
        "reward/beta_raft":    rw.get("beta_raft"),

        # ── PPO optimisation ────────────────────────────────────────────────
        "ppo/lr":              tr.get("learning_rate"),
        "ppo/anneal_lr":       tr.get("anneal_lr"),
        "ppo/clip_range":      tr.get("policy_clip_range"),
        "ppo/value_coef":      tr.get("value_loss_weight"),
        "ppo/entropy_coef":    tr.get("entropy_bonus_weight"),
        "ppo/max_grad_norm":   tr.get("max_grad_norm"),
        "ppo/epochs":          tr.get("epochs_per_update"),
        "ppo/minibatch":       tr.get("minibatch_size"),

        # ── GAE / returns ───────────────────────────────────────────────────
        "gae/gamma":           tr.get("discount_factor"),
        "gae/lambda":          tr.get("gae_lambda"),

        # ── Rollout ─────────────────────────────────────────────────────────
        "rollout/parallel_envs":   tr.get("parallel_envs"),
        "rollout/moves_per_update": tr.get("moves_per_rollout"),
        "rollout/total_moves":     tr.get("total_env_moves"),

        # ── Curriculum ──────────────────────────────────────────────────────
        "curriculum/switch_steps":  ds.get("curriculum_switch_steps"),
        "curriculum/easy_radius":   ds.get("curriculum_easy_radius"),
        "curriculum/dataset":       ds.get("path"),

        # ── Model architecture ──────────────────────────────────────────────
        "model/name":          cfg.models.get("name"),
        "model/hidden_dim":    cfg.models.get("hidden_dim"),
        "model/scalar_dim":    cfg.models.get("scalar_dim"),
        "model/cnn_channels":  cfg.models.get("cnn_channels"),

        # ── Environment ─────────────────────────────────────────────────────
        "env/map_size":   mg.get("size"),
        "env/max_steps":  env.get("max_steps"),
        "env/seed":       mg.get("seed", env.get("seed")),
    }
    return {k: v for k, v in slim.items() if v is not None}


def _flatten_cfg(cfg) -> dict:
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)


# Metrics shown in the final test summary table (in display order)
# Each entry: (metric_suffix, is_plain_rate)
# is_plain_rate=True  → logged as "{prefix}/{suffix}"      (no _mean)
# is_plain_rate=False → logged as "{prefix}/{suffix}_mean" (with _std)
_TEST_SUMMARY_METRICS: list[tuple[str, bool]] = [
    ("success_rate",   True),
    ("return",         False),
    ("episode_length", False),
    ("directness",     False),
    ("exploration",    False),
    ("risk_exposure",  False),
    ("final_hp",       False),
    ("final_resources",False),
]


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
                name=log_cfg.get("name") or _make_run_name(cfg),
                group=log_cfg.get("group") or _make_group_name(cfg),
                mode=log_cfg.mode,
                config=_make_run_config(cfg),
                tags=[cfg.models.name],
                save_code=False,   # git is the source of truth; avoids artifact clutter
            )
            # Archive full config as a JSON string — accessible via run.config["_full_config"]
            # but stored as a single opaque value so it doesn't add columns.
            self._run.config.update(
                {"_full_config": json.dumps(_flatten_cfg(cfg))},
                allow_val_change=True,
            )
            # Pre-initialize key eval metrics in summary so they appear as columns
            # in the runs table from the start (filled in during/after training).
            self._run.summary.update({
                "test_det/env/success_rate":      None,
                "test_det/env/return_mean":       None,
                "test_det/env/directness_mean":   None,
                "test_det/env/exploration_mean":  None,
                "test_det/env/risk_exposure_mean":None,
            })
        # Per-namespace row lists for terrain distribution; grow across eval steps
        self._terrain_history: dict[str, list] = {}

    _SUMMARY_SUFFIXES = ("_std", "_max", "_min")

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        """Log scalars. Keys ending in _std/_max/_min go to run summary only
        (no time-series plot); all other keys are logged normally."""
        if not self.enabled or self._run is None:
            return
        summary = {k: v for k, v in data.items() if k.endswith(self._SUMMARY_SUFFIXES)}
        series  = {k: v for k, v in data.items() if k not in summary}
        if series:
            self._run.log(series, step=step)
        if summary:
            self._run.summary.update(summary)

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

    def log_final_test_summary(self, metrics: dict[str, float]) -> None:
        """Push all test metrics to run.summary and log a readable summary table.

        Call this once at the end of training after the final test evaluation.
        Two things happen:
        1. Every test metric is pushed to ``run.summary`` so it appears as a
           selectable column in the WandB runs table.
        2. A ``wandb.Table`` with rows [metric | det mean | det std | stoch mean | stoch std]
           is logged under ``test/summary_table`` for at-a-glance comparison.
        """
        if not self.enabled or self._run is None:
            return
        import wandb

        # Push everything to summary (makes all test metrics available as columns)
        self._run.summary.update(metrics)

        # Build a clean two-column (det / stoch) table for the important metrics
        rows = []
        for metric, is_rate in _TEST_SUMMARY_METRICS:
            if is_rate:
                det   = metrics.get(f"test_det/env/{metric}")
                stoch = metrics.get(f"test_stoch/env/{metric}")
                if det is None and stoch is None:
                    continue
                rows.append([
                    metric,
                    f"{det:.3f}"   if det   is not None else "—",
                    "—",
                    f"{stoch:.3f}" if stoch is not None else "—",
                    "—",
                ])
            else:
                det_m  = metrics.get(f"test_det/env/{metric}_mean")
                det_s  = metrics.get(f"test_det/env/{metric}_std")
                sto_m  = metrics.get(f"test_stoch/env/{metric}_mean")
                sto_s  = metrics.get(f"test_stoch/env/{metric}_std")
                if det_m is None and sto_m is None:
                    continue
                rows.append([
                    metric,
                    f"{det_m:.3f}" if det_m is not None else "—",
                    f"{det_s:.3f}" if det_s is not None else "—",
                    f"{sto_m:.3f}" if sto_m is not None else "—",
                    f"{sto_s:.3f}" if sto_s is not None else "—",
                ])

        if rows:
            table = wandb.Table(
                columns=["metric", "det_mean", "det_std", "stoch_mean", "stoch_std"],
                data=rows,
            )
            self._run.log({"test/summary_table": table})

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._run.finish()
