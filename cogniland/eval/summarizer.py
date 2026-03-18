"""CognilandSummarizer — aggregates EvalResult into flat dicts for logging."""

from __future__ import annotations

from cogniland.eval.runner import EvalResult

_TERRAIN_NAMES = [
    "ocean", "deep_water", "water", "beach", "sandy",
    "grassland", "forest", "rocky", "mountains",
]

_SCALAR_METRIC_KEYS = [
    "min_hp", "final_hp", "mean_hp", "danger_fraction",
    "final_resources", "mean_resources", "max_resources",
    "directness", "risk_exposure", "exploration",
]


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


class CognilandSummarizer:
    """Converts EvalResult into flat scalar dicts. No WandB calls."""

    def scalar_metrics(self, result: EvalResult) -> dict[str, float]:
        """Return success_rate as the single aggregate scalar.

        All other per-episode metrics are handled by ``per_episode_metrics()``
        and logged as interactive multi-line charts via WandBLogger.log_eval_charts().
        """
        prefix = f"{result.split}_{result.mode}/env"
        eps = result.episodes
        n_success = sum(1 for ep in eps if ep.outcome == "success")
        return {f"{prefix}/success_rate": n_success / len(eps) if eps else 0.0}

    def per_episode_metrics(self, result: EvalResult) -> dict[str, list[float]]:
        """Return per-metric lists of episode values for charting.

        Returns ``{metric_name: [val_ep0, val_ep1, …]}``.
        """
        eps = result.episodes
        metrics: dict[str, list[float]] = {
            "return": [ep.total_return for ep in eps],
            "episode_length": [float(ep.episode_length) for ep in eps],
        }
        for key in _SCALAR_METRIC_KEYS:
            metrics[key] = [ep.metrics[key] for ep in eps]
        return metrics

    def terrain_pcts(self, result: EvalResult) -> dict[str, float]:
        """Return per-terrain mean visit fractions for the stacked area chart."""
        eps = result.episodes
        pcts: dict[str, float] = {}
        for name in _TERRAIN_NAMES:
            vals = [ep.metrics[f"terrain_visit_{name}"] for ep in eps]
            pcts[name] = _mean(vals)
        return pcts

    def eval_table_rows(self, result: EvalResult) -> tuple[list[str], list[list]]:
        """Return (columns, rows) suitable for a WandB Table."""
        columns = ["episode", "outcome", "return", "episode_length", "final_hp", "trajectory"]
        rows = []
        for i, ep in enumerate(result.episodes):
            traj = ep.trajectory
            traj_str = " → ".join(f"({r},{c})" for r, c in traj) if traj else ""
            rows.append([
                i,
                ep.outcome,
                round(ep.total_return, 2),
                ep.episode_length,
                round(ep.metrics["final_hp"], 2),
                traj_str,
            ])
        return columns, rows
