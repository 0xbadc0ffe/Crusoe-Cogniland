"""CognilandSummarizer — aggregates EvalResult into flat dicts for logging."""

from __future__ import annotations

import numpy as np

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


class CognilandSummarizer:
    """Converts EvalResult into flat scalar dicts. No WandB calls."""

    def scalar_metrics(self, result: EvalResult) -> dict[str, float]:
        """Return success_rate plus mean/std/max/min for every metric.

        Keys are logged as standard WandB scalars and render as native line plots.
        """
        prefix = f"{result.split}_{result.mode}/env"
        eps = result.episodes
        n = len(eps)
        n_success = sum(1 for ep in eps if ep.outcome == "success")
        out: dict[str, float] = {f"{prefix}/success_rate": n_success / n if n else 0.0}

        all_metrics: dict[str, list[float]] = {
            "return": [ep.total_return for ep in eps],
            "episode_length": [float(ep.episode_length) for ep in eps],
        }
        for key in _SCALAR_METRIC_KEYS:
            all_metrics[key] = [ep.metrics[key] for ep in eps]

        for name, vals in all_metrics.items():
            arr = np.array(vals, dtype=float)
            out[f"{prefix}/{name}_mean"] = float(arr.mean())
            out[f"{prefix}/{name}_std"]  = float(arr.std())
            out[f"{prefix}/{name}_max"]  = float(arr.max())
            out[f"{prefix}/{name}_min"]  = float(arr.min())

        return out

    def terrain_pcts(self, result: EvalResult) -> dict[str, float]:
        """Return per-terrain mean visit fractions for the stacked area chart."""
        eps = result.episodes
        pcts: dict[str, float] = {}
        for name in _TERRAIN_NAMES:
            vals = [ep.metrics[f"terrain_visit_{name}"] for ep in eps]
            pcts[name] = sum(vals) / len(vals) if vals else 0.0
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
