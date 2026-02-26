"""WandB logging and behavioral metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from cogniland.env.types import EnvState


# ---------------------------------------------------------------------------
# Behavioral metrics
# ---------------------------------------------------------------------------

TERRAIN_NAMES = [
    "ocean", "deep_water", "water",
    "beach", "sandy", "grassland",
    "forest", "rocky", "mountains",
]


def compute_behavioral_metrics(
    terrain_visits: torch.Tensor,
    total_moves: torch.Tensor,
    final_state: EnvState,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Compute behavioral metrics averaged over evaluation episodes.

    Args:
        terrain_visits: [num_episodes, 9] count of visits to each terrain type
        total_moves: [num_episodes] moves per episode
        final_state: final EnvState from evaluation

    Returns:
        (scalars, distributions) — scalars dict of behavioral/ prefixed means,
        distributions dict of per-episode tensors for violin plots.
    """
    moves_safe = total_moves.clamp(min=1).unsqueeze(1)  # [N, 1]
    visit_pct = terrain_visits / moves_safe  # [N, 9]

    scalars: dict[str, float] = {}
    for i, name in enumerate(TERRAIN_NAMES):
        scalars[f"behavioral/terrain_{name}_pct"] = visit_pct[:, i].mean().item()

    resource_held = final_state.resources       # [N]
    hp_score = final_state.hp                    # [N]

    scalars["behavioral/resource_held"] = resource_held.mean().item()
    scalars["behavioral/hp_score"] = hp_score.mean().item()

    distributions = {
        "resource_held": resource_held.cpu(),
        "hp_score": hp_score.cpu(),
    }
    return scalars, distributions


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
        self._terrain_history: dict[str, list[dict]] = {}
        self._heatmap_history: dict[str, list[tuple[int, list]]] = {}

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
        prefix: str = "",
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
        key = f"eval-{prefix}/episode_table" if prefix else "eval/episode_table"
        self._run.log({key: table}, step=step)

    def log_density_heatmap(
        self,
        key: str,
        values: np.ndarray,
        step: int,
        y_label: str = "",
        n_bins: int = 40,
    ) -> None:
        """Log a 2D histogram density heatmap that accumulates history over training."""
        if not self.enabled or self._run is None:
            return
        import wandb
        try:
            import plotly.graph_objects as go
        except ImportError:
            return

        if key not in self._heatmap_history:
            self._heatmap_history[key] = []
        self._heatmap_history[key].append((step, values.tolist()))

        # Collect all values to determine global Y range
        all_vals = []
        for _, vals in self._heatmap_history[key]:
            all_vals.extend(vals)
        y_min, y_max = min(all_vals), max(all_vals)
        pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.5
        y_min -= pad
        y_max += pad
        bin_edges = np.linspace(y_min, y_max, n_bins + 1)
        bin_centers = np.round(0.5 * (bin_edges[:-1] + bin_edges[1:]), 2)

        # Build density matrix: rows=bins (Y), cols=steps (X)
        steps_list = []
        z = []
        for s, vals in self._heatmap_history[key]:
            counts, _ = np.histogram(vals, bins=bin_edges)
            total = max(len(vals), 1)
            z.append(counts / total)
            steps_list.append(str(s))

        # z is list of columns → transpose to [n_bins, n_steps]
        z_matrix = np.array(z).T

        fig = go.Figure(data=go.Heatmap(
            x=steps_list,
            y=bin_centers,
            z=z_matrix,
            colorscale=[[0, "blue"], [1, "red"]],
            showscale=True,
        ))
        fig.update_layout(
            xaxis_title="Update",
            showlegend=False,
        )
        if y_label:
            fig.update_layout(yaxis_title=y_label)

        self._run.log({key: wandb.Plotly(fig)}, step=step)

    # Stacking order bottom→top and palette-matched hex colors
    _TERRAIN_COLORS = {
        "ocean":      "#0523E1",
        "deep_water": "#1941E1",
        "water":      "#4169E1",
        "beach":      "#EED6AF",
        "sandy":      "#D2B48C",
        "grassland":  "#228B22",
        "forest":     "#006400",
        "rocky":      "#8B8989",
        "mountains":  "#FFFAFA",
    }

    def log_terrain_distribution(
        self,
        terrain_pcts: dict[str, float],
        step: int,
        mode_prefix: str = "deterministic",
    ) -> None:
        """Log an interactive plotly stacked area chart of terrain visit distribution."""
        if not self.enabled or self._run is None:
            return
        import wandb

        entry = {"step": step}
        entry.update(terrain_pcts)

        if mode_prefix not in self._terrain_history:
            self._terrain_history[mode_prefix] = []
        self._terrain_history[mode_prefix].append(entry)

        try:
            import plotly.graph_objects as go
        except ImportError:
            return

        history = self._terrain_history[mode_prefix]
        steps = [h["step"] for h in history]

        fig = go.Figure()
        for name in TERRAIN_NAMES:
            fig.add_trace(go.Scatter(
                x=steps,
                y=[h[name] for h in history],
                name=name,
                mode="lines",
                stackgroup="one",
                groupnorm="percent",
                line=dict(width=0.5, color=self._TERRAIN_COLORS[name]),
                fill="tonexty",
                fillcolor=self._TERRAIN_COLORS[name],
            ))
        fig.update_layout(
            xaxis_title="Update",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, ticksuffix="%", range=[0, 100]),
            showlegend=True,
        )

        self._run.log(
            {f"behavioral-{mode_prefix}/terrain_distribution": wandb.Plotly(fig)},
            step=step,
        )

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._run.finish()
