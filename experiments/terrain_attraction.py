#!/usr/bin/env python3
"""Terrain Attraction Experiment.

Assesses how two frozen PPO agents (MLP 1M, RNN 250K) are attracted to forest
and mountain tiles by measuring trajectory deviation from a straight-line
baseline on controlled grassland maps.

Varies:
  - Terrain patch type: forest, mountain
  - Perpendicular offset of patch from spawn→target line
  - Starting HP and resources

Outputs:
  - Trajectory grid figures
  - Attraction score heatmaps
  - Value function overlays
  - Summary CSV
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cogniland.env.custom_maps import (
    OCEAN, GRASSLAND, FOREST, MOUNTAINS, SIZE,
    _canvas, _circle, _to_tensor,
)
from cogniland.env.types import AgentConfig, EnvConfig, MapGenConfig, MinimapConfig, TerrainDef
from cogniland.env.wrappers import BatchedIslandEnv
from cogniland.env.core import compute_minimap_batch, compute_terrain_levels

# Terrain definitions matching training YAML (configs/env/default.yaml)
TRAINING_TERRAINS = (
    TerrainDef("ocean",      0.007,  1.0,  -1.0,  0.0, 16, (5,35,225),      ("water",)),
    TerrainDef("deep_water", 0.025,  1.25, -0.5,  0.0, 12, (25,65,225),     ("water",)),
    TerrainDef("water",      0.05,   1.5,  -0.2,  0.0, 10, (65,105,225),    ("water",)),
    TerrainDef("beach",      0.06,   1.75, -1.0,  0.0,  7, (238,214,175),   ("land",)),
    TerrainDef("sandy",      0.1,    2.0,  -1.0,  0.0,  7, (210,180,140),   ("land",)),
    TerrainDef("grassland",  0.25,   2.25, -1.0,  0.0,  7, (34,139,34),     ("land",)),
    TerrainDef("forest",     0.6,    3.0,   5.0,  5.0,  5, (0,100,0),       ("land","forest")),
    TerrainDef("rocky",      0.7,    3.5,  -2.0,  0.0, 10, (139,137,137),   ("land",)),
    TerrainDef("mountains",  1.0,    4.0,  -5.0,  0.0, 22, (255,250,250),   ("land",)),
)
from cogniland.models.ppo import ActorCritic
from cogniland.models.recurrent_ppo import RecurrentActorCritic
from interpretability.viz import (
    fast_colorize_float,
    plot_trajectory_on_map,
    plot_value_function_overlay,
)

# ── Constants ─────────────────────────────────────────────────────────────

SPAWN = (125, 90)
TARGET = (125, 140)
PATCH_RADIUS = 6
TERRAIN_TYPES = ["forest", "mountain"]
TERRAIN_VAL = {"forest": FOREST, "mountain": MOUNTAINS}

# Per-terrain offset grids (different semantics)
FOREST_OFFSETS = [10, 12, 14, 16, 18, 20]   # perpendicular distance to forest
MOUNTAIN_OFFSETS = [2, 4, 8, 16, 32]         # vertical sigma of mountain barrier
OFFSETS_BY_TERRAIN = {"forest": FOREST_OFFSETS, "mountain": MOUNTAIN_OFFSETS}

# Forest patch is shifted 10 tiles left (closer to agent spawn)
FOREST_PATCH_COL = 115
# Mountain is centered on the path midpoint
MTN_CENTER = (125, 115)
MTN_SIGMA_X = 6  # horizontal extent (fixed)
# Small forest Gaussian placed just right of mountain's rocky edge
MTN_FOREST_SIGMA = 2  # ~4 tiles vertical extent
MTN_FOREST_COL_GAP = 0  # hug the rocky edge

# Terrain indices for mountain step counting (matches TRAINING_TERRAINS order)
ROCKY_IDX = 7
MOUNTAINS_IDX = 8

# Starting conditions: (hp, resources)
# HP never < 100 if resources > 0
CONDITIONS = [
    (100, 100),
    (100, 50),
    (100, 20),
    (100, 0),
    (80, 0),
]

CONDITION_LABELS = [
    "HP=100 R=100",
    "HP=100 R=50",
    "HP=100 R=20",
    "HP=100 R=0",
    "HP=80 R=0",
]

# ── Map generation ────────────────────────────────────────────────────────

def _gaussian_patch(
    canvas: np.ndarray, cy: float, cx: float,
    sigma: float, terrain_type: str,
    sigma_y: float | None = None,
    sigma_x: float | None = None,
) -> None:
    """Paint a smooth Gaussian terrain patch onto the canvas.

    Uses two overlaid Gaussians to simulate natural Perlin-noise terrain:
      - A wide base Gaussian (3× sigma) that gently elevates surrounding
        grassland above the flat baseline, creating a smooth hill.
      - A sharp peak Gaussian (1× sigma) that pushes the center into
        the target terrain type (forest or mountains).

    Supports elliptical Gaussians via sigma_y / sigma_x overrides.
    """
    sy = sigma_y if sigma_y is not None else sigma
    sx = sigma_x if sigma_x is not None else sigma

    Y, X = np.ogrid[:SIZE, :SIZE]
    d2_peak = ((Y - cy) / sy) ** 2 + ((X - cx) / sx) ** 2
    # For mountains, ensure base is wide enough for visible grassland gradient
    min_base = 30.0 if terrain_type == "mountain" else 0.0
    base_sy = max(sy * 3.0, min_base)
    base_sx = max(sx * 3.0, min_base)
    d2_base = ((Y - cy) / base_sy) ** 2 + ((X - cx) / base_sx) ** 2

    # Wide base: raises surrounding grassland slightly
    base_gauss = np.exp(-d2_base / 2).astype(np.float32)
    # Peak: pushes center into target terrain
    peak_gauss = np.exp(-d2_peak / 2).astype(np.float32)

    if terrain_type == "forest":
        target_height = FOREST
    elif terrain_type == "mountain":
        target_height = MOUNTAINS
    else:
        return

    # Base elevates grassland; peak reaches target terrain height
    base_ceiling = GRASSLAND + 0.4 * (target_height - GRASSLAND)
    base_vals = GRASSLAND + (base_ceiling - GRASSLAND) * base_gauss
    peak_vals = GRASSLAND + (target_height - GRASSLAND) * peak_gauss

    # Combine: take the max of base and peak at each cell
    patch_vals = np.maximum(base_vals, peak_vals)

    # Only overwrite where patch is higher than current canvas
    mask = patch_vals > canvas
    canvas[mask] = patch_vals[mask]


def make_map(offset: int | None = None, terrain_type: str | None = None) -> torch.Tensor:
    """Create a grassland map with optional smooth Gaussian terrain patch.

    Forest: circular patch shifted left (closer to agent) at perpendicular
            distance `offset` above the spawn→target line.
    Mountain: elliptical barrier centered on path midpoint with vertical
              sigma = `offset` (agent must go around or through).
    """
    canvas = _canvas()
    _circle(canvas, 125, 125, 110, GRASSLAND)

    if offset is not None and terrain_type is not None:
        if terrain_type == "forest":
            # Circular patch, offset rows above path, shifted left toward agent
            patch_row = 125 - offset
            _gaussian_patch(canvas, patch_row, FOREST_PATCH_COL, PATCH_RADIUS, "forest")
        elif terrain_type == "mountain":
            ocean_mask = canvas < 0.06
            cy, cx = MTN_CENTER
            _gaussian_patch(
                canvas, cy, cx, PATCH_RADIUS, "mountain",
                sigma_y=offset, sigma_x=MTN_SIGMA_X,
            )
            # Compress the base-Gaussian hill into grassland range so
            # terrain rises smoothly but never enters forest (0.25–0.6).
            # Values above 0.6 (rocky/mountain) stay untouched.
            land = ~ocean_mask
            rocky_threshold = 0.6
            grassland_ceil = 0.249  # just below forest (0.25)
            in_hill = (canvas > GRASSLAND) & (canvas <= rocky_threshold) & land
            t = (canvas[in_hill] - GRASSLAND) / (rocky_threshold - GRASSLAND)
            canvas[in_hill] = GRASSLAND + t * (grassland_ceil - GRASSLAND)
            canvas[ocean_mask] = OCEAN
            # Small forest Gaussian attached to rocky edge (right side)
            # Find rightmost rocky cell on path row to place forest just past it
            rocky_end = cx + MTN_SIGMA_X + MTN_FOREST_COL_GAP
            for c_scan in range(cx, min(cx + 40, SIZE)):
                if canvas[cy, c_scan] <= grassland_ceil and c_scan > cx:
                    rocky_end = c_scan
                    break
            forest_cx = rocky_end + MTN_FOREST_COL_GAP + MTN_FOREST_SIGMA
            _gaussian_patch(canvas, cy, forest_cx, MTN_FOREST_SIGMA, "forest")

    return _to_tensor(canvas)


# ── Environment factory ──────────────────────────────────────────────────

def make_env(
    world_map: torch.Tensor,
    init_hp: float,
    init_resources: float,
    device: str,
) -> BatchedIslandEnv:
    agent_cfg = AgentConfig(
        init_hp=init_hp,
        max_hp=100.0,
        init_resources=init_resources,
        max_resources=100.0,
    )
    config = EnvConfig(
        agent=agent_cfg,
        minimap=MinimapConfig(max_ray=22, occlude=True),
        terrains=TRAINING_TERRAINS,
        device=device,
    )
    env = BatchedIslandEnv(config, num_envs=1, world_maps=world_map.unsqueeze(0))
    env.env._fixed_spawn = SPAWN
    env.env._fixed_target = TARGET
    env.env.compass_noise_deg = 0.0
    return env


# ── Model loading ─────────────────────────────────────────────────────────

def load_mlp_model(device: str) -> ActorCritic:
    model = ActorCritic(
        scalar_dim=5, minimap_channels=3, hidden_dim=448,
        action_dim=5, cnn_channels=64, cnn_out_spatial=5, scalar_hidden=128,
    )
    ckpt = torch.load(
        PROJECT_ROOT / "artifacts" / "ppo_1m_uw4aeis5" / "ckpt_best.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_rnn_model(device: str) -> RecurrentActorCritic:
    model = RecurrentActorCritic(
        scalar_dim=5, minimap_channels=3, hidden_dim=256, rnn_hidden_dim=64,
        action_dim=5, cnn_channels=32, cnn_out_spatial=4, scalar_hidden=64,
    )
    ckpt = torch.load(
        PROJECT_ROOT / "artifacts" / "ppo_rnn_250k_aqs6s31v" / "ckpt_best.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ── Episode runner ────────────────────────────────────────────────────────

@dataclass
class EpisodeData:
    trajectory: list[tuple[int, int]]
    resources: list[float]
    hp: list[float]
    reached: bool
    steps: int
    observed_mask: np.ndarray  # [H, W] bool — cells seen during the episode
    terrain_trace: list[int] | None = None  # terrain_idx at each step


def _accumulate_visibility(
    vis_counts: np.ndarray,
    position: torch.Tensor,
    minimap: torch.Tensor,
    max_ray: int,
    H: int,
    W: int,
) -> None:
    """Stamp the minimap visibility channel onto the global vis_counts array."""
    pos = position[0].cpu()
    vis = minimap[0, 2].cpu().numpy()  # channel 2 = visibility mask
    D = 2 * max_ray + 1
    pr, pc = int(pos[0].item()), int(pos[1].item())
    for dy in range(-max_ray, max_ray + 1):
        for dx in range(-max_ray, max_ray + 1):
            wr, wc = pr + dy, pc + dx
            if 0 <= wr < H and 0 <= wc < W:
                my, mx = dy + max_ray, dx + max_ray
                if vis[my, mx] > 0.5:
                    vis_counts[wr, wc] = 1


def run_episode(
    env: BatchedIslandEnv,
    model: ActorCritic | RecurrentActorCritic,
    is_rnn: bool,
    device: str,
    max_steps: int = 1000,
) -> EpisodeData:
    obs = env.reset(seed=42)
    h = None
    if is_rnn:
        h = model.init_hidden(1, torch.device(device))

    H = W = env.config.size
    max_ray = env.config.minimap_max_ray
    vis_counts = np.zeros((H, W), dtype=np.uint8)

    trajectory = [tuple(env.state.position[0].cpu().tolist())]
    resources = [env.state.resources[0].item()]
    hp_trace = [env.state.hp[0].item()]
    terrain_trace = [int(env.state.terrain_idx[0].item())]
    reached = False

    # Record initial visibility
    _accumulate_visibility(vis_counts, env.state.position, env.state.minimap, max_ray, H, W)

    with torch.no_grad():
        for _ in range(max_steps):
            if is_rnn:
                action, h = model.get_deterministic_action(obs, h)
            else:
                action = model.get_deterministic_action(obs)

            obs, reward, done, info = env.step(action)

            if done[0]:
                reached = bool(info.get("reached", torch.zeros(1))[0].item())
                break

            # Record after step (skip if done — env was auto-reset)
            trajectory.append(tuple(env.state.position[0].cpu().tolist()))
            resources.append(env.state.resources[0].item())
            hp_trace.append(env.state.hp[0].item())
            terrain_trace.append(int(env.state.terrain_idx[0].item()))
            _accumulate_visibility(vis_counts, env.state.position, env.state.minimap, max_ray, H, W)

    return EpisodeData(
        trajectory=trajectory,
        resources=resources,
        hp=hp_trace,
        reached=reached,
        steps=len(trajectory) - 1,
        observed_mask=vis_counts > 0,
        terrain_trace=terrain_trace,
    )


# ── Attraction metrics ───────────────────────────────────────────────────

def compute_metrics(ep: EpisodeData, offset: int | None) -> dict[str, float]:
    """Compute attraction metrics from episode data."""
    traj = np.array(ep.trajectory)
    spawn_row = SPAWN[0]

    # Signed perpendicular distance: negative = toward patch (above line)
    perp = traj[:, 0] - spawn_row  # positive = below, negative = above

    # Path length (L1 steps)
    diffs = np.abs(np.diff(traj, axis=0))
    path_length = float(diffs.sum())
    euclidean = np.sqrt((TARGET[0] - SPAWN[0])**2 + (TARGET[1] - SPAWN[1])**2)

    metrics = {
        "mean_perp_dist": float(np.mean(np.abs(perp))),
        "mean_signed_perp": float(np.mean(perp)),  # negative = toward patch
        "max_perp_dist": float(np.max(np.abs(perp))),
        "path_length": path_length,
        "path_ratio": path_length / max(euclidean, 1.0),
        "reached": float(ep.reached),
        "steps": float(ep.steps),
    }

    if offset is not None and offset > 0:
        # Normalized attraction: how much of the offset was "consumed" by deviation
        metrics["normalized_attraction"] = -float(np.mean(perp)) / offset
    else:
        metrics["normalized_attraction"] = 0.0

    # Mountain steps: count steps on rocky (7) or mountains (8) terrain
    if ep.terrain_trace is not None:
        metrics["mountain_steps"] = float(sum(1 for t in ep.terrain_trace if t >= ROCKY_IDX))
    else:
        metrics["mountain_steps"] = 0.0

    return metrics


# ── Value function heatmap ───────────────────────────────────────────────

@torch.no_grad()
def compute_value_heatmap(
    env: BatchedIslandEnv,
    model: ActorCritic | RecurrentActorCritic,
    is_rnn: bool,
    device: str,
    init_resources: float,
    init_hp: float,
    step_size: int = 3,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute V(s) over a grid of land positions.

    Returns:
        positions_grid: [N, 2] array of (row, col)
        value_grid: [N] array of values
    """
    world_map = env.env.world_maps[0]  # [H, W]
    compiled = env.compiled
    target_pos = env.target_pos  # [1, 2]
    H, W = world_map.shape

    # Find land positions on a grid
    rows = np.arange(0, H, step_size)
    cols = np.arange(0, W, step_size)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    candidates = np.stack([rr.ravel(), cc.ravel()], axis=1)

    # Filter to land
    heights = world_map.cpu().numpy()
    land_thresh = compiled.land_threshold.item() if hasattr(compiled.land_threshold, 'item') else compiled.land_threshold
    mask = heights[candidates[:, 0], candidates[:, 1]] > land_thresh
    positions = candidates[mask]

    if len(positions) == 0:
        return np.zeros((0, 2)), np.zeros(0)

    all_values = []
    num_terrains = compiled.num_terrains

    for start in range(0, len(positions), batch_size):
        batch_pos = positions[start:start + batch_size]
        B = len(batch_pos)

        pos_t = torch.tensor(batch_pos, dtype=torch.long, device=device)
        target_batch = target_pos.expand(B, 2)
        map_batch = world_map.unsqueeze(0).expand(B, H, W)

        terrain_idx = compute_terrain_levels(map_batch, pos_t, compiled)
        minimap = compute_minimap_batch(
            map_batch, pos_t,
            env.config.minimap_max_ray, terrain_idx,
            env.config.minimap_occlude,
            env.config.minimap_clear_tolerance,
            compiled, target_pos=target_batch,
        )

        # Compass: unit direction toward target
        compass_raw = (target_batch - pos_t).float()
        compass_norm = torch.norm(compass_raw, dim=1, keepdim=True).clamp(min=1e-8)
        compass_unit = compass_raw / compass_norm

        scalars = torch.stack([
            compass_unit[:, 0],
            compass_unit[:, 1],
            terrain_idx / max(num_terrains - 1, 1),
            torch.full((B,), init_resources / 100.0, device=device),
            torch.full((B,), init_hp / 100.0, device=device),
        ], dim=1)

        obs = {"scalars": scalars, "minimap": minimap}

        if is_rnn:
            h = model.init_hidden(B, torch.device(device))
            values = model.get_value(obs, h)
        else:
            values = model.get_value(obs)

        all_values.append(values.cpu().numpy())

    return positions, np.concatenate(all_values)


# ── Visualization ─────────────────────────────────────────────────────────

def _draw_gradient_trajectory(ax, traj_list: list[tuple[int, int]], reached: bool):
    """Draw trajectory with inferno gradient based on visit count per cell.

    Dots at each visited cell center make positions unambiguous.
    """
    traj = np.array(traj_list, dtype=float)
    if len(traj) < 2:
        return

    visit_counts = Counter(traj_list)

    # Dots at each unique visited cell, colored by visit count
    unique_cells = np.array(list(visit_counts.keys()), dtype=float)
    cell_visits = np.array([visit_counts[k] for k in visit_counts], dtype=float)
    norm = Normalize(vmin=1, vmax=max(cell_visits.max(), 2))
    ax.scatter(unique_cells[:, 1], unique_cells[:, 0], c=cell_visits,
               cmap="inferno", norm=norm, s=3, zorder=4, edgecolors="none")

    # Thin connecting line in chronological order
    ax.plot(traj[:, 1], traj[:, 0], color="white", linewidth=0.4, alpha=0.5, zorder=3)

    # Death marker
    if not reached:
        last = traj[-1]
        ax.scatter(last[1], last[0], c="red", s=25, marker="X",
                   alpha=0.5, zorder=10, linewidths=0.3)


def plot_trajectory_grid(
    results: dict,
    world_maps: dict,
    compiled,
    model_name: str,
    terrain_type: str,
    output_dir: str,
):
    """Plot grid: rows=conditions, cols=[baseline, offset_0, ...]."""
    offsets = OFFSETS_BY_TERRAIN[terrain_type]
    col_keys = ["baseline"] + offsets
    n_rows = len(CONDITIONS)
    n_cols = len(col_keys)

    # Dynamic crop region
    if terrain_type == "mountain":
        max_detour = max(offsets)
        r_lo = max(0, 125 - max_detour - 10)
        r_hi = min(250, 125 + max_detour + 10)
    else:
        r_lo = max(0, SPAWN[0] - max(offsets) - PATCH_RADIUS - 10)
        r_hi = min(250, SPAWN[0] + 25)
    c_lo = max(0, SPAWN[1] - 15)
    c_hi = min(250, TARGET[1] + 15)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows))

    for i, ((hp, res), label) in enumerate(zip(CONDITIONS, CONDITION_LABELS)):
        for j, col_key in enumerate(col_keys):
            ax = axes[i, j]
            key = (col_key, hp, res)
            if key not in results:
                ax.set_visible(False)
                continue

            ep = results[key]
            wm_key = col_key if col_key != "baseline" else "baseline"
            wm_np = world_maps[wm_key].cpu().numpy()

            rgb = fast_colorize_float(wm_np, compiled)
            fog = np.where(ep.observed_mask[:, :, None], 1.0, 0.35)
            rgb = rgb * fog
            ax.imshow(rgb, origin="upper", interpolation="nearest")

            # Gradient trajectory: yellow (1 visit) → red (many visits)
            _draw_gradient_trajectory(ax, ep.trajectory, ep.reached)

            ax.scatter(SPAWN[1], SPAWN[0], c="lime", s=60, marker="o",
                       edgecolors="k", linewidth=1.0, zorder=5)
            ax.scatter(TARGET[1], TARGET[0], c="gold", s=80, marker="*",
                       edgecolors="k", linewidth=1.0, zorder=5)

            ax.plot([SPAWN[1], TARGET[1]], [SPAWN[0], TARGET[0]],
                    "w--", linewidth=0.8, alpha=0.6)

            ax.set_xlim(c_lo, c_hi)
            ax.set_ylim(r_hi, r_lo)
            ax.set_axis_off()

            # Status annotation
            status = "OK" if ep.reached else f"DIED ({ep.steps})"
            color = "green" if ep.reached else "red"
            ax.text(0.98, 0.02, status, transform=ax.transAxes,
                    fontsize=7, ha="right", va="bottom", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

            if i == 0:
                if col_key == "baseline":
                    title = "Baseline"
                elif terrain_type == "mountain":
                    title = f"σ_v={col_key}"
                else:
                    title = f"d={col_key}"
                ax.set_title(title, fontsize=11, fontweight="bold")

        axes[i, 0].text(-0.15, 0.5, label, transform=axes[i, 0].transAxes,
                        fontsize=9, ha="right", va="center", fontweight="bold")

    fig.suptitle(f"{model_name} — {terrain_type} attraction", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0.08, 0, 1, 0.98])
    subdir = os.path.join(output_dir, terrain_type)
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, f"{model_name}_trajectories.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_merged_attraction_heatmap(
    all_metrics: list[dict],
    terrain_type: str,
    output_dir: str,
):
    """Side-by-side heatmaps for MLP and RNN with shared color scale."""
    offsets = OFFSETS_BY_TERRAIN[terrain_type]
    n_conds = len(CONDITIONS)
    n_offsets = len(offsets)
    model_names = ["mlp_1m", "rnn_250k"]

    if terrain_type == "mountain":
        metric_key = "mountain_steps"
        cbar_label = "Steps on mountain terrain"
        title_metric = "Steps on mountain terrain"
    else:
        metric_key = "mean_perp_dist"
        cbar_label = "Mean |perp dist| (cells)"
        title_metric = "Mean |perp distance|"

    matrices = {}
    for mname in model_names:
        matrix = np.full((n_conds, n_offsets), np.nan)
        for row in all_metrics:
            if row["model"] != mname or row["terrain"] != terrain_type:
                continue
            if row["offset"] == "baseline":
                continue
            offset = int(row["offset"])
            hp = int(row["hp"])
            res = int(row["resources"])
            try:
                ci = CONDITIONS.index((hp, res))
                oi = offsets.index(offset)
            except ValueError:
                continue
            matrix[ci, oi] = row[metric_key]
        matrices[mname] = matrix

    all_vals = np.concatenate([m.ravel() for m in matrices.values()])
    vmin = np.nanmin(all_vals) if not np.all(np.isnan(all_vals)) else 0
    vmax = np.nanmax(all_vals) if not np.all(np.isnan(all_vals)) else 1

    import pandas as pd
    col_labels = [str(o) for o in offsets]
    xlabel = "Mountain vertical radius (σ_v)" if terrain_type == "mountain" else "Distance to forest (cells)"

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 5))
    for ax, mname in zip([ax_l, ax_r], model_names):
        df = pd.DataFrame(matrices[mname], index=CONDITION_LABELS, columns=col_labels)
        import seaborn as sns
        sns.heatmap(df, ax=ax, vmin=vmin, vmax=vmax, cmap="viridis",
                    annot=True, fmt=".1f", annot_kws={"size": 8},
                    linewidths=0.5, linecolor="white",
                    cbar=(ax is ax_r), cbar_kws={"label": cbar_label})
        ax.set_xlabel(xlabel)
        ax.set_title(mname, fontweight="bold")

    fig.suptitle(f"{terrain_type.capitalize()} — {title_metric}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    subdir = os.path.join(output_dir, terrain_type)
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, "comparison_attraction_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_comparison(
    all_metrics: list[dict],
    terrain_type: str,
    output_dir: str,
):
    """Line plot comparing both models."""
    offsets = OFFSETS_BY_TERRAIN[terrain_type]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    model_names = ["mlp_1m", "rnn_250k"]

    compact_labels = [f"HP+R={hp + res}" for hp, res in CONDITIONS]

    if terrain_type == "mountain":
        metric_key = "mountain_steps"
        ylabel = "Steps on mountain terrain"
    else:
        metric_key = "mean_perp_dist"
        ylabel = "Mean |perp distance| (cells)"

    for ax, mname in zip(axes, model_names):
        for (hp, res), label in zip(CONDITIONS, compact_labels):
            vals = []
            for offset in offsets:
                for row in all_metrics:
                    if (row["model"] == mname and row["terrain"] == terrain_type
                            and row["offset"] != "baseline"
                            and int(row["offset"]) == offset
                            and int(row["hp"]) == hp
                            and int(row["resources"]) == res):
                        vals.append(row[metric_key])
                        break
                else:
                    vals.append(np.nan)
            ax.plot(offsets, vals, "o-", label=label, markersize=4)

        if terrain_type == "mountain":
            ax.set_xscale("log", base=2)
            ax.set_xticks(offsets)
            ax.set_xticklabels([str(o) for o in offsets])
            ax.set_xlabel("Mountain vertical radius (σ_v)")
        else:
            ax.set_xticks(offsets)
            ax.set_xticklabels([str(o) for o in offsets])
            ax.set_xlabel("Distance to forest (cells)")
        ax.set_ylabel(ylabel)
        ax.set_title(mname, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3, which="major")
        ax.minorticks_off()

    fig.suptitle(f"{terrain_type.capitalize()} comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    subdir = os.path.join(output_dir, terrain_type)
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, "comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_raw_heightmap(world_map: torch.Tensor, compiled, output_dir: str, terrain_type: str = "forest") -> None:
    """Plot the raw heightmap with terrain threshold lines for a single map."""
    hm = world_map.cpu().numpy()
    thresholds = compiled.thresholds.cpu().numpy()
    terrain_names = compiled.terrain_names

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: raw heightmap
    ax = axes[0]
    im = ax.imshow(hm, origin="upper", cmap="terrain", interpolation="nearest")
    ax.scatter(SPAWN[1], SPAWN[0], c="lime", s=120, marker="o",
               edgecolors="k", linewidth=1.5, zorder=5, label="Spawn")
    ax.scatter(TARGET[1], TARGET[0], c="gold", s=160, marker="*",
               edgecolors="k", linewidth=1.5, zorder=5, label="Target")
    ax.set_title("Raw heightmap", fontweight="bold")
    ax.legend(fontsize=9)
    plt.colorbar(im, ax=ax, label="Height", shrink=0.8)

    # Right: heightmap profile along a vertical slice through patch center
    ax = axes[1]
    mid_col = (SPAWN[1] + TARGET[1]) // 2
    profile = hm[:, mid_col]
    rows = np.arange(len(profile))
    ax.plot(rows, profile, "k-", linewidth=1.5)
    # Draw threshold lines
    for i, (thr, name) in enumerate(zip(thresholds, terrain_names)):
        if thr < 0.001 or thr > 0.9:
            continue
        ax.axhline(thr, color=f"C{i}", linestyle="--", alpha=0.6, linewidth=0.8)
        ax.text(len(profile) - 2, thr + 0.005, name, fontsize=7,
                ha="right", color=f"C{i}")
    ax.axvline(SPAWN[0], color="lime", linestyle=":", linewidth=1.5, label="Spawn row")
    ax.set_xlabel("Row")
    ax.set_ylabel("Height")
    ax.set_title(f"Vertical profile at col={mid_col}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    subdir = os.path.join(output_dir, terrain_type)
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, "raw_heightmap.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_value_comparison(
    models: list[tuple[str, "ActorCritic | RecurrentActorCritic", bool]],
    envs_by_res: dict[int, BatchedIslandEnv],
    episodes_by_key: dict[tuple[str, int], EpisodeData],
    terrain_type: str,
    offset: int,
    hp: float,
    resource_levels: list[int],
    device: str,
    output_dir: str,
):
    """4-panel value comparison: rows=models, cols=resource levels."""
    n_rows = len(models)
    n_cols = len(resource_levels)

    # Use first env for shared terrain map
    first_env = list(envs_by_res.values())[0]
    compiled = first_env.compiled
    world_map_np = first_env.env.world_maps[0].cpu().numpy()
    rgb = fast_colorize_float(world_map_np, compiled)

    # Crop region
    offsets = OFFSETS_BY_TERRAIN[terrain_type]
    if terrain_type == "mountain":
        max_d = max(offsets)
        r_lo = max(0, 125 - max_d - 10)
        r_hi = min(250, 125 + max_d + 10)
    else:
        r_lo = max(0, SPAWN[0] - max(offsets) - PATCH_RADIUS - 10)
        r_hi = min(250, SPAWN[0] + 25)
    c_lo = max(0, SPAWN[1] - 15)
    c_hi = min(250, TARGET[1] + 15)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Collect all values for shared color scale
    all_values = []
    precomputed = {}
    for i, (mname, model, is_rnn) in enumerate(models):
        for j, res in enumerate(resource_levels):
            env = envs_by_res[res]
            positions, values = compute_value_heatmap(
                env, model, is_rnn, device, float(res), hp, step_size=2,
            )
            precomputed[(i, j)] = (positions, values)
            if len(values) > 0:
                all_values.append(values)

    if not all_values:
        plt.close(fig)
        return
    all_v = np.concatenate(all_values)
    vmin = np.nanpercentile(all_v, 1)
    vmax = np.nanpercentile(all_v, 99)

    for i, (mname, model, is_rnn) in enumerate(models):
        for j, res in enumerate(resource_levels):
            ax = axes[i, j]
            positions, values = precomputed[(i, j)]

            ax.imshow(rgb, alpha=0.2, origin="upper", interpolation="nearest")

            if len(positions) > 0:
                sc = ax.scatter(
                    positions[:, 1], positions[:, 0],
                    c=values, cmap="Spectral", alpha=1.0, s=70,
                    edgecolors="none", vmin=vmin, vmax=vmax,
                )

            ep = episodes_by_key.get((mname, res))
            if ep is not None:
                _draw_gradient_trajectory(ax, ep.trajectory, ep.reached)

            ax.scatter(SPAWN[1], SPAWN[0], c="lime", s=80, marker="o",
                       edgecolors="k", linewidth=1.2, zorder=5)
            ax.scatter(TARGET[1], TARGET[0], c="gold", s=100, marker="*",
                       edgecolors="k", linewidth=1.2, zorder=5)
            ax.set_xlim(c_lo, c_hi)
            ax.set_ylim(r_hi, r_lo)
            ax.set_axis_off()

            if i == 0:
                ax.set_title(f"R={res}", fontsize=13, fontweight="bold")
            if j == 0:
                ax.text(-0.08, 0.5, mname, transform=ax.transAxes,
                        fontsize=12, ha="right", va="center", fontweight="bold")

    offset_label = f"d={offset}" if terrain_type == "forest" else f"σ_v={offset}"
    fig.suptitle(
        f"{terrain_type.capitalize()} {offset_label} | HP={int(hp)} — Value function V(s)",
        fontsize=16, fontweight="bold",
    )
    # Shared colorbar
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="V(s)", shrink=0.6, pad=0.03)

    subdir = os.path.join(output_dir, terrain_type)
    os.makedirs(subdir, exist_ok=True)
    fname = f"value_comparison_d{offset}_hp{int(hp)}.png"
    path = os.path.join(subdir, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Terrain Attraction Experiment")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments" / "figures" / "terrain_attraction"))
    parser.add_argument("--skip-value-heatmaps", action="store_true", help="Skip expensive value heatmap computation")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # ── Load models ───────────────────────────────────────────────────
    print("\nLoading models...")
    mlp_model = load_mlp_model(device)
    rnn_model = load_rnn_model(device)
    models = [
        ("mlp_1m", mlp_model, False),
        ("rnn_250k", rnn_model, True),
    ]
    print("  Models loaded.")

    # ── Pre-generate all maps ─────────────────────────────────────────
    print("\nGenerating maps...")
    world_maps: dict = {}
    world_maps["baseline"] = make_map()
    for terrain_type in TERRAIN_TYPES:
        for offset in OFFSETS_BY_TERRAIN[terrain_type]:
            world_maps[(terrain_type, offset)] = make_map(offset, terrain_type)
    print(f"  Generated {len(world_maps)} maps.")

    # Get compiled terrain data from a throwaway env
    tmp_env = make_env(world_maps["baseline"], 100, 100, device)
    tmp_env.reset(seed=0)
    compiled = tmp_env.compiled

    # Plot raw heightmaps for example maps
    print("\nPlotting raw heightmaps...")
    plot_raw_heightmap(world_maps[("forest", 14)], compiled, output_dir, "forest")
    plot_raw_heightmap(world_maps[("mountain", 8)], compiled, output_dir, "mountain")

    # ── Run experiments ───────────────────────────────────────────────
    all_metrics: list[dict] = []
    all_results: dict = {}
    all_envs: dict = {}

    for model_name, model, is_rnn in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        for terrain_type in TERRAIN_TYPES:
            offsets = OFFSETS_BY_TERRAIN[terrain_type]
            print(f"\n  Terrain: {terrain_type}")
            results: dict = {}

            for hp, res in CONDITIONS:
                # Baseline
                wm = world_maps["baseline"]
                env = make_env(wm, hp, res, device)
                ep = run_episode(env, model, is_rnn, device)
                results[("baseline", hp, res)] = ep
                metrics = compute_metrics(ep, None)
                all_metrics.append({
                    "model": model_name, "terrain": terrain_type,
                    "offset": "baseline", "hp": hp, "resources": res,
                    **metrics,
                })
                mtn_info = f" mtn_steps={metrics['mountain_steps']:.0f}" if terrain_type == "mountain" else ""
                print(f"    baseline HP={hp} R={res}: steps={ep.steps} reached={ep.reached} "
                      f"mean_perp={metrics['mean_perp_dist']:.1f}{mtn_info}")

                # With patches
                for offset in offsets:
                    wm = world_maps[(terrain_type, offset)]
                    env = make_env(wm, hp, res, device)
                    ep = run_episode(env, model, is_rnn, device)
                    results[(offset, hp, res)] = ep
                    metrics = compute_metrics(ep, offset)
                    all_metrics.append({
                        "model": model_name, "terrain": terrain_type,
                        "offset": offset, "hp": hp, "resources": res,
                        **metrics,
                    })
                    all_envs[(model_name, terrain_type, offset, hp, res)] = env
                    label = f"σ_v={offset:2d}" if terrain_type == "mountain" else f"d={offset:2d}"
                    mtn_info = f" mtn_steps={metrics['mountain_steps']:.0f}" if terrain_type == "mountain" else ""
                    print(f"    {label} HP={hp} R={res}: steps={ep.steps} reached={ep.reached} "
                          f"mean_perp={metrics['mean_perp_dist']:.1f}{mtn_info}")

            # ── Trajectory grid ───────────────────────────────────────
            print(f"\n  Plotting trajectory grid for {model_name} / {terrain_type}...")
            wm_dict = {"baseline": world_maps["baseline"]}
            for offset in offsets:
                wm_dict[offset] = world_maps[(terrain_type, offset)]
            plot_trajectory_grid(results, wm_dict, compiled, model_name, terrain_type, output_dir)

            all_results[(model_name, terrain_type)] = results

    # ── Merged attraction heatmaps & comparison plots ────────────────
    for terrain_type in TERRAIN_TYPES:
        plot_merged_attraction_heatmap(all_metrics, terrain_type, output_dir)
        plot_comparison(all_metrics, terrain_type, output_dir)

    # ── Value function comparison (4-panel: 2 models × R=0 vs R=100) ─
    if not args.skip_value_heatmaps:
        print("\nComputing value function comparisons...")
        value_offset = 14  # forest d=14
        value_hp = 100.0
        value_resources = [0, 100]

        # Build envs and episode lookups for the comparison
        envs_by_res = {}
        episodes_by_key: dict[tuple[str, int], EpisodeData] = {}
        for res in value_resources:
            wm = world_maps[("forest", value_offset)]
            env = make_env(wm, value_hp, res, device)
            env.reset(seed=42)
            envs_by_res[res] = env

        for model_name, model, is_rnn in models:
            results = all_results.get((model_name, "forest"), {})
            for res in value_resources:
                ep = results.get((value_offset, int(value_hp), res))
                if ep is not None:
                    episodes_by_key[(model_name, res)] = ep

        plot_value_comparison(
            models, envs_by_res, episodes_by_key,
            "forest", value_offset, value_hp, value_resources,
            device, output_dir,
        )

    # ── Save CSV ──────────────────────────────────────────────────────
    # Save per-terrain CSVs
    for terrain_type in TERRAIN_TYPES:
        subdir = os.path.join(output_dir, terrain_type)
        os.makedirs(subdir, exist_ok=True)
        terrain_metrics = [m for m in all_metrics if m["terrain"] == terrain_type]
        if terrain_metrics:
            fieldnames = list(terrain_metrics[0].keys())
            csv_path = os.path.join(subdir, "results.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(terrain_metrics)
            print(f"  Saved CSV: {csv_path}")
    # Also save combined CSV
    csv_path = os.path.join(output_dir, "terrain_attraction_results.csv")
    if all_metrics:
        fieldnames = list(all_metrics[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\nSaved CSV: {csv_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
