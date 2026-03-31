"""Visualization utilities for the interpretability pipeline.

All functions use consistent terrain colors derived from CompiledTerrainData.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import seaborn as sns

# Default style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
sns.set_style("whitegrid")

TERRAIN_NAMES = [
    "Ocean", "Deep Water", "Water", "Beach", "Sandy",
    "Grassland", "Forest", "Rocky", "Mountains",
]

TERRAIN_NAMES_SHORT = [
    "ocean", "deep_water", "water", "beach", "sandy",
    "grassland", "forest", "rocky", "mountains",
]


def get_terrain_colors(compiled) -> np.ndarray:
    """Get terrain colors as [9, 3] float in [0, 1]."""
    return compiled.color_lut.float().cpu().numpy() / 255.0


def get_terrain_cmap(compiled) -> ListedColormap:
    """Build a discrete colormap for terrain types."""
    return ListedColormap(get_terrain_colors(compiled))


def fast_colorize(world_map_np: np.ndarray, compiled) -> np.ndarray:
    """Vectorized heightmap -> [H, W, 3] uint8 RGB."""
    thresholds = compiled.thresholds.cpu().numpy()
    color_lut = compiled.color_lut.cpu().numpy()  # [9, 3] uint8
    indices = np.searchsorted(thresholds, world_map_np).clip(0, compiled.num_terrains - 1)
    return color_lut[indices]


def fast_colorize_float(world_map_np: np.ndarray, compiled) -> np.ndarray:
    """Vectorized heightmap -> [H, W, 3] float in [0, 1]."""
    return fast_colorize(world_map_np, compiled).astype(np.float32) / 255.0


# ── Core plots ───────────────────────────────────────────────────────────────

def plot_trajectory_on_map(
    world_map_np: np.ndarray,
    trajectory: list[tuple[int, int]] | np.ndarray,
    compiled,
    spawn: tuple[int, int] | None = None,
    target: tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 10),
) -> plt.Figure:
    """Plot trajectory overlaid on terrain map."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    rgb = fast_colorize_float(world_map_np, compiled)
    ax.imshow(rgb, origin="upper", interpolation="nearest")

    pos = np.array(trajectory)
    if len(pos) >= 2:
        ax.plot(pos[:, 1], pos[:, 0], "r-", linewidth=1.2, alpha=0.8)

    if spawn is not None:
        ax.scatter(spawn[1], spawn[0], c="lime", s=120, marker="o",
                   edgecolors="k", linewidth=1.5, zorder=5, label="Start")
    elif len(pos) > 0:
        ax.scatter(pos[0, 1], pos[0, 0], c="lime", s=120, marker="o",
                   edgecolors="k", linewidth=1.5, zorder=5, label="Start")

    if target is not None:
        ax.scatter(target[1], target[0], c="gold", s=160, marker="*",
                   edgecolors="k", linewidth=1.5, zorder=5, label="Target")

    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_axis_off()
    return fig


def plot_value_function_overlay(
    world_map_np: np.ndarray,
    value_grid: np.ndarray,
    positions_grid: np.ndarray,
    compiled,
    trajectory: list[tuple[int, int]] | np.ndarray | None = None,
    target: tuple[int, int] | None = None,
    spawn: tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "RdYlGn",
    alpha: float = 0.6,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 10),
) -> plt.Figure:
    """Overlay V(s) heatmap on terrain map.

    Args:
        value_grid: [N_positions] array of values.
        positions_grid: [N_positions, 2] array of (row, col) positions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    rgb = fast_colorize_float(world_map_np, compiled)
    ax.imshow(rgb, alpha=0.4, origin="upper", interpolation="nearest")

    # Scatter plot of values at grid positions
    sc = ax.scatter(
        positions_grid[:, 1], positions_grid[:, 0],
        c=value_grid, cmap=cmap, alpha=alpha, s=8, edgecolors="none",
        vmin=np.nanpercentile(value_grid, 2),
        vmax=np.nanpercentile(value_grid, 98),
    )
    plt.colorbar(sc, ax=ax, label="V(s)", shrink=0.8)

    if trajectory is not None:
        pos = np.array(trajectory)
        ax.plot(pos[:, 1], pos[:, 0], "k-", linewidth=1.5, alpha=0.8)

    if spawn is not None:
        ax.scatter(spawn[1], spawn[0], c="lime", s=120, marker="o",
                   edgecolors="k", linewidth=1.5, zorder=5)
    if target is not None:
        ax.scatter(target[1], target[0], c="gold", s=160, marker="*",
                   edgecolors="k", linewidth=1.5, zorder=5)

    if title:
        ax.set_title(title, fontweight="bold")
    ax.set_axis_off()
    return fig


def plot_activation_bar(
    delta: np.ndarray,
    title: str = "Activation Difference",
    top_k: int = 50,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Bar chart of top activation differences by magnitude."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Select top-k by absolute value
    order = np.argsort(np.abs(delta))[::-1][:top_k]
    vals = delta[order]

    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors, width=0.8)
    ax.set_xlabel("Neuron index (sorted by |delta|)")
    ax.set_ylabel("Activation difference")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(0, len(vals), max(1, len(vals) // 10)))
    ax.set_xticklabels([str(order[i]) for i in range(0, len(vals), max(1, len(vals) // 10))],
                       fontsize=8, rotation=45)
    ax.axhline(0, color="k", linewidth=0.5)
    return fig


def plot_umap_embedding(
    embedding_2d: np.ndarray,
    labels: np.ndarray | None = None,
    continuous_color: np.ndarray | None = None,
    cmap: str = "viridis",
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 8),
    legend: bool = True,
) -> plt.Figure:
    """UMAP scatter with cluster or continuous coloring."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if continuous_color is not None:
        sc = ax.scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=continuous_color, cmap=cmap, s=20, alpha=0.7, edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, shrink=0.8)
    elif labels is not None:
        unique_labels = np.unique(labels)
        palette = plt.cm.get_cmap("tab10", len(unique_labels))
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            label_str = f"Cluster {lab}" if lab >= 0 else "Noise"
            ax.scatter(
                embedding_2d[mask, 0], embedding_2d[mask, 1],
                c=[palette(i)], s=20, alpha=0.7, label=label_str, edgecolors="none",
            )
        if legend:
            ax.legend(fontsize=9, markerscale=2)
    else:
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=20, alpha=0.7, edgecolors="none")

    if title:
        ax.set_title(title, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    return fig


def plot_trajectory_gallery(
    trajectories: list[list[tuple[int, int]] | np.ndarray],
    world_maps_np: list[np.ndarray],
    compiled,
    spawns: list[tuple[int, int]] | None = None,
    targets: list[tuple[int, int]] | None = None,
    titles: list[str] | None = None,
    ncols: int = 4,
    figsize: tuple[float, float] = (20, 15),
) -> plt.Figure:
    """Grid of trajectory-on-map plots."""
    n = len(trajectories)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(n):
        spawn = spawns[i] if spawns else None
        target = targets[i] if targets else None
        title = titles[i] if titles else None
        plot_trajectory_on_map(
            world_maps_np[i], trajectories[i], compiled,
            spawn=spawn, target=target, ax=axes[i], title=title,
        )

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    return fig


def plot_linear_probe_accuracy(
    results: dict[str, dict[str, dict[str, float]]],
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Bar chart of linear probe test accuracy per layer per concept.

    Args:
        results: concept_name → layer_name → {"test_acc": float, ...}
    """
    fig, ax = plt.subplots(figsize=figsize)

    concepts = list(results.keys())
    all_layers = []
    for c in concepts:
        for l in results[c]:
            if l not in all_layers:
                all_layers.append(l)

    x = np.arange(len(all_layers))
    width = 0.8 / max(len(concepts), 1)

    for i, concept in enumerate(concepts):
        accs = [results[concept].get(l, {}).get("test_acc", 0) for l in all_layers]
        offset = (i - len(concepts) / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=concept, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_layers, rotation=30, ha="right")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Linear Probe Accuracy by Layer and Concept", fontweight="bold")
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="Chance (binary)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    return fig


def plot_activation_over_trajectory(
    pc_timeseries: np.ndarray,
    terrain_types_per_step: np.ndarray,
    compiled,
    events: dict[str, list[int]] | None = None,
    pc_labels: list[str] | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """PCA components over time with terrain-colored background strips."""
    n_pcs = pc_timeseries.shape[1]
    fig, axes = plt.subplots(n_pcs, 1, figsize=(figsize[0], figsize[1] * n_pcs),
                              sharex=True)
    if n_pcs == 1:
        axes = [axes]

    terrain_colors = get_terrain_colors(compiled)
    T = len(terrain_types_per_step)

    for pc_idx, ax in enumerate(axes):
        # Background colored by terrain
        for t in range(T):
            t_idx = int(terrain_types_per_step[t])
            if 0 <= t_idx < len(terrain_colors):
                ax.axvspan(t - 0.5, t + 0.5, color=terrain_colors[t_idx], alpha=0.25)

        ax.plot(pc_timeseries[:, pc_idx], "k-", linewidth=1.0)

        label = pc_labels[pc_idx] if pc_labels else f"PC{pc_idx + 1}"
        ax.set_ylabel(label)

        # Mark events
        if events:
            markers = {"target_enters_view": ("v", "gold"),
                       "water_entry": ("s", "blue"),
                       "forest_entry": ("^", "green"),
                       "low_hp": ("x", "red")}
            for ename, steps in events.items():
                mk, col = markers.get(ename, ("o", "gray"))
                for s in steps:
                    if 0 <= s < T:
                        ax.axvline(s, color=col, alpha=0.5, linewidth=0.8, linestyle="--")

    axes[-1].set_xlabel("Step")
    fig.suptitle("Activation PCs over Trajectory", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_rdm(
    rdm_matrix: np.ndarray,
    labels: list[str],
    title: str = "Representational Dissimilarity Matrix",
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8, 7),
) -> plt.Figure:
    """Representational dissimilarity matrix heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(rdm_matrix, cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    plt.colorbar(im, ax=ax, label="Cosine Distance", shrink=0.8)
    ax.set_title(title, fontweight="bold")
    return fig


def plot_correlation_matrix(
    df: "pd.DataFrame",
    columns: list[str] | None = None,
    title: str = "Behavioral Metric Correlations (Spearman)",
    figsize: tuple[float, float] = (12, 10),
) -> plt.Figure:
    """Spearman correlation heatmap."""
    if columns:
        df = df[columns]
    corr = df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_activation_patching_results(
    neuron_indices: np.ndarray,
    kl_values: np.ndarray,
    title: str = "Activation Patching: Per-Neuron Causal Effect",
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Bar chart of per-neuron KL divergence from activation patching."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(neuron_indices)), kl_values, color="#3498db", alpha=0.85)
    ax.set_xticks(range(len(neuron_indices)))
    ax.set_xticklabels([str(i) for i in neuron_indices], fontsize=8, rotation=45)
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("KL Divergence")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig
