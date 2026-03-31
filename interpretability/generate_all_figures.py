#!/usr/bin/env python3
"""Generate all thesis figures from collected trajectory data.

Usage::

    python interpretability/generate_all_figures.py \
        --data-dir interpretability/data/ \
        --output-dir interpretability/figures/ \
        --model-path artifacts/good_old/ckpt_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from cogniland.env.core import compute_minimap_batch, compute_terrain_levels
from cogniland.env.types import EnvConfig
from cogniland.models.ppo import ActorCritic
from cogniland.utils import load_checkpoint

from interpretability.data_manager import TrajectoryDataManager
from interpretability.trajectory_features import featurize_all
from interpretability import probes, viz


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="interpretability/data/")
    p.add_argument("--output-dir", default="interpretability/figures/")
    p.add_argument("--model-path", default="artifacts/good_old/ckpt_best.pt")
    p.add_argument("--test-maps", default="data/test_seed42_n16.pt")
    p.add_argument("--behavioral-maps", default="data/test_behavior.pt")
    p.add_argument("--format", default="pdf", choices=["pdf", "png"])
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _save(fig, name, out_dir, fmt):
    path = Path(out_dir) / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def _infer_arch_from_checkpoint(ckpt_path: str, device: str = "cpu") -> dict:
    """Infer ActorCritic architecture params from checkpoint weight shapes."""
    import math
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    minimap_channels = sd["cnn.0.weight"].shape[1]
    cnn_channels = sd["cnn.3.weight"].shape[0]
    scalar_hidden = sd["scalar_net.0.weight"].shape[0]
    scalar_dim = sd["scalar_net.0.weight"].shape[1]
    hidden_dim = sd["trunk.0.weight"].shape[0]
    trunk_in = sd["trunk.0.weight"].shape[1]
    cnn_out = trunk_in - scalar_hidden
    cnn_out_spatial = int(math.isqrt(cnn_out // cnn_channels))
    action_dim = sd["actor.weight"].shape[0]
    return dict(
        scalar_dim=scalar_dim, minimap_channels=minimap_channels,
        hidden_dim=hidden_dim, action_dim=action_dim,
        cnn_channels=cnn_channels, cnn_out_spatial=cnn_out_spatial,
        scalar_hidden=scalar_hidden,
    )


def _load_model(args):
    arch = _infer_arch_from_checkpoint(args.model_path, args.device)
    print(f"  Architecture: {arch}")
    model = ActorCritic(**arch).to(args.device)
    load_checkpoint(args.model_path, model, device=args.device)
    model.eval()
    return model


def generate_clustering_figures(dm, out_dir, fmt):
    """Figures 1-3: UMAP, clusters, trajectory gallery."""
    import umap
    import hdbscan

    print("\n[1/8] Trajectory clustering...")
    features, feature_names = featurize_all(
        dm.summary, str(dm.h5_path), max_episode_length=1000,
    )

    # PCA → UMAP
    pca = PCA(n_components=min(features.shape[1], features.shape[0]) - 1)
    pca.fit(features)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, 0.95)) + 1
    n_keep = max(n_keep, 2)
    features_pca = pca.transform(features)[:, :n_keep]
    print(f"  PCA: {features.shape[1]} -> {n_keep} components (95% variance)")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(features_pca)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(3, len(features) // 10), min_samples=2)
    labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  HDBSCAN found {n_clusters} clusters")

    # Fig 1: UMAP colored by cluster
    fig = viz.plot_umap_embedding(embedding, labels=labels, title="Strategy Clusters (UMAP + HDBSCAN)")
    _save(fig, "fig_umap_clusters", out_dir, fmt)

    # Fig 2: UMAP colored by metrics (2x3 panel)
    metric_cols = ["directness_ratio", "ocean_usage_ratio", "forest_usage_ratio",
                   "risk_score", "episode_length", "average_hp"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, col in enumerate(metric_cols):
        ax = axes[i // 3][i % 3]
        vals = dm.summary[col].values if col in dm.summary.columns else np.zeros(len(embedding))
        viz.plot_umap_embedding(embedding, continuous_color=vals, title=col, ax=ax)
    fig.tight_layout()
    _save(fig, "fig_umap_metrics", out_dir, fmt)

    # Fig 3: Correlation matrix
    corr_cols = [c for c in ["directness_ratio", "risk_score", "ocean_usage_ratio",
                              "forest_usage_ratio", "episode_length", "average_hp",
                              "average_resources", "map_coverage", "total_return",
                              "min_hp"] if c in dm.summary.columns]
    if corr_cols:
        fig = viz.plot_correlation_matrix(dm.summary, columns=corr_cols)
        _save(fig, "fig_correlation_matrix", out_dir, fmt)

    return embedding, labels, features_pca


def generate_trajectory_gallery(dm, labels, compiled, out_dir, fmt, test_maps_path, beh_maps_path):
    """Fig 4: Representative trajectories per cluster."""
    print("\n[2/8] Trajectory gallery...")
    unique_labels = sorted(set(labels[labels >= 0]))

    trajs = []
    maps_np = []
    spawns = []
    targets = []
    titles = []

    for lab in unique_labels[:6]:
        cluster_ids = dm.summary[labels == lab]["traj_id"].values
        # Pick up to 3 representative (shortest + longest + random)
        for tid in cluster_ids[:3]:
            try:
                wmap, spawn, target = dm.get_map_for_trajectory(
                    int(tid), test_maps_path, beh_maps_path)
                tdata = dm.get_trajectory(int(tid))
                trajs.append(tdata["positions"])
                maps_np.append(wmap)
                spawns.append(spawn)
                targets.append(target)
                outcome = tdata["attrs"].get("outcome", "?")
                titles.append(f"C{lab} T{tid} ({outcome})")
            except Exception:
                continue

    if trajs:
        fig = viz.plot_trajectory_gallery(
            trajs, maps_np, compiled, spawns=spawns, targets=targets,
            titles=titles, ncols=3, figsize=(18, 6 * ((len(trajs) + 2) // 3)),
        )
        _save(fig, "fig_trajectory_gallery", out_dir, fmt)


def generate_value_function_maps(model, dm, compiled, out_dir, fmt, test_maps_path, device):
    """Figs 5-7: Value function overlays on maps."""
    print("\n[3/8] Value function maps...")
    env_config = EnvConfig(device=device)

    test_data = torch.load(test_maps_path, map_location="cpu", weights_only=True)
    test_maps = test_data["maps"]

    # Pick 3 maps that have successful trajectories
    successful = dm.filter(outcome="success", map_source="test")
    if len(successful) == 0:
        successful = dm.filter(map_source="test")
    map_ids = successful["map_id"].unique()[:3]

    for map_idx in map_ids:
        world_map = test_maps[int(map_idx)]
        wm_np = world_map.numpy()

        # Get a trajectory on this map for overlay
        map_trajs = successful[successful["map_id"] == map_idx]
        traj_id = int(map_trajs.iloc[0]["traj_id"]) if len(map_trajs) > 0 else None

        trajectory = None
        target = None
        spawn = None
        if traj_id is not None:
            tdata = dm.get_trajectory(traj_id)
            trajectory = tdata["positions"]
            spawn = (int(tdata["attrs"]["spawn_row"]), int(tdata["attrs"]["spawn_col"]))
            target = (int(tdata["attrs"]["target_row"]), int(tdata["attrs"]["target_col"]))

        # Build grid of positions on land
        land_thresh = compiled.land_threshold
        stride = 5
        land_positions = []
        for r in range(0, 250, stride):
            for c in range(0, 250, stride):
                if wm_np[r, c] > land_thresh:
                    land_positions.append([r, c])

        if len(land_positions) < 10:
            continue

        positions = torch.tensor(land_positions, dtype=torch.long)
        B = positions.shape[0]

        # Default target for compass if we have one
        if target is not None:
            target_t = torch.tensor([list(target)], dtype=torch.long).expand(B, 2)
        else:
            target_t = torch.tensor([[125, 125]], dtype=torch.long).expand(B, 2)

        # Compute observations for all positions in batches
        values = []
        batch_size = 256
        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            pos_batch = positions[start:end]
            b = pos_batch.shape[0]
            wm_batch = world_map.unsqueeze(0).expand(b, -1, -1)
            tgt_batch = target_t[start:end]

            terrain_idx = compute_terrain_levels(wm_batch, pos_batch, compiled)
            minimap = compute_minimap_batch(
                wm_batch, pos_batch, env_config.minimap_max_ray,
                terrain_idx, env_config.minimap_occlude,
                env_config.minimap_clear_tolerance, compiled,
                target_pos=tgt_batch,
            )

            compass_raw = (tgt_batch - pos_batch).float()
            compass_norm = compass_raw / compass_raw.norm(dim=1, keepdim=True).clamp(min=1e-8)

            scalars = torch.stack([
                compass_norm[:, 0], compass_norm[:, 1],
                terrain_idx / max(compiled.num_terrains - 1, 1),
                torch.ones(b) * 0.5,  # resources = 50%
                torch.ones(b) * 1.0,  # hp = 100%
            ], dim=1)

            obs = {"scalars": scalars.to(device), "minimap": minimap.to(device)}
            with torch.no_grad():
                v = model.get_value(obs).cpu().numpy()
            values.append(v)

        value_arr = np.concatenate(values)

        fig = viz.plot_value_function_overlay(
            wm_np, value_arr, positions.numpy(), compiled,
            trajectory=trajectory, target=target, spawn=spawn,
            title=f"Value Function — Map {map_idx}",
        )
        _save(fig, f"fig_value_function_map{map_idx}", out_dir, fmt)


def generate_activation_analysis(dm, compiled, out_dir, fmt):
    """Figs 8-10: Activation trajectory, PCA, terrain coloring."""
    print("\n[4/8] Activation analysis...")

    # Find an interesting trajectory (successful, moderate length)
    successful = dm.filter(outcome="success")
    if len(successful) == 0:
        successful = dm.summary
    # Sort by episode length, pick one in the middle
    sorted_df = successful.sort_values("episode_length")
    mid_idx = len(sorted_df) // 2
    traj_id = int(sorted_df.iloc[mid_idx]["traj_id"])

    tdata = dm.get_trajectory(traj_id)
    trunk_acts = tdata["activations"].get("trunk_2")

    if trunk_acts is not None and len(trunk_acts) > 5:
        acts_f32 = trunk_acts.astype(np.float32)
        if acts_f32.ndim > 2:
            acts_f32 = acts_f32.reshape(acts_f32.shape[0], -1)
        pca = PCA(n_components=3)
        pcs = pca.fit_transform(acts_f32)
        terrain = np.array(tdata.get("terrain_idx", [0] * len(pcs)))

        # Build events dict
        events = {}
        flags = tdata.get("flags", {})
        if "target_just_entered_view" in flags:
            events["target_enters_view"] = list(np.where(flags["target_just_entered_view"])[0])
        if "is_on_water" in flags:
            water = flags["is_on_water"]
            transitions = []
            for i in range(1, len(water)):
                if water[i] and not water[i - 1]:
                    transitions.append(i)
            events["water_entry"] = transitions
        if "is_low_hp" in flags:
            events["low_hp"] = list(np.where(flags["is_low_hp"])[0][:5])

        fig = viz.plot_activation_over_trajectory(
            pcs, terrain, compiled, events=events,
            pc_labels=[f"PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})" for i in range(3)],
        )
        _save(fig, "fig_activation_trajectory", out_dir, fmt)

    # All activations PCA colored by terrain
    print("  Computing global activation PCA...")
    all_acts, all_tids, all_terrain = dm.get_all_activations_flat("trunk_2")
    if len(all_acts) > 100:
        # Subsample if very large
        if len(all_acts) > 50000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(all_acts), 50000, replace=False)
            all_acts = all_acts[idx]
            all_terrain = all_terrain[idx]

        pca_global = PCA(n_components=2)
        embedding = pca_global.fit_transform(all_acts)

        terrain_colors_float = viz.get_terrain_colors(compiled)
        colors = np.array([terrain_colors_float[int(t)] if 0 <= int(t) < 9 else [0.5, 0.5, 0.5]
                           for t in all_terrain])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=5, alpha=0.4, edgecolors="none")
        # Legend
        for i in range(9):
            ax.scatter([], [], c=[terrain_colors_float[i]], s=40, label=viz.TERRAIN_NAMES[i])
        ax.legend(fontsize=9, markerscale=2, loc="upper right")
        ax.set_xlabel(f"PC1 ({pca_global.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca_global.explained_variance_ratio_[1]:.1%})")
        ax.set_title("Trunk.2 Activations (PCA) Colored by Terrain", fontweight="bold")
        _save(fig, "fig_activation_pca_terrain", out_dir, fmt)


def generate_contrast_figures(dm, out_dir, fmt):
    """Figs 11-13: Activation contrasts + linear probes."""
    print("\n[5/8] Activation contrasts...")
    layers = ["trunk_0", "trunk_2", "actor", "critic"]

    all_probe_results = {}

    # Contrast 1: HP/resources
    print("  HP/resources contrast...")
    _, _, delta_hp = probes.compute_activation_contrast(
        dm,
        positive_flags={"is_low_hp": True, "is_low_resources": True},
        negative_flags={"is_low_hp": False, "is_low_resources": False},
        layer="trunk_2",
    )
    if len(delta_hp) > 1:
        fig = viz.plot_activation_bar(delta_hp, title="Low vs High HP+Resources (trunk.2)")
        _save(fig, "fig_contrast_hp_resources", out_dir, fmt)

    hp_probes = probes.run_probes_for_concept(
        dm, "low_hp",
        positive_flags={"is_low_hp": True},
        negative_flags={"is_low_hp": False},
        layers=layers,
    )
    all_probe_results["Low HP"] = hp_probes

    # Contrast 2: Target visible
    print("  Target visibility contrast...")
    _, _, delta_tgt = probes.compute_activation_contrast(
        dm,
        positive_flags={"is_target_visible": True},
        negative_flags={"is_target_visible": False},
        layer="trunk_2",
    )
    if len(delta_tgt) > 1:
        fig = viz.plot_activation_bar(delta_tgt, title="Target Visible vs Not (trunk.2)")
        _save(fig, "fig_contrast_target", out_dir, fmt)

    tgt_probes = probes.run_probes_for_concept(
        dm, "target_visible",
        positive_flags={"is_target_visible": True},
        negative_flags={"is_target_visible": False},
        layers=layers,
    )
    all_probe_results["Target Visible"] = tgt_probes

    # Contrast 3: Forest
    print("  Forest contrast...")
    _, _, delta_forest = probes.compute_activation_contrast(
        dm,
        positive_flags={"is_in_forest": True},
        negative_flags={"is_in_forest": False},
        layer="trunk_2",
    )
    if len(delta_forest) > 1:
        fig = viz.plot_activation_bar(delta_forest, title="In Forest vs Not (trunk.2)")
        _save(fig, "fig_contrast_forest", out_dir, fmt)

    forest_probes = probes.run_probes_for_concept(
        dm, "in_forest",
        positive_flags={"is_in_forest": True},
        negative_flags={"is_in_forest": False},
        layers=layers,
    )
    all_probe_results["Forest"] = forest_probes

    # Contrast 4: Water
    water_probes = probes.run_probes_for_concept(
        dm, "on_water",
        positive_flags={"is_on_water": True},
        negative_flags={"is_on_water": False},
        layers=layers,
    )
    all_probe_results["Water"] = water_probes

    # Linear probe accuracy figure
    print("  Generating linear probe accuracy plot...")
    fig = viz.plot_linear_probe_accuracy(all_probe_results)
    _save(fig, "fig_linear_probe_accuracy", out_dir, fmt)

    return all_probe_results


def generate_rdm_figures(dm, out_dir, fmt):
    """Fig 14: Representational dissimilarity matrices per layer."""
    print("\n[6/8] RSA / RDM...")
    layer_list = ["cnn_0", "cnn_5", "trunk_0", "trunk_2"]
    rdm_results = probes.compute_terrain_rdms(dm, layers=layer_list)

    if rdm_results:
        n = len(rdm_results)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
        if n == 1:
            axes = [axes]
        for i, (layer, (rdm, names)) in enumerate(rdm_results.items()):
            viz.plot_rdm(rdm, names, title=f"RDM — {layer}", ax=axes[i])
        fig.tight_layout()
        _save(fig, "fig_rdm_layers", out_dir, fmt)


def generate_patching_figure(model, dm, compiled, out_dir, fmt, device):
    """Fig 15: Activation patching causal analysis."""
    print("\n[7/8] Activation patching...")

    # Get one target-visible and one target-not-visible observation
    vis_steps = dm.get_steps_where(is_target_visible=True)
    novis_steps = dm.get_steps_where(is_target_visible=False)

    if len(vis_steps["traj_ids"]) == 0 or len(novis_steps["traj_ids"]) == 0:
        print("  Skipping patching: not enough contrast data")
        return

    # Load actual observations for patching (need obs_scalars at minimum)
    # Pick one step from each condition
    vis_tid = int(vis_steps["traj_ids"][0])
    vis_sidx = int(vis_steps["step_indices"][0])
    novis_tid = int(novis_steps["traj_ids"][0])
    novis_sidx = int(novis_steps["step_indices"][0])

    vis_traj = dm.get_trajectory(vis_tid)
    novis_traj = dm.get_trajectory(novis_tid)

    # Build observation tensors
    def _build_obs(tdata, step_idx):
        scalars = torch.tensor(tdata["obs_scalars"][step_idx], dtype=torch.float32).unsqueeze(0)
        if "obs_minimaps" in tdata and len(tdata["obs_minimaps"]) > step_idx:
            minimap = torch.tensor(tdata["obs_minimaps"][step_idx], dtype=torch.float32).unsqueeze(0)
        else:
            # Reconstruct minimap from position and map
            pos = tdata["positions"][step_idx]
            map_source = tdata["attrs"]["map_source"]
            map_id = int(tdata["attrs"]["map_id"])
            # Fallback: create a dummy minimap (the patching still works on trunk layer)
            minimap = torch.zeros(1, 3, 45, 45)
        return {"scalars": scalars.to(device), "minimap": minimap.to(device)}

    obs_vis = _build_obs(vis_traj, vis_sidx)
    obs_novis = _build_obs(novis_traj, novis_sidx)

    # Per-neuron patching
    try:
        top_neurons, kl_vals = probes.batch_activation_patching_by_neuron(
            model, obs_novis, obs_vis, patch_layer="trunk.2", top_k=20,
        )
        fig = viz.plot_activation_patching_results(
            top_neurons, kl_vals,
            title="Activation Patching: Target Visibility (trunk.2, per neuron)",
        )
        _save(fig, "fig_activation_patching", out_dir, fmt)
    except Exception as e:
        print(f"  Patching failed: {e}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = TrajectoryDataManager(args.data_dir)
    print(f"Loaded {dm.n_trajectories} trajectories from {args.data_dir}")

    env_config = EnvConfig(device=args.device)
    compiled = env_config.compile_terrain(args.device)

    model = _load_model(args)

    # Generate all figures
    embedding, labels, features_pca = generate_clustering_figures(dm, out_dir, args.format)
    generate_trajectory_gallery(dm, labels, compiled, out_dir, args.format,
                                args.test_maps, args.behavioral_maps)
    generate_value_function_maps(model, dm, compiled, out_dir, args.format,
                                 args.test_maps, args.device)
    generate_activation_analysis(dm, compiled, out_dir, args.format)
    generate_contrast_figures(dm, out_dir, args.format)
    generate_rdm_figures(dm, out_dir, args.format)
    generate_patching_figure(model, dm, compiled, out_dir, args.format, args.device)

    print("\n[8/8] Done! All figures saved to", out_dir)


if __name__ == "__main__":
    main()
