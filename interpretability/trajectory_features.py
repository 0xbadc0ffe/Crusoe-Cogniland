"""Trajectory featurization for clustering and dimensionality reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def _direction_changes(positions: np.ndarray) -> int:
    """Count the number of direction changes in a trajectory."""
    if len(positions) < 3:
        return 0
    diffs = np.diff(positions, axis=0)  # [T-1, 2]
    changes = 0
    for i in range(1, len(diffs)):
        if not np.array_equal(diffs[i], diffs[i - 1]):
            changes += 1
    return changes


def _tortuosity(positions: np.ndarray) -> float:
    """Path length / euclidean distance between start and end."""
    if len(positions) < 2:
        return 1.0
    path_len = float(len(positions) - 1)  # Manhattan steps
    euclid = np.linalg.norm(positions[-1].astype(float) - positions[0].astype(float))
    if euclid < 1e-6:
        return path_len  # Stayed in place
    return path_len / euclid


def _convex_hull_area(positions: np.ndarray) -> float:
    """Area of the convex hull of visited cells."""
    unique = np.unique(positions, axis=0)
    if len(unique) < 3:
        return 0.0
    try:
        hull = ConvexHull(unique.astype(float))
        return hull.volume  # 2D → volume is area
    except Exception:
        return 0.0


def _mean_curvature(positions: np.ndarray) -> float:
    """Mean angular change along the trajectory."""
    if len(positions) < 3:
        return 0.0
    diffs = np.diff(positions.astype(float), axis=0)
    angles = np.arctan2(diffs[:, 0], diffs[:, 1])
    d_angles = np.abs(np.diff(angles))
    # Wrap to [0, pi]
    d_angles = np.minimum(d_angles, 2 * np.pi - d_angles)
    return float(d_angles.mean()) if len(d_angles) > 0 else 0.0


def featurize_trajectory(
    summary_row: pd.Series,
    positions: np.ndarray,
    max_episode_length: int = 1000,
) -> np.ndarray:
    """Convert a single trajectory into a fixed-length feature vector.

    Returns:
        1D numpy array of features.
    """
    features = []

    # Behavioral metrics (8)
    features.append(summary_row.get("directness_ratio", 0.0))
    features.append(summary_row.get("risk_score", 0.0))
    features.append(summary_row.get("ocean_usage_ratio", 0.0))
    features.append(summary_row.get("forest_usage_ratio", 0.0))
    features.append(summary_row.get("episode_length", 0) / max_episode_length)
    features.append(summary_row.get("average_hp", 50.0) / 100.0)
    features.append(summary_row.get("average_resources", 50.0) / 100.0)
    features.append(summary_row.get("map_coverage", 0.0))

    # Shape descriptors (4)
    features.append(_direction_changes(positions) / max(len(positions), 1))
    features.append(min(_tortuosity(positions), 20.0) / 20.0)  # Normalize
    features.append(_convex_hull_area(positions) / (250 * 250))  # Normalize by map area
    features.append(_mean_curvature(positions) / np.pi)  # Normalize to [0, 1]

    # Terrain distribution (9)
    for i in range(9):
        features.append(summary_row.get(f"terrain_frac_{i}", 0.0))

    return np.array(features, dtype=np.float32)


def featurize_all(
    summary: pd.DataFrame,
    h5_path: str,
    max_episode_length: int = 1000,
) -> tuple[np.ndarray, list[str]]:
    """Featurize all trajectories.

    Args:
        summary: summary DataFrame from TrajectoryCollector.
        h5_path: path to trajectories.h5.
        max_episode_length: for normalizing episode length.

    Returns:
        (features [N, D], feature_names [D])
    """
    import h5py

    feature_names = [
        "directness_ratio", "risk_score", "ocean_usage_ratio", "forest_usage_ratio",
        "episode_length_norm", "average_hp_norm", "average_resources_norm", "map_coverage",
        "direction_changes_norm", "tortuosity_norm", "hull_area_norm", "mean_curvature_norm",
        "terrain_ocean", "terrain_deep_water", "terrain_water", "terrain_beach",
        "terrain_sandy", "terrain_grassland", "terrain_forest", "terrain_rocky",
        "terrain_mountains",
    ]

    all_features = []

    with h5py.File(h5_path, "r") as f:
        for _, row in summary.iterrows():
            tid = int(row["traj_id"])
            gname = f"trajectory_{tid:04d}"
            if gname in f:
                positions = np.array(f[gname]["positions"])
            else:
                positions = np.array([[0, 0]])
            feat = featurize_trajectory(row, positions, max_episode_length)
            all_features.append(feat)

    return np.stack(all_features), feature_names
