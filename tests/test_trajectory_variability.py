"""Tests for cogniland.trajectory_variability — the pure metric core."""
from __future__ import annotations

import numpy as np
import pytest

from cogniland.trajectory_variability import (
    count_modes,
    occupancy_entropy,
    summarize,
)

GRID = (20, 20)


def _straight_path(c: int, t: int = 10) -> np.ndarray:
    """Vertical path down column ``c`` for ``t`` rows."""
    return np.stack([np.arange(t), np.full(t, c)], axis=1)


# ───────────────────────── occupancy entropy ─────────────────────────────


def test_identical_trajectories_have_zero_across_traj_jsd():
    trajs = [_straight_path(5) for _ in range(20)]
    m = occupancy_entropy(trajs, GRID)
    # identical occupancies → no across-trajectory variability
    assert m["across_traj_jsd_nats"] == pytest.approx(0.0, abs=1e-9)
    # but the aggregate occupancy still has entropy (the path spans 10 cells)
    assert m["occupancy_entropy_nats"] > 0.0


def test_spread_trajectories_have_higher_entropy_than_concentrated():
    concentrated = [_straight_path(5) for _ in range(20)]
    spread = [_straight_path(c) for c in range(20)]  # one per column
    m_c = occupancy_entropy(concentrated, GRID)
    m_s = occupancy_entropy(spread, GRID)
    assert m_s["occupancy_entropy_nats"] > m_c["occupancy_entropy_nats"]
    assert m_s["across_traj_jsd_nats"] > m_c["across_traj_jsd_nats"]
    assert m_s["n_distinct_cells"] > m_c["n_distinct_cells"]


def test_normalised_entropy_in_unit_range():
    spread = [_straight_path(c) for c in range(20)]
    m = occupancy_entropy(spread, GRID)
    assert 0.0 <= m["occupancy_entropy_norm"] <= 1.0
    assert 0.0 <= m["across_traj_jsd_norm"] <= 1.0


def test_single_cell_trajectories_zero_entropy():
    trajs = [np.array([[3, 3]]) for _ in range(5)]
    m = occupancy_entropy(trajs, GRID)
    assert m["occupancy_entropy_nats"] == pytest.approx(0.0)
    assert m["n_distinct_cells"] == 1


# ───────────────────────────── modes ─────────────────────────────────────


def test_count_modes_single_bundle():
    # 30 near-identical paths (tiny jitter) → one mode
    rng = np.random.default_rng(0)
    trajs = []
    for _ in range(30):
        p = _straight_path(10).astype(float)
        p[:, 1] += rng.normal(0, 0.2, size=len(p))
        trajs.append(np.rint(p).astype(int))
    m = count_modes(trajs, GRID)
    assert m["n_modes"] == 1


def test_count_modes_three_bundles():
    # three well-separated columns, 10 paths each → three modes
    trajs = (
        [_straight_path(2) for _ in range(10)]
        + [_straight_path(10) for _ in range(10)]
        + [_straight_path(18) for _ in range(10)]
    )
    m = count_modes(trajs, GRID, dist_frac=0.10)
    assert m["n_modes"] == 3
    assert sorted(m["cluster_sizes"], reverse=True)[:3] == [10, 10, 10]


def test_count_modes_ignores_tiny_clusters():
    # one big bundle of 30 + a single outlier path → still one *mode*
    trajs = [_straight_path(5) for _ in range(30)] + [_straight_path(18)]
    m = count_modes(trajs, GRID, dist_frac=0.10, min_cluster_frac=0.05)
    assert m["n_modes"] == 1
    assert m["n_clusters_total"] == 2


# ───────────────────────────── summarize ─────────────────────────────────


def test_summarize_mean_std():
    per_map = [
        {"occupancy_entropy_nats": 1.0, "n_modes": 2},
        {"occupancy_entropy_nats": 3.0, "n_modes": 4},
    ]
    s = summarize(per_map)
    assert s["n_maps"] == 2
    assert s["occupancy_entropy_nats/mean"] == pytest.approx(2.0)
    assert s["n_modes/mean"] == pytest.approx(3.0)
    assert s["occupancy_entropy_nats/std"] == pytest.approx(1.0)
