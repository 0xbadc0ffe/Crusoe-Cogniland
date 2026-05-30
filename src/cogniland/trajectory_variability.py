"""Behavioural-variability metrics for a stochastic navigation policy.

Given ``N`` trajectories sampled from the *same* stochastic policy on a
*fixed* map + start (so the only variation is the policy's own
stochasticity plus env slip), this module quantifies how varied that
behaviour is. Two metrics, both framework-agnostic (pure numpy) so they
can be unit-tested and reused by the Dreamer and PPO eval drivers alike:

1. **State-occupancy entropy** — ``occupancy_entropy``.
   Aggregate the visited-cell distribution over all ``N`` rollouts and
   take its Shannon entropy. High ⇒ the policy spreads its probability
   mass over many cells / routes. We also return the *across-trajectory*
   Jensen–Shannon divergence, which isolates the spread that is due to
   trajectories *differing from each other* (0 if every rollout has an
   identical occupancy, regardless of how long each path is).

2. **Number of modes** — ``count_modes``.
   How many distinct *macro-trajectories* (path bundles) the policy's
   path distribution contains. Each trajectory is arc-length-resampled
   to a fixed number of waypoints and the set is clustered by
   single-linkage at a distance threshold; the count of clusters that
   carry a non-trivial share of trajectories is the number of modes.
   E.g. "raft straight across" vs "detour left" vs "detour right" ⇒ 3.

A trajectory is an ``(T, 2)`` array/list of integer ``(row, col)`` cells
*including the start cell*. Variable lengths are fine.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Trajectory = np.ndarray  # (T, 2) int


# ────────────────────────────── helpers ──────────────────────────────────


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy (nats) of a non-negative vector, renormalised."""
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p[p > 0] / s
    return float(-np.sum(p * np.log(p)))


def _as_traj(traj) -> np.ndarray:
    a = np.asarray(traj, dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"each trajectory must be (T, 2); got {a.shape}")
    return a


# ───────────────────────── occupancy entropy ─────────────────────────────


def occupancy_entropy(
    trajectories: Sequence[Trajectory], grid_shape: tuple[int, int]
) -> dict:
    """State-occupancy entropy + across-trajectory JS divergence.

    Returns a dict with the headline ``occupancy_entropy_*`` plus the
    ``across_traj_jsd_*`` decomposition and supporting counts.
    """
    H, W = grid_shape
    ncells = H * W
    trajs = [_as_traj(t) for t in trajectories]
    if not trajs:
        raise ValueError("need at least one trajectory")
    N = len(trajs)

    agg = np.zeros(ncells, dtype=np.float64)
    per_traj_occ = np.zeros((N, ncells), dtype=np.float64)
    for i, t in enumerate(trajs):
        idx = t[:, 0] * W + t[:, 1]
        counts = np.bincount(idx, minlength=ncells).astype(np.float64)
        agg += counts
        per_traj_occ[i] = counts / counts.sum()

    H_agg = _entropy(agg)
    n_distinct = int((agg > 0).sum())

    # Jensen–Shannon divergence across the N per-trajectory occupancies:
    #   JSD = H(mean_i o_i) − mean_i H(o_i)  ≥ 0, =0 iff all o_i equal.
    mean_occ = per_traj_occ.mean(axis=0)
    H_mean = _entropy(mean_occ)
    mean_H = float(np.mean([_entropy(o) for o in per_traj_occ]))
    jsd = max(0.0, H_mean - mean_H)

    return {
        "n_trajectories": N,
        "occupancy_entropy_nats": H_agg,
        "occupancy_entropy_bits": H_agg / math.log(2),
        # normalised to [0, 1] against the whole grid and against the
        # number of cells actually touched (more sensitive comparison)
        "occupancy_entropy_norm": H_agg / math.log(ncells) if ncells > 1 else 0.0,
        "occupancy_entropy_norm_visited": (
            H_agg / math.log(n_distinct) if n_distinct > 1 else 0.0
        ),
        "n_distinct_cells": n_distinct,
        "across_traj_jsd_nats": jsd,
        # JSD over N distributions is bounded by log(N); clamp fp overshoot
        "across_traj_jsd_norm": min(1.0, jsd / math.log(N)) if N > 1 else 0.0,
    }


# ───────────────────────────── modes ─────────────────────────────────────


def _resample_path(traj: np.ndarray, k: int) -> np.ndarray:
    """Arc-length resample a path to ``k`` waypoints, shape ``(k, 2)``."""
    traj = np.asarray(traj, dtype=np.float64)
    if len(traj) == 1:
        return np.repeat(traj, k, axis=0)
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return np.repeat(traj[:1], k, axis=0)
    s = np.linspace(0.0, cum[-1], k)
    r = np.interp(s, cum, traj[:, 0])
    c = np.interp(s, cum, traj[:, 1])
    return np.stack([r, c], axis=1)


def count_modes(
    trajectories: Sequence[Trajectory],
    grid_shape: tuple[int, int],
    n_waypoints: int = 20,
    dist_frac: float = 0.12,
    min_cluster_frac: float = 0.05,
) -> dict:
    """Count the macro-trajectory modes in the path distribution.

    Each path is resampled to ``n_waypoints`` and the set is clustered by
    single-linkage agglomerative clustering with a *mean-waypoint-distance*
    threshold of ``dist_frac * map_diag``. Clusters holding at least
    ``min_cluster_frac`` of the trajectories count as modes.

    Returns ``n_modes`` (the headline), the total cluster count, the
    per-trajectory cluster ``labels`` (1-indexed), and ``cluster_sizes``.
    """
    H, W = grid_shape
    trajs = [_as_traj(t) for t in trajectories]
    N = len(trajs)
    if N == 0:
        raise ValueError("need at least one trajectory")

    feats = np.stack([_resample_path(t, n_waypoints).ravel() for t in trajs])  # (N, 2k)
    diag = math.hypot(H, W)
    # threshold expressed as a Euclidean distance on the flattened feature:
    # euclidean = sqrt(k) * mean_waypoint_distance.
    tau = dist_frac * diag * math.sqrt(n_waypoints)

    if N == 1:
        labels = np.array([1])
    else:
        from scipy.cluster.hierarchy import fcluster, linkage

        Z = linkage(feats, method="single", metric="euclidean")
        labels = fcluster(Z, t=tau, criterion="distance")

    uniq, sizes = np.unique(labels, return_counts=True)
    order = np.argsort(-sizes)
    uniq, sizes = uniq[order], sizes[order]
    min_count = max(1, int(math.ceil(min_cluster_frac * N)))
    n_modes = int((sizes >= min_count).sum())

    return {
        "n_modes": n_modes,
        "n_clusters_total": int(len(uniq)),
        "cluster_sizes": sizes.tolist(),
        "labels": labels.astype(int).tolist(),
        "dist_threshold_cells": dist_frac * diag,
    }


# ───────────────────────── aggregation across maps ───────────────────────


def summarize(per_map: list[dict]) -> dict:
    """Mean ± std of every scalar key across a list of per-map metric dicts."""
    if not per_map:
        return {}
    keys = [k for k, v in per_map[0].items() if isinstance(v, (int, float))]
    out: dict = {"n_maps": len(per_map)}
    for k in keys:
        vals = np.array([float(d[k]) for d in per_map], dtype=np.float64)
        out[f"{k}/mean"] = float(vals.mean())
        out[f"{k}/std"] = float(vals.std())
    return out


# ───────────────────────────── visualisation ─────────────────────────────


def render_overlay(
    terrain: np.ndarray,
    target: tuple[int, int],
    trajectories: Sequence[Trajectory],
    labels: Sequence[int] | None,
    out_path,
    tile_colors: np.ndarray,
    title: str = "",
) -> None:
    """Save a PNG of all trajectories overlaid on the map, coloured by mode.

    ``labels`` (from :func:`count_modes`) colours each path by its cluster so
    the modes are visible; pass ``None`` for a single translucent colour.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(tile_colors[terrain], interpolation="nearest")
    if labels is not None:
        labels = np.asarray(labels)
        uniq = np.unique(labels)
        cmap = plt.get_cmap("tab10" if len(uniq) <= 10 else "tab20")
        color_of = {lab: cmap(i % cmap.N) for i, lab in enumerate(uniq)}
    for i, t in enumerate(trajectories):
        t = _as_traj(t)
        col = color_of[labels[i]] if labels is not None else (1, 1, 1, 1)
        ax.plot(t[:, 1], t[:, 0], "-", color=col, linewidth=0.8, alpha=0.5)
    t0 = _as_traj(trajectories[0])
    ax.plot(t0[0, 1], t0[0, 0], "o", color="cyan", markersize=7, label="start")
    ax.plot(target[1], target[0], "*", color="yellow", markersize=14, label="target")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


__all__ = [
    "occupancy_entropy",
    "count_modes",
    "summarize",
    "render_overlay",
]
