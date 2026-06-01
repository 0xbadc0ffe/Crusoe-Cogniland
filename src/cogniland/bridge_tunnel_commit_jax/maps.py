"""Map dataset loader for the pure-JAX bridge_tunnel_commit env.

The numpy ``cogniland.bridge_tunnel_commit.mapgen`` stays the authoritative map
source (opensimplex + scipy are not jittable). This module:

1. Generates a **class-balanced** dataset of natural maps across the three
   categories (balanced / lakes / rocky), each winnable under its intended
   commitment.
2. Precomputes, per map, the three commitment-indexed static cost-to-go fields
   (the SAME ``BridgeTunnelCommitEnv._compute_all_ctg`` the PyTorch env uses for
   its PBRS potential) + the goal-cell mask + the integer category label.
3. Stacks everything into numpy arrays + pickles them.

The arrays feed ``EnvParams.from_map_arrays(...)``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from cogniland.bridge_tunnel_commit.env import BridgeTunnelCommitEnv
from cogniland.bridge_tunnel_commit.mapgen import (
    CATEGORIES, MapRecord, generate_commit_map,
)
from cogniland.bridge_tunnel_commit.tiles import TARGET

_CAT_TO_INT = {c: i for i, c in enumerate(CATEGORIES)}   # balanced=0, lakes=1, rocky=2

# Canonical commit task: 32×64, 3-cell central door (goal_half=1), edge-biased trees.
NATURAL_KWARGS = dict(size=32, width=64, tree_frac=0.03, goal_half=1)


def _stack_records(records: list[MapRecord]) -> dict:
    n = len(records)
    H, W = records[0].terrain.shape
    terrain = np.zeros((n, H, W), dtype=np.int8)
    spawn = np.zeros((n, 2), dtype=np.int32)
    target = np.zeros((n, 2), dtype=np.int32)
    goal_mask = np.zeros((n, H, W), dtype=bool)
    ctg = np.zeros((n, 3, H, W), dtype=np.float32)
    category = np.zeros((n,), dtype=np.int32)

    for i, rec in enumerate(records):
        assert rec.terrain.shape == (H, W), "all maps must share a shape"
        terrain[i] = rec.terrain
        spawn[i] = rec.spawn
        target[i] = rec.target
        goal_mask[i] = (rec.terrain == TARGET)
        # authoritative commit-aware Dijkstra (none/build/mine) → identical PBRS
        ctg[i] = BridgeTunnelCommitEnv._compute_all_ctg(rec.terrain, rec.target)
        category[i] = _CAT_TO_INT[rec.category]

    return {
        "terrain": terrain, "spawn": spawn, "target": target,
        "goal_mask": goal_mask, "ctg": ctg, "category": category,
    }


def generate_map_dataset(
    n_maps: int = 4096,
    seed_start: int = 0,
    categories: tuple[str, ...] = CATEGORIES,
    **map_kwargs,
) -> dict:
    """Generate a class-balanced dataset of ``n_maps`` reachable natural maps.

    ``n_maps`` is split as evenly as possible across ``categories`` (each
    category draws its own deterministic seed block starting at ``seed_start``).
    Training seeds start at ``seed_start`` (default 0) — kept distinct from the
    held-out validation seeds (10_000+).
    """
    kw = {**NATURAL_KWARGS, **map_kwargs}
    per = n_maps // len(categories)
    rem = n_maps - per * len(categories)
    records: list[MapRecord] = []
    for j, cat in enumerate(categories):
        count = per + (1 if j < rem else 0)
        for i in range(count):
            records.append(generate_commit_map(seed=seed_start + i, category=cat, **kw))
    return _stack_records(records)


def records_to_arrays(records: list[MapRecord]) -> dict:
    """Stack an explicit list of ``MapRecord`` (e.g. a pickled val set)."""
    return _stack_records(list(records))


def save_map_arrays(arrays: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(arrays, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_map_arrays(path: str | Path) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)
