"""Map dataset loader for crafter_in_cogniland.

The numpy-based map generator in ``cogniland.nav.mapgen`` stays the
authoritative source. This module:

1. Calls ``generate_map`` N times, possibly filtered by map_type.
2. Computes per-skill Dijkstra cost-to-go grids (cells, unit-cost).
3. Stacks everything into NumPy arrays + saves to disk.
4. Provides ``load_map_arrays(path)`` to read them back.

The arrays then feed ``EnvParams.from_map_arrays(...)``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from cogniland.nav.mapgen import (
    MapRecord,
    cost_to_go_unit,
    generate_map,
)
from cogniland.nav import skills as sk


MAP_TYPE_TO_ID = {"balanced": 0, "lake": 1, "rocky": 2}


def generate_map_dataset(
    n_maps: int = 256,
    size: int = 64,
    map_types: Iterable[Literal["balanced", "lake", "rocky"]] = ("balanced", "lake", "rocky"),
    seed: int = 0,
) -> dict:
    """Generate ``n_maps`` validated maps spread across ``map_types``.

    Returns a dict of stacked numpy arrays ready for ``EnvParams.from_map_arrays``.
    """
    map_types = tuple(map_types)
    per_type = n_maps // len(map_types)
    rng = np.random.default_rng(seed)

    records: list[MapRecord] = []
    for mt in map_types:
        for k in range(per_type):
            s = int(rng.integers(0, 2**31 - 1))
            try:
                rec = generate_map(size=size, map_type=mt, seed=s, max_retries=200)
            except Exception:
                # try a few more seeds on rare gen failures
                for _ in range(20):
                    s = int(rng.integers(0, 2**31 - 1))
                    try:
                        rec = generate_map(size=size, map_type=mt, seed=s, max_retries=200)
                        break
                    except Exception:
                        continue
                else:
                    raise
            records.append(rec)

    arrays = _stack_records(records, size=size)
    return arrays


def _stack_records(records: list[MapRecord], size: int) -> dict:
    n = len(records)
    terrain = np.zeros((n, size, size), dtype=np.int8)
    spawn = np.zeros((n, 2), dtype=np.int32)
    target = np.zeros((n, 2), dtype=np.int32)
    ctg_none = np.zeros((n, size, size), dtype=np.float32)
    ctg_raft = np.zeros((n, size, size), dtype=np.float32)
    ctg_harness = np.zeros((n, size, size), dtype=np.float32)
    map_type = np.zeros((n,), dtype=np.int8)

    for i, rec in enumerate(records):
        terrain[i] = rec.terrain
        spawn[i] = rec.spawn
        target[i] = rec.target
        tgt = (int(rec.target[0]), int(rec.target[1]))
        ctg_none[i] = cost_to_go_unit(rec.terrain, tgt, sk.NONE).astype(np.float32)
        ctg_raft[i] = cost_to_go_unit(rec.terrain, tgt, sk.RAFT).astype(np.float32)
        ctg_harness[i] = cost_to_go_unit(rec.terrain, tgt, sk.HARNESS).astype(np.float32)
        map_type[i] = MAP_TYPE_TO_ID[rec.map_type]

    return {
        "terrain": terrain, "spawn": spawn, "target": target,
        "ctg_none": ctg_none, "ctg_raft": ctg_raft, "ctg_harness": ctg_harness,
        "map_type": map_type,
    }


def save_map_arrays(arrays: dict, path: str | Path) -> None:
    """Pickle the stacked arrays to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(arrays, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_map_arrays(path: str | Path) -> dict:
    """Load arrays saved by ``save_map_arrays``."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
