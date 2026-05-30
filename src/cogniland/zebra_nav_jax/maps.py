"""Map dataset loader for the pure-JAX zebra_nav env.

The numpy ``cogniland.zebra_nav.mapgen.generate_zebra_map`` stays the
authoritative source (natural maps use opensimplex + scipy, which are NOT
jittable). This module:

1. Generates N natural maps deterministically by seed.
2. Precomputes, per map, the **static-terrain** min-action cost-to-go field
   (the SAME ``ZebraNavEnv._compute_ctg`` Dijkstra the PyTorch env uses for its
   PBRS potential) and the goal-cell mask.
3. Stacks everything into numpy arrays + pickles them.
4. Provides ``load_map_arrays(path)`` to read them back.

The arrays feed ``EnvParams.from_map_arrays(...)``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from cogniland.zebra_nav.env import ZebraNavEnv
from cogniland.zebra_nav.mapgen import MapRecord, generate_zebra_map, is_reachable
from cogniland.zebra_nav.tiles import TARGET


# Canonical natural task. The goal is the ENTIRE right wall (goal_half=None);
# this is the default environment. (A positive goal_half would instead carve a
# central door — the earlier central-door variant.)
NATURAL_KWARGS = dict(
    size=32,
    width=64,
    orientation="natural",
    water_frac=0.14,
    rock_frac=0.14,
    tree_frac=0.03,
    goal_half=None,
)


def _stack_records(records: list[MapRecord]) -> dict:
    n = len(records)
    H, W = records[0].terrain.shape
    terrain = np.zeros((n, H, W), dtype=np.int8)
    spawn = np.zeros((n, 2), dtype=np.int32)
    target = np.zeros((n, 2), dtype=np.int32)
    goal_mask = np.zeros((n, H, W), dtype=bool)
    ctg = np.zeros((n, H, W), dtype=np.float32)

    for i, rec in enumerate(records):
        assert rec.terrain.shape == (H, W), "all maps must share a shape"
        terrain[i] = rec.terrain
        spawn[i] = rec.spawn
        target[i] = rec.target
        goal_mask[i] = (rec.terrain == TARGET)
        # Use the authoritative PyTorch Dijkstra so the PBRS potential is
        # identical by construction.
        ctg[i] = ZebraNavEnv._compute_ctg(rec.terrain, rec.target).astype(np.float32)

    return {
        "terrain": terrain, "spawn": spawn, "target": target,
        "goal_mask": goal_mask, "ctg": ctg,
    }


def generate_map_dataset(
    n_maps: int = 4096,
    seed_start: int = 0,
    **map_kwargs,
) -> dict:
    """Generate ``n_maps`` reachable natural maps and stack them.

    ``map_kwargs`` overrides ``NATURAL_KWARGS`` (the natural_agent.yaml task).
    Training seeds start at ``seed_start`` (default 0) — kept distinct from the
    held-out validation seeds (10_000+) used by ``make_zebra_val_maps.py``.
    """
    kw = {**NATURAL_KWARGS, **map_kwargs}
    records: list[MapRecord] = []
    s = int(seed_start)
    attempts = 0
    while len(records) < n_maps:
        rec = generate_zebra_map(seed=s, **kw)
        s += 1
        attempts += 1
        if is_reachable(rec):
            records.append(rec)
        if attempts > n_maps * 4 + 1000:
            raise RuntimeError("too many unreachable maps; check kwargs")
    return _stack_records(records)


def records_to_arrays(records: list[MapRecord]) -> dict:
    """Stack an explicit list of ``MapRecord`` (e.g. the pickled val set)."""
    return _stack_records(list(records))


def save_map_arrays(arrays: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(arrays, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_map_arrays(path: str | Path) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)
