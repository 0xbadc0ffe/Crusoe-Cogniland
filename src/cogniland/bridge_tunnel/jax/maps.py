"""Map dataset loader for the pure-JAX bridge_tunnel env (both variants).

The numpy ``cogniland.bridge_tunnel.mapgen`` stays the authoritative map source
(opensimplex/scipy aren't jittable). Per map we precompute the static cost-to-go
the PyTorch env uses for PBRS — **bt**: a single ``(H,W)`` field
(``BridgeTunnelEnv._compute_ctg``); **btc**: the ``(3,H,W)`` commit-indexed stack
(``_compute_all_ctg``). ``records_to_arrays`` auto-detects the variant from
``rec.category`` (None → bt). The arrays feed ``EnvParams.from_map_arrays``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from cogniland.bridge_tunnel.env import BridgeTunnelEnv
from cogniland.bridge_tunnel.mapgen import (
    CATEGORIES, MapRecord, generate_bridge_tunnel_map, generate_map, is_reachable,
)
from cogniland.bridge_tunnel.tiles import TARGET

_CAT_TO_INT = {c: i for i, c in enumerate(CATEGORIES)}

# canonical task kwargs (bt): 32×64, central door
NATURAL_KWARGS = dict(size=32, width=64, water_frac=0.14, rock_frac=0.14,
                      tree_frac=0.03, goal_half=1)


def _stack_records(records: list[MapRecord]) -> dict:
    n = len(records)
    H, W = records[0].terrain.shape
    commit = any(r.category is not None for r in records)
    terrain = np.zeros((n, H, W), np.int8)
    spawn = np.zeros((n, 2), np.int32); target = np.zeros((n, 2), np.int32)
    goal_mask = np.zeros((n, H, W), bool)
    ctg = np.zeros((n, 3, H, W) if commit else (n, H, W), np.float32)
    category = np.zeros((n,), np.int32)
    for i, rec in enumerate(records):
        terrain[i] = rec.terrain; spawn[i] = rec.spawn; target[i] = rec.target
        goal_mask[i] = (rec.terrain == TARGET)
        if commit:
            ctg[i] = BridgeTunnelEnv._compute_all_ctg(rec.terrain, rec.target)
            category[i] = _CAT_TO_INT.get(rec.category, 0)
        else:
            ctg[i] = BridgeTunnelEnv._compute_ctg(rec.terrain, rec.target).astype(np.float32)
    out = {"terrain": terrain, "spawn": spawn, "target": target,
           "goal_mask": goal_mask, "ctg": ctg}
    if commit:
        out["category"] = category
    return out


def generate_map_dataset(n_maps: int = 4096, seed_start: int = 0, variant: str = "bt",
                         categories: tuple[str, ...] = CATEGORIES, **map_kwargs) -> dict:
    """bt: ``n_maps`` reachable natural maps. btc: class-balanced across categories
    (distinct per-category seed blocks so map seeds are globally unique)."""
    kw = {**NATURAL_KWARGS, **map_kwargs}
    records: list[MapRecord] = []
    if variant == "btc":
        from cogniland.bridge_tunnel.mapgen import generate_commit_map
        per = n_maps // len(categories); rem = n_maps - per * len(categories)
        for j, cat in enumerate(categories):
            base = seed_start + j * 100_000
            count = per + (1 if j < rem else 0)
            for i in range(count):
                records.append(generate_commit_map(seed=base + i, category=cat,
                               size=kw["size"], width=kw["width"],
                               tree_frac=kw["tree_frac"], goal_half=kw["goal_half"]))
    else:
        s = int(seed_start); attempts = 0
        while len(records) < n_maps:
            rec = generate_bridge_tunnel_map(seed=s, **kw); s += 1; attempts += 1
            if is_reachable(rec):
                records.append(rec)
            if attempts > n_maps * 4 + 1000:
                raise RuntimeError("too many unreachable maps; check kwargs")
    return _stack_records(records)


def records_to_arrays(records: list[MapRecord]) -> dict:
    return _stack_records(list(records))


def save_map_arrays(arrays: dict, path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(arrays, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_map_arrays(path) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)
