"""Fixed-map pool for fork_wall training on a pre-generated dataset.

Loads a pickled list of ``MapRecord`` (see make_forkwall_dataset.py) and hands
one out per episode. A ``BridgeTunnelEnv`` created with ``map_record=None`` can
be driven off a pool by setting ``env._fixed_record = pool.sample(rng)`` before
each ``reset()`` -- the env then builds terrain/ctg from that fixed record
instead of generating a fresh procedural map.

This is what makes every model (DreamerV3, PPO) train on the SAME fixed dataset
rather than the default per-seed procedural stream.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class MapPool:
    def __init__(self, path: str | Path):
        with open(path, "rb") as f:
            self.records = pickle.load(f)
        if not self.records:
            raise ValueError(f"empty map pool: {path}")
        self.path = str(path)

    def __len__(self):
        return len(self.records)

    def sample(self, rng: np.random.Generator):
        """Uniformly sample one MapRecord."""
        return self.records[int(rng.integers(0, len(self.records)))]

    def get(self, idx: int):
        return self.records[idx % len(self.records)]
