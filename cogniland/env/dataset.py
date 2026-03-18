"""MapDataset — pre-generated map pool with guaranteed train/val/test splits.

The dataset is a single .pt file containing three disjoint sets of island maps.
Seed assignment ensures no overlap: train=[base, base+n_train),
val=[base+n_train, base+n_train+n_val), test=[base+n_train+n_val, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class MapDataset:
    """Three disjoint sets of pre-generated island maps stored as CPU float32 tensors."""

    train_maps: Tensor   # [n_train, H, W] float32 CPU
    val_maps:   Tensor   # [n_val,   H, W] float32 CPU
    test_maps:  Tensor   # [n_test,  H, W] float32 CPU
    seed: int
    map_size: int

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def n_train(self) -> int:
        return self.train_maps.shape[0]

    @property
    def n_val(self) -> int:
        return self.val_maps.shape[0]

    @property
    def n_test(self) -> int:
        return self.test_maps.shape[0]

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_maps": self.train_maps,
                "val_maps":   self.val_maps,
                "test_maps":  self.test_maps,
                "seed":       self.seed,
                "map_size":   self.map_size,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MapDataset":
        data = torch.load(str(path), map_location="cpu", weights_only=True)
        return cls(
            train_maps=data["train_maps"],
            val_maps=data["val_maps"],
            test_maps=data["test_maps"],
            seed=data["seed"],
            map_size=data["map_size"],
        )
