"""MapDataset utilities for train/val/test island map splits."""

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

    @staticmethod
    def _load_split_file(path: str | Path) -> tuple[Tensor, int, int]:
        data = torch.load(str(path), map_location="cpu", weights_only=True)
        return data["maps"], int(data["seed"]), int(data["map_size"])

    @classmethod
    def from_split_files(
        cls,
        train_path: str | Path,
        val_path: str | Path,
        test_path: str | Path,
    ) -> "MapDataset":
        train_maps, train_seed, train_size = cls._load_split_file(train_path)
        val_maps, val_seed, val_size = cls._load_split_file(val_path)
        test_maps, test_seed, test_size = cls._load_split_file(test_path)

        if len({train_size, val_size, test_size}) != 1:
            raise ValueError(
                "Train/val/test dataset splits must have the same map_size: "
                f"train={train_size}, val={val_size}, test={test_size}"
            )

        return cls(
            train_maps=train_maps,
            val_maps=val_maps,
            test_maps=test_maps,
            seed=train_seed,
            map_size=train_size,
        )

    @classmethod
    def load_from_config(cls, dataset_cfg) -> "MapDataset | None":
        train_path = dataset_cfg.get("train_path", "")
        val_path = dataset_cfg.get("val_path", "")
        test_path = dataset_cfg.get("test_path", "")

        if not (train_path or val_path or test_path):
            return None

        if not (train_path and val_path and test_path):
            raise ValueError(
                "Dataset config must provide all of train_path, val_path, and test_path."
            )

        return cls.from_split_files(train_path, val_path, test_path)
