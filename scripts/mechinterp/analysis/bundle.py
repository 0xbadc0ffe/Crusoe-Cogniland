"""Load a self-contained activation-dataset bundle and expose it for analysis.

Works on both the BTC bundle (map categories + skill commitment) and the BT
bundle (neither). The schema is *detected from the data*, not hard-coded:

    b = ActivationBundle("activation_datasets/btc_ppo")
    b.sources              # ['gru_h', 'enc_embed']   (auto-discovered in the h5)
    b.has_belief           # True if a `category` label column exists
    b.has_skill            # True if a `commit_state`/`final_commit` column exists
    X = b.load_activations("gru_h", row_ids)   # (len(row_ids), D) float32
    rgb = b.render_obs(row_id)                  # (H, W, 3) uint8 egocentric frame

Only numpy / pandas / h5py are required to load; matplotlib is optional (palette
rendering uses numpy directly).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import style

# h5 datasets that are NOT learned activation sources
_NON_SOURCE = {"row_id", "action_probs", "minimap", "scalars"}
_BELIEF_COL = "category"
_SKILL_COLS = ("final_commit", "commit_state")


@dataclass
class ActivationBundle:
    path: Path
    manifest: dict = field(init=False)
    labels: pd.DataFrame = field(init=False)
    maps: dict = field(init=False)
    palette: np.ndarray = field(init=False)

    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.labels = pd.read_parquet(self.path / "labels.parquet")
        npz = np.load(self.path / "maps.npz", allow_pickle=True)
        self.maps = {k: npz[k] for k in npz.files}
        self.palette = np.asarray(self.manifest["tile_colors"], dtype=np.uint8)
        # add the ordinal belief target if categories are present
        if self.has_belief:
            self.labels["belief_ordinal_true"] = (
                self.labels[_BELIEF_COL].map(style.CATEGORY_ORDINAL).astype("float32")
            )

    # ------------------------------------------------------------------ schema
    @property
    def name(self) -> str:
        return self.path.name

    @property
    def is_commit(self) -> bool:
        return bool(self.manifest.get("is_commit", False))

    @property
    def has_belief(self) -> bool:
        return _BELIEF_COL in self.labels.columns

    @property
    def has_skill(self) -> bool:
        return any(c in self.labels.columns for c in _SKILL_COLS)

    @property
    def view_size(self) -> int:
        return int(self.manifest["view_size"])

    @property
    def action_names(self) -> list:
        return list(self.manifest["action_names"])

    @property
    def sources(self) -> list:
        """Learned activation sources present in the h5 (2-D float datasets)."""
        with h5py.File(self.path / "activations.h5", "r") as f:
            out = []
            for k in f.keys():
                if k in _NON_SOURCE:
                    continue
                d = f[k]
                if d.ndim == 2 and np.issubdtype(d.dtype, np.floating):
                    out.append(k)
        # stable, friendly order
        pref = [s for s in ("gru_h", "enc_embed") if s in out]
        return pref + [s for s in out if s not in pref]

    def source_dim(self, source: str) -> int:
        return int(self.manifest.get("activation_sites", {}).get(source, 0)) or self._h5_dim(source)

    def _h5_dim(self, key: str) -> int:
        with h5py.File(self.path / "activations.h5", "r") as f:
            return int(f[key].shape[1])

    # -------------------------------------------------------------- activations
    def load_activations(self, source: str, row_ids=None) -> np.ndarray:
        """Return (n, D) float32. `row_ids` are values of the `row_id` column,
        which equal the h5 row index (the bundle guarantees this alignment)."""
        with h5py.File(self.path / "activations.h5", "r") as f:
            dset = f[source]
            if row_ids is None:
                return dset[:].astype(np.float32)
            idx = np.asarray(row_ids, dtype=np.int64)
            order = np.argsort(idx, kind="stable")
            arr = dset[idx[order]].astype(np.float32)      # h5 needs increasing idx
            out = np.empty_like(arr)
            out[order] = arr
            return out

    def load_extra(self, key: str, row_ids=None) -> np.ndarray:
        """Load a non-source h5 dataset (e.g. 'action_probs', 'minimap')."""
        return self.load_activations(key, row_ids)

    # ------------------------------------------------------------------ render
    def render_obs(self, row_id: int, upscale: int = 6) -> np.ndarray:
        """Egocentric minimap of a row as an upscaled (H*u, W*u, 3) uint8 image."""
        with h5py.File(self.path / "activations.h5", "r") as f:
            mm = np.asarray(f["minimap"][int(row_id)])
        rgb = self.palette[mm]
        return np.kron(rgb, np.ones((upscale, upscale, 1), dtype=np.uint8))

    def render_map(self, map_id: int) -> np.ndarray:
        """Full terrain of a map as an RGB uint8 image."""
        return self.palette[self.maps["terrain"][int(map_id)]]

    # ------------------------------------------------------------------ summary
    def summary(self) -> str:
        s = [f"bundle '{self.name}'  env={self.manifest.get('env')}  rows={len(self.labels):,}"]
        s.append(f"  sources: {', '.join(f'{x}({self.source_dim(x)})' for x in self.sources)}")
        s.append(f"  belief={self.has_belief}  skill={self.has_skill}  view={self.view_size}")
        if self.has_belief:
            s.append("  category: " + ", ".join(
                f"{k}={v}" for k, v in self.labels[_BELIEF_COL].value_counts().items()))
        if self.has_skill:
            col = next(c for c in _SKILL_COLS if c in self.labels.columns)
            s.append(f"  {col}: " + ", ".join(
                f"{k}={v}" for k, v in self.labels[col].value_counts().items()))
        return "\n".join(s)
