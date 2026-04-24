"""Build train/val/test dataset splits and save as ``.pt`` files.

Mirrors ``scripts/generate_dataset.py`` from the legacy repo, but pulls
the terrain generator + visibility LUT precompute from the local
``cogniland_jax.mapgen`` subpackage so cogniland-jax is self-contained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cogniland_jax.mapgen import terrain as gt
from cogniland_jax.mapgen.visibility import compute_visibility_luts

SPLITS_PER_BIOME: dict[str, int] = {"train": 64, "val": 4, "test": 4}
DEFAULT_BASE_SEED = 1


def _build_map(seed: int, biome: str):
    """Return (rgb, heightmap, terrain_idx, berry_mask) for one map.

    Deadly 1-px border is painted into both terrain_idx (-1) and the
    heightmap (value = DEADLY_VALUE).
    """
    raw = gt.generate_raw_heightmap(seed, biome)
    modified = gt.apply_biome_mods(raw, biome, seed)
    cropped = gt.center_crop(modified, gt.CROP_SIZE)
    berry_mask = gt.sample_berry_mask(cropped, biome, gt.BERRY_FRAC, seed)
    terrain_idx = gt._terrain_idx(cropped, biome).astype(np.int8)
    terrain_idx[0, :] = -1
    terrain_idx[-1, :] = -1
    terrain_idx[:, 0] = -1
    terrain_idx[:, -1] = -1
    bordered = gt.paint_deadly_border(cropped).astype(np.float32)
    rgb = gt.colorize_gradient(bordered, biome, berry_mask)
    return rgb, bordered, terrain_idx, berry_mask


def build_split(
    name: str,
    n_per_biome: int,
    base_seed: int,
    num_workers: Optional[int] = None,
) -> tuple[dict, int]:
    """Build one named split ("train"/"val"/"test"), return (dict, next_seed)."""
    rgbs, hms, tidxs, masks, biomes, seeds = [], [], [], [], [], []
    seed = base_seed
    for biome in gt.ALL_BIOMES:
        print(
            f"  {name:<5} {biome:<12} ({n_per_biome} maps) "
            f"seeds [{seed}, {seed + n_per_biome})",
            flush=True,
        )
        for _ in range(n_per_biome):
            rgb, hm, tidx, mask = _build_map(seed, biome)
            rgbs.append(rgb)
            hms.append(hm)
            tidxs.append(tidx)
            masks.append(mask)
            biomes.append(biome)
            seeds.append(seed)
            seed += 1

    heightmaps = np.stack(hms)
    print(
        f"  {name:<5} precomputing visibility LUTs "
        f"({heightmaps.shape[0]} maps × 128×128 cells)",
        flush=True,
    )
    vis_lut = compute_visibility_luts(heightmaps, num_workers=num_workers)

    return {
        "rgb": torch.from_numpy(np.stack(rgbs)),
        "heightmap": torch.from_numpy(heightmaps),
        "terrain_idx": torch.from_numpy(np.stack(tidxs)),
        "berry_mask": torch.from_numpy(np.stack(masks)),
        "visibility_lut": torch.from_numpy(vis_lut),
        "biomes": biomes,
        "seeds": seeds,
    }, seed


def build_dataset(
    output_dir: str | Path = "data/maps",
    base_seed: int = DEFAULT_BASE_SEED,
    splits_per_biome: Optional[dict[str, int]] = None,
    num_workers: Optional[int] = None,
    preview: bool = False,
) -> dict[str, dict]:
    """Build all splits and save to ``output_dir``. Returns the dict of splits."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = dict(splits_per_biome or SPLITS_PER_BIOME)

    print(f"Generating map dataset (base seed {base_seed})")
    print(
        "  per biome: "
        + " ".join(f"{k}={v}" for k, v in splits.items())
    )

    seed = base_seed
    saved: dict[str, dict] = {}
    for name, n in splits.items():
        split, seed = build_split(name, n, seed, num_workers)
        path = out_dir / f"{name}.pt"
        torch.save(split, path)
        mb = path.stat().st_size / 1e6
        print(f"  saved {path.name}: {split['rgb'].shape[0]} maps  ({mb:.1f} MB)")
        saved[name] = split

    if preview and "val" in saved:
        _save_preview(saved["val"], out_dir / "preview_val.png")
    return saved


def _save_preview(split: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = split["rgb"].numpy()
    biomes = split["biomes"]
    seeds = split["seeds"]
    per_biome: dict[str, list[int]] = {b: [] for b in gt.ALL_BIOMES}
    for i, b in enumerate(biomes):
        per_biome[b].append(i)
    ncols = len(gt.ALL_BIOMES)
    nrows = max(len(v) for v in per_biome.values())
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    for col, biome in enumerate(gt.ALL_BIOMES):
        for row in range(nrows):
            ax = axes[row, col]
            idxs = per_biome[biome]
            if row < len(idxs):
                i = idxs[row]
                ax.imshow(rgb[i], interpolation="nearest")
                title = f"seed={seeds[i]}"
                if row == 0:
                    title = f"{biome.upper()}\n{title}"
                ax.set_title(title, fontsize=9)
            ax.set_axis_off()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  preview: {path}")


__all__ = ["build_dataset", "build_split", "SPLITS_PER_BIOME"]
