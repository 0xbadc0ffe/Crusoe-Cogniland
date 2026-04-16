"""Generate the RGB-based strategy dataset (train / val / test).

Each map is a 128x128x3 uint8 RGB image (the observation the CNN will see),
accompanied by an int8 terrain index grid and a boolean berry mask. The
deadly 1-pixel border is marked as terrain_idx == -1.

During play, two extra channels will be stacked on top of the RGB image
(current berry mask + target indicator). Spawn/target positions are NOT
baked into the dataset — they are sampled at play time.

Layout:
    data/strategy/
        strategy_train.pt    64 maps × 4 biomes = 256
        strategy_val.pt       4 maps × 4 biomes =  16
        strategy_test.pt      4 maps × 4 biomes =  16
        preview_val.png       small grid visualization

Each .pt file is a dict:
    {
        "rgb":         uint8   [N, 128, 128, 3],
        "heightmap":   float32 [N, 128, 128],    # post-border, in [-1, 1]
        "terrain_idx": int8    [N, 128, 128],    # -1 = deadly border
        "berry_mask":  bool    [N, 128, 128],
        "biomes":      list[str]   (length N),
        "seeds":       list[int]   (length N),
    }

The heightmap is kept so the runtime env can reuse height-based line-of-sight
occlusion (mountains/forest shadow rays) rather than only class-based blocking.

Usage:
    python scripts/generate_strategy_dataset.py
    python scripts/generate_strategy_dataset.py --preview --base-seed 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import generate_strategy_maps as gt


SPLITS_PER_BIOME = {"train": 64, "val": 4, "test": 4}
DEFAULT_BASE_SEED = 1


def _build_map(seed: int, biome: str):
    """Return (rgb, heightmap, terrain_idx, berry_mask) for one generated map."""
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


def _build_split(name: str, n_per_biome: int, base_seed: int):
    rgbs, hms, tidxs, masks, biomes, seeds = [], [], [], [], [], []
    seed = base_seed
    for biome in gt.ALL_BIOMES:
        print(f"  {name:<5} {biome:<12} ({n_per_biome} maps) seeds "
              f"[{seed}, {seed + n_per_biome})", flush=True)
        for _ in range(n_per_biome):
            rgb, hm, tidx, mask = _build_map(seed, biome)
            rgbs.append(rgb)
            hms.append(hm)
            tidxs.append(tidx)
            masks.append(mask)
            biomes.append(biome)
            seeds.append(seed)
            seed += 1
    return {
        "rgb": torch.from_numpy(np.stack(rgbs)),
        "heightmap": torch.from_numpy(np.stack(hms)),
        "terrain_idx": torch.from_numpy(np.stack(tidxs)),
        "berry_mask": torch.from_numpy(np.stack(masks)),
        "biomes": biomes,
        "seeds": seeds,
    }, seed


def _save_preview(split: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = split["rgb"].numpy()
    biomes = split["biomes"]
    seeds = split["seeds"]
    n = rgb.shape[0]
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
    fig.suptitle(f"{path.stem} ({n} maps)", fontsize=12, y=1.01)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  preview: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--output-dir", type=str, default="data/strategy")
    parser.add_argument("--preview", action="store_true",
                        help="Save a val-set preview PNG")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating strategy dataset (base seed {args.base_seed})")
    print(f"  per biome: train={SPLITS_PER_BIOME['train']} "
          f"val={SPLITS_PER_BIOME['val']} test={SPLITS_PER_BIOME['test']}")

    seed = args.base_seed
    saved = {}
    for name, n in SPLITS_PER_BIOME.items():
        split, seed = _build_split(name, n, seed)
        path = out_dir / f"strategy_{name}.pt"
        torch.save(split, path)
        mb = path.stat().st_size / 1e6
        print(f"  saved {path.name}: {split['rgb'].shape[0]} maps  ({mb:.1f} MB)")
        saved[name] = split

    if args.preview:
        _save_preview(saved["val"], out_dir / "preview_val.png")


if __name__ == "__main__":
    main()
