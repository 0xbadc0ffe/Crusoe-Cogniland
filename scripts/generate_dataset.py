"""Generate pre-built island maps with disjoint train/val/test splits.

Each split is saved as its own .pt file containing a single float32 tensor
of shape [N, H, W] (heightmaps).

After generation, a forest-disconnection pass is applied: Gaussian-smoothed
blob noise is overlaid and forest tiles where noise < blob_threshold are
converted to grassland, fragmenting contiguous forest into patches.

Usage:
    python scripts/generate_dataset.py [options]

    # Quick smoke-test (small dataset):
    python scripts/generate_dataset.py --seed 42 --train 8 --val 4 --test 4

    # Full-size dataset with custom forest params:
    python scripts/generate_dataset.py --seed 42 --train 128 --val 16 --test 16 \
        --forest-threshold 0.45 --blob-sigma 8 --blob-threshold 0.65

Seed assignment (disjoint by construction):
    train maps: seeds [base, base+n_train)
    val   maps: seeds [base+n_train, base+n_train+n_val)
    test  maps: seeds [base+n_train+n_val, ...)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cogniland.env.islands import colorize, generate_island
from cogniland.env.types import CurriculumStage, EnvConfig, MapGenConfig


# ── Forest disconnection ─────────────────────────────────────────────────────

def _blob_noise(size: int, sigma: float, seed: int = 0) -> np.ndarray:
    """Gaussian-smoothed uniform noise normalised to [0, 1]."""
    rng = np.random.RandomState(seed)
    raw = rng.uniform(0.0, 1.0, (size, size))
    blurred = gaussian_filter(raw, sigma=sigma)
    lo, hi = blurred.min(), blurred.max()
    return (blurred - lo) / (hi - lo + 1e-9)


def _disconnect_forest(
    hm: torch.Tensor,
    thresholds: np.ndarray,
    forest_idx: int,
    grassland_idx: int,
    forest_threshold: float,
    blob_sigma: float,
    blob_threshold: float,
    seed: int = 0,
) -> torch.Tensor:
    """Return a modified heightmap where parts of the forest are erased.

    Forest tiles whose heightmap value < forest_threshold and where
    blob noise < blob_threshold are set to (grassland_threshold - epsilon),
    making them classify as grassland after terrain quantisation.
    """
    h = hm.numpy().copy()
    size = h.shape[0]

    idx = np.full(h.shape, len(thresholds) - 1, dtype=np.int8)
    for k in range(len(thresholds) - 1, -1, -1):
        idx[h < thresholds[k]] = k

    target = (idx == forest_idx) & (h < forest_threshold)
    noise = _blob_noise(size, blob_sigma, seed)
    erase = target & (noise < blob_threshold)

    grassland_upper = float(thresholds[grassland_idx])
    h[erase] = grassland_upper - 1e-4

    return torch.from_numpy(h)


# ── Map generation ────────────────────────────────────────────────────────────

def _generate_maps(base_seed: int, count: int, config: EnvConfig) -> torch.Tensor:
    """Generate `count` maps with consecutive seeds starting at base_seed."""
    maps = []
    for i in range(count):
        seed_i = base_seed + i
        torch.manual_seed(seed_i)
        random.seed(seed_i)
        np.random.seed(seed_i)
        maps.append(generate_island(config))
        if (i + 1) % 10 == 0 or (i + 1) == count:
            print(f"  [{i+1}/{count}]", flush=True)
    return torch.stack(maps)


def _apply_forest_disconnection(
    maps: torch.Tensor,
    compiled,
    forest_threshold: float,
    blob_sigma: float,
    blob_threshold: float,
    base_seed: int,
) -> torch.Tensor:
    """Apply forest disconnection to a batch of maps."""
    thresholds = compiled.thresholds.numpy()
    names = compiled.terrain_names
    forest_idx = names.index("forest")
    grassland_idx = names.index("grassland")

    out = []
    for i in range(maps.shape[0]):
        out.append(_disconnect_forest(
            maps[i], thresholds, forest_idx, grassland_idx,
            forest_threshold, blob_sigma, blob_threshold,
            seed=base_seed + i,
        ))
    return torch.stack(out)


# ── Preview ───────────────────────────────────────────────────────────────────

def _sample_land_position(
    world_map: torch.Tensor,
    land_threshold: float,
    rng: random.Random,
    stage: CurriculumStage = CurriculumStage.NORMAL,
    center: int = 125,
    radius: int = 50,
) -> tuple[int, int]:
    """Sample a single land position for visualization."""
    size = world_map.shape[0]
    wm = world_map.numpy()
    while True:
        if stage == CurriculumStage.EASY:
            r = rng.randint(max(0, center - radius), min(size - 1, center + radius))
            c = rng.randint(max(0, center - radius), min(size - 1, center + radius))
            if (r - center) ** 2 + (c - center) ** 2 > radius * radius:
                continue
        else:
            r = rng.randint(0, size - 1)
            c = rng.randint(0, size - 1)
        if wm[r, c] > land_threshold:
            return r, c


def _make_preview(
    train_maps: torch.Tensor,
    config: EnvConfig,
    stage: CurriculumStage,
    output_path: Path,
    n_maps: int = 3,
    easy_radius: int = 50,
) -> None:
    """Save a 2×n_maps grid of maps with spawn (green) and target (red) overlaid."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    compiled = config.compile_terrain("cpu")
    land_threshold = compiled.land_threshold
    center = config.size // 2
    radius = easy_radius
    rng = random.Random(0)

    fig, axes = plt.subplots(2, n_maps, figsize=(4 * n_maps, 9))
    fig.suptitle(f"Spawn/Target placement — {stage.value.upper()} stage", fontsize=14)

    for col in range(n_maps):
        wm = train_maps[col]
        rgb = colorize(wm, compiled).numpy().astype("uint8")

        for row in range(2):
            ax = axes[row, col]
            ax.imshow(rgb)
            ax.axis("off")
            ax.set_title(f"Map {col}", fontsize=9)

            spawn = _sample_land_position(wm, land_threshold, rng, stage, center, radius)
            target = _sample_land_position(wm, land_threshold, rng, stage, center, radius)

            ax.scatter([spawn[1]], [spawn[0]], c="lime", s=60, zorder=5, marker="o",
                       edgecolors="black", linewidths=0.5)
            ax.scatter([target[1]], [target[0]], c="red", s=60, zorder=5, marker="*",
                       edgecolors="black", linewidths=0.5)

            if stage == CurriculumStage.EASY:
                circle = plt.Circle(
                    (center, center), radius,
                    fill=False, edgecolor="yellow", linewidth=1.5, linestyle="--",
                )
                ax.add_patch(circle)

    spawn_patch = mpatches.Patch(color="lime", label="spawn")
    target_patch = mpatches.Patch(color="red", label="target")
    fig.legend(handles=[spawn_patch, target_patch], loc="lower center", ncol=2, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Preview saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _save_split(maps: torch.Tensor, seed: int, map_size: int, path: Path) -> None:
    """Save a single split as a .pt dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"maps": maps, "seed": seed, "map_size": map_size}, path)
    size_mb = path.stat().st_size / 1e6
    print(f"  {path.name}: {maps.shape[0]} maps  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cogniland map splits")
    parser.add_argument("--seed",   type=int, default=42,    help="Base RNG seed")
    parser.add_argument("--train",  type=int, default=128,   help="Number of training maps")
    parser.add_argument("--val",    type=int, default=16,    help="Number of validation maps")
    parser.add_argument("--test",   type=int, default=16,    help="Number of test maps")
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory for the 3 split files (default: data/)",
    )
    parser.add_argument("--no-preview", action="store_true", help="Skip PNG preview generation")

    fg = parser.add_argument_group("forest disconnection")
    fg.add_argument("--forest-threshold", type=float, default=0.45,
                    help="Heightmap cutoff for forest band (default: 0.45)")
    fg.add_argument("--blob-sigma",       type=float, default=8.0,
                    help="Gaussian blur radius for blob noise (default: 8)")
    fg.add_argument("--blob-threshold",   type=float, default=0.65,
                    help="Fraction of forest to erase (default: 0.65)")
    fg.add_argument("--no-forest-disconnect", action="store_true",
                    help="Skip forest disconnection (keep original terrain)")

    args = parser.parse_args()

    n_train, n_val, n_test = args.train, args.val, args.test
    base_seed = args.seed
    out_dir = Path(args.output_dir)
    tag = f"seed{base_seed}"

    config = EnvConfig(map_generation=MapGenConfig(seed=base_seed))
    compiled = config.compile_terrain("cpu")

    total = n_train + n_val + n_test
    print(f"Generating {total} maps ({config.size}x{config.size}) with base seed {base_seed}")
    print(f"  train: seeds [{base_seed}, {base_seed+n_train})")
    print(f"  val:   seeds [{base_seed+n_train}, {base_seed+n_train+n_val})")
    print(f"  test:  seeds [{base_seed+n_train+n_val}, {base_seed+total})")
    if not args.no_forest_disconnect:
        print(f"  forest disconnect: threshold={args.forest_threshold} "
              f"sigma={args.blob_sigma} blob_thr={args.blob_threshold}")
    print()

    splits = {
        "train": (base_seed, n_train),
        "val":   (base_seed + n_train, n_val),
        "test":  (base_seed + n_train + n_val, n_test),
    }

    saved_maps = {}
    for name, (seed_start, count) in splits.items():
        print(f"Generating {count} {name} maps ...")
        maps = _generate_maps(seed_start, count, config)

        if not args.no_forest_disconnect:
            print(f"  Applying forest disconnection ...")
            maps = _apply_forest_disconnection(
                maps, compiled,
                args.forest_threshold, args.blob_sigma, args.blob_threshold,
                base_seed=seed_start,
            )

        path = out_dir / f"{name}_{tag}_n{count}.pt"
        _save_split(maps, seed_start, config.size, path)
        saved_maps[name] = maps
        print()

    if not args.no_preview and n_train >= 3:
        print("Generating spawn/target preview images ...")
        for stage in [CurriculumStage.EASY, CurriculumStage.NORMAL]:
            preview_path = out_dir / f"dataset_preview_{stage.value}.png"
            _make_preview(
                saved_maps["train"], config, stage, preview_path,
                n_maps=min(3, n_train),
                easy_radius=50,
            )


if __name__ == "__main__":
    main()
