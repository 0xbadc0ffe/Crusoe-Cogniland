"""Generate a MapDataset of pre-built island maps with disjoint train/val/test splits.

Usage:
    python scripts/generate_dataset.py [options]

    # Quick smoke-test (small dataset):
    python scripts/generate_dataset.py --seed 42 --train 8 --val 4 --test 4 \
        --output /tmp/test_maps.pt

    # Full-size dataset:
    python scripts/generate_dataset.py --seed 42 --train 128 --val 16 --test 16

Seed assignment (disjoint by construction):
    train maps: seeds [base, base+n_train)
    val   maps: seeds [base+n_train, base+n_train+n_val)
    test  maps: seeds [base+n_train+n_val, ...)

Also saves two preview PNG images showing EASY vs NORMAL spawn/target placements
for 3 sample training maps.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Make sure the package is importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cogniland.env.constants import TERRAIN_THRESHOLDS
from cogniland.env.dataset import MapDataset
from cogniland.env.islands import colorize, generate_island
from cogniland.env.types import CurriculumStage, EnvConfig


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
    return torch.stack(maps)  # [count, H, W]


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
) -> None:
    """Save a 2×n_maps grid of maps with spawn (green) and target (red) overlaid."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    land_threshold = TERRAIN_THRESHOLDS[2].item()
    center = config.size // 2
    radius = config.curriculum_easy_radius
    rng = random.Random(0)

    fig, axes = plt.subplots(2, n_maps, figsize=(4 * n_maps, 9))
    fig.suptitle(f"Spawn/Target placement — {stage.value.upper()} stage", fontsize=14)

    for col in range(n_maps):
        wm = train_maps[col]
        rgb = colorize(wm, config).numpy().astype("uint8")

        for row in range(2):
            ax = axes[row, col]
            ax.imshow(rgb)
            ax.axis("off")
            ax.set_title(f"Map {col}", fontsize=9)

            # Sample positions
            spawn = _sample_land_position(wm, land_threshold, rng, stage, center, radius)
            target = _sample_land_position(wm, land_threshold, rng, stage, center, radius)

            # Dots for spawn (green) and target (red)
            ax.scatter([spawn[1]], [spawn[0]], c="lime", s=60, zorder=5, marker="o",
                       edgecolors="black", linewidths=0.5)
            ax.scatter([target[1]], [target[0]], c="red", s=60, zorder=5, marker="*",
                       edgecolors="black", linewidths=0.5)

            # EASY boundary circle
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Cogniland MapDataset")
    parser.add_argument("--seed",   type=int, default=42,    help="Base RNG seed")
    parser.add_argument("--train",  type=int, default=128,   help="Number of training maps")
    parser.add_argument("--val",    type=int, default=16,    help="Number of validation maps")
    parser.add_argument("--test",   type=int, default=16,    help="Number of test maps")
    parser.add_argument(
        "--output", type=str,
        default="",
        help="Output .pt path (default: data/maps_seed{seed}_train{n}_val{n}_test{n}.pt)",
    )
    parser.add_argument("--no-preview", action="store_true", help="Skip PNG preview generation")
    args = parser.parse_args()

    n_train, n_val, n_test = args.train, args.val, args.test
    base_seed = args.seed

    output_path = Path(args.output) if args.output else Path(
        f"data/maps_seed{base_seed}_train{n_train}_val{n_val}_test{n_test}.pt"
    )
    preview_dir = output_path.parent

    # Build a minimal EnvConfig (default params, just needs island generation settings)
    config = EnvConfig(seed=base_seed)

    total = n_train + n_val + n_test
    print(f"Generating {total} maps ({config.size}×{config.size}) with base seed {base_seed}")
    print(f"  train: seeds [{base_seed}, {base_seed+n_train})")
    print(f"  val:   seeds [{base_seed+n_train}, {base_seed+n_train+n_val})")
    print(f"  test:  seeds [{base_seed+n_train+n_val}, {base_seed+total})")
    print()

    print(f"Generating {n_train} training maps ...")
    train_maps = _generate_maps(base_seed, n_train, config)

    print(f"\nGenerating {n_val} validation maps ...")
    val_maps = _generate_maps(base_seed + n_train, n_val, config)

    print(f"\nGenerating {n_test} test maps ...")
    test_maps = _generate_maps(base_seed + n_train + n_val, n_test, config)

    dataset = MapDataset(
        train_maps=train_maps,
        val_maps=val_maps,
        test_maps=test_maps,
        seed=base_seed,
        map_size=config.size,
    )
    dataset.save(output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nDataset saved: {output_path}  ({size_mb:.1f} MB)")
    print(f"  train={n_train}  val={n_val}  test={n_test}  map_size={config.size}")

    if not args.no_preview and n_train >= 3:
        print("\nGenerating spawn/target preview images ...")
        for stage in [CurriculumStage.EASY, CurriculumStage.NORMAL]:
            preview_path = preview_dir / f"dataset_preview_{stage.value}.png"
            _make_preview(train_maps, config, stage, preview_path, n_maps=min(3, n_train))


if __name__ == "__main__":
    main()
