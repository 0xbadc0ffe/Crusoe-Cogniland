"""CLI: build the cogniland-jax map dataset.

Default output matches the env's load path:

    data/maps/
        train.pt       64 maps × 4 biomes = 256
        val.pt          4 maps × 4 biomes =  16
        test.pt         4 maps × 4 biomes =  16

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --base-seed 100 --output-dir data/maps --preview
"""

from __future__ import annotations

import argparse

from cogniland_jax.mapgen.build import build_dataset, SPLITS_PER_BIOME


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--output-dir", type=str, default="data/maps")
    ap.add_argument("--preview", action="store_true",
                    help="Save a val-set grid PNG alongside the .pt files")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Processes for visibility LUT precompute "
                         "(default: os.cpu_count(); set 1 to run inline)")
    ap.add_argument("--train-per-biome", type=int, default=SPLITS_PER_BIOME["train"])
    ap.add_argument("--val-per-biome", type=int, default=SPLITS_PER_BIOME["val"])
    ap.add_argument("--test-per-biome", type=int, default=SPLITS_PER_BIOME["test"])
    args = ap.parse_args()

    splits = {
        "train": args.train_per_biome,
        "val": args.val_per_biome,
        "test": args.test_per_biome,
    }
    build_dataset(
        output_dir=args.output_dir,
        base_seed=args.base_seed,
        splits_per_biome=splits,
        num_workers=args.num_workers,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
