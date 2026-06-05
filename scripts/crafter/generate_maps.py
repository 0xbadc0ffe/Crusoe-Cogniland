"""Pre-generate map datasets for crafter_in_cogniland.

Run this once before launching a sweep so multiple agents don't race to
write the same .pkl file. By default it generates 4 sizes ×
balanced/lake/rocky × 256 maps each, into
``data/crafter_in_cogniland/train_<size>x<size>_n256.pkl``.

Usage:
    python scripts/crafter/generate_maps.py
    python scripts/crafter/generate_maps.py --sizes 64 128 --num-maps 512
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cogniland.crafter_in_cogniland import (
    generate_map_dataset, save_map_arrays,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[32, 64, 96, 128],
                   help="map side lengths to generate")
    p.add_argument("--num-maps", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="data/crafter_in_cogniland")
    p.add_argument("--map-types", nargs="+",
                   default=["balanced", "lake", "rocky"])
    p.add_argument("--generator", default="composed",
                   choices=["components", "composed", "simplex"],
                   help="terrain recipe (see cogniland.nav.mapgen)")
    p.add_argument("--force", action="store_true",
                   help="re-generate even if the .pkl already exists")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        path = out_dir / f"train_{size}x{size}_n{args.num_maps}.pkl"
        if path.exists() and not args.force:
            print(f"[skip] {path} exists")
            continue
        t = time.time()
        print(f"[gen ] {path} ({size}x{size}, n={args.num_maps}, "
              f"types={args.map_types}, gen={args.generator}) …", flush=True)
        arrays = generate_map_dataset(
            n_maps=args.num_maps, size=size,
            map_types=tuple(args.map_types), seed=args.seed,
            generator=args.generator,
        )
        save_map_arrays(arrays, path)
        print(f"[done] {path}  ({time.time() - t:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
