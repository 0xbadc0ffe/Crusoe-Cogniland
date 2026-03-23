#!/usr/bin/env python3
"""Generate a small set of maps for the Cogniland demo app.

Usage:
    python scripts/generate_demo_maps.py            # 16 maps, seed 42 → data/demo_maps.pt
    python scripts/generate_demo_maps.py --n 8 --seed 0 --output data/my_maps.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cogniland.env.islands import generate_island
from cogniland.env.types import EnvConfig, MapGenConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo maps for Cogniland")
    parser.add_argument("--seed",   type=int, default=42,               help="Base RNG seed")
    parser.add_argument("--n",      type=int, default=16,               help="Number of maps")
    parser.add_argument("--output", type=str, default="data/demo_maps.pt", help="Output .pt path")
    args = parser.parse_args()

    config = EnvConfig(map_generation=MapGenConfig(seed=args.seed))
    print(f"Generating {args.n} demo maps ({config.size}×{config.size}), base seed {args.seed} ...")

    maps = []
    for i in range(args.n):
        seed_i = args.seed + i
        torch.manual_seed(seed_i)
        random.seed(seed_i)
        np.random.seed(seed_i)
        maps.append(generate_island(config))
        print(f"  [{i + 1}/{args.n}]", flush=True)

    maps_tensor = torch.stack(maps)  # [N, H, W] float32

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"maps": maps_tensor, "seed": args.seed, "map_size": config.size}, out)

    size_mb = out.stat().st_size / 1e6
    print(f"\nSaved {args.n} maps → {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
