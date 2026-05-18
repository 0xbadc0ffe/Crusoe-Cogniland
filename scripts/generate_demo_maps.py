#!/usr/bin/env python3
"""Generate the 12 curated demo maps used by ``scripts/play_cogniland.py``.

We render four maps each of three biomes (``balanced``, ``lake``, ``rocky``)
at size 64. Seeds are deterministic so re-running this script produces the
same dataset:

    seed = 1000 + biome_idx * 100 + sample_idx

For each map we save two artifacts to ``data/demo_maps/``:

* ``<biome>_<idx>.pkl`` — the full :class:`cogniland.nav.MapRecord` instance
  pickled directly. We pickle the dataclass (rather than the dict from
  ``to_dict()``) so the precomputed ``ctg_*`` arrays travel with it.
* ``<biome>_<idx>.png`` — a small thumbnail rendered with
  :class:`SpriteSheet.render_full` for use in the play-cogniland map picker.

Usage
-----
    python scripts/generate_demo_maps.py
    python scripts/generate_demo_maps.py --tile-px 6 --size 64
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# allow `python scripts/generate_demo_maps.py` from the repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogniland.nav.mapgen import generate_map  # noqa: E402
from cogniland.nav.renderer import SpriteSheet  # noqa: E402


# Three biomes × 4 samples each = 12 demo maps.
BIOMES: tuple[str, ...] = ("balanced", "lake", "rocky")
SAMPLES_PER_BIOME: int = 4


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--tile-px", type=int, default=6,
                   help="thumbnail tile size in pixels")
    p.add_argument("--out-dir", default="data/demo_maps")
    p.add_argument("--force", action="store_true",
                   help="re-generate even if the .pkl already exists")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sprites = SpriteSheet(tile_px=args.tile_px)

    for biome_idx, biome in enumerate(BIOMES):
        for sample_idx in range(SAMPLES_PER_BIOME):
            seed = 1000 + biome_idx * 100 + sample_idx
            pkl_path = out_dir / f"{biome}_{sample_idx}.pkl"
            png_path = out_dir / f"{biome}_{sample_idx}.png"

            if pkl_path.exists() and png_path.exists() and not args.force:
                print(f"[skip] {pkl_path}")
                continue

            print(f"[gen ] {biome}_{sample_idx}  seed={seed}", flush=True)
            record = generate_map(size=args.size, map_type=biome, seed=seed)

            # Pickle the dataclass directly. Re-loading via
            # `pickle.load(open(...))` returns a MapRecord instance, which
            # `CognilandNavEnv(map_record=...)` consumes as-is.
            with pkl_path.open("wb") as f:
                pickle.dump(record, f)

            thumb = sprites.render_full(
                terrain=record.terrain,
                agent_pos=(int(record.spawn[0]), int(record.spawn[1])),
                target_pos=(int(record.target[0]), int(record.target[1])),
                view_rect=None,
                agent_facing="down",
            )
            Image.fromarray(thumb.astype(np.uint8)).save(png_path)
            print(f"  wrote {pkl_path}  +  {png_path}")

    print(f"\ndone — {len(BIOMES) * SAMPLES_PER_BIOME} maps in {out_dir}")


if __name__ == "__main__":
    main()
