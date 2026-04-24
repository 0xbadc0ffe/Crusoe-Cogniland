"""Classic map generator: balanced terrain, round shape, mountain-edge berries.

Reuses the existing pipeline from ``generate_maps.py`` but overrides:

  * sink_mode = 0 (grassland-style), so islands come out round and filled
    rather than the jagged cubic-sink shapes of default "balanced" maps.
  * filtering = "circle", so the island falloff is Euclidean (round).
  * biome thresholds = BALANCED, keeping forests / mountains / rocky / beach.
  * berry_mask = single-pixel berries on forest tiles that border a rocky
    tile (8-connected). No 2x2 dilation, no beach berries.

Usage:
    # Preview only
    python scripts/generate_classic_maps.py --preview --count 8

    # Generate train/val/test dataset with visibility LUTs
    python scripts/generate_classic_maps.py --dataset \\
        --output-dir data/maps_classic \\
        --train 16 --val 4 --test 4
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

import generate_maps as gt
from precompute_visibility import compute_visibility_luts


CLASSIC_BIOME = "balanced"
CLASSIC_SINK_MODE = 0  # grassland's sink_mode: round, filled islands
CLASSIC_FILTERING = "circle"  # Euclidean island falloff → round landmass

# Per-eligible-tile probability for berries. Eligible = forest tile with at
# least one rocky neighbour (8-connected).
BERRY_PROB_FOREST_NEAR_ROCKY = 0.05


def sample_classic_berry_mask(terrain_idx: np.ndarray, seed: int) -> np.ndarray:
    """Single-pixel berries on forest tiles adjacent to a rocky tile."""
    forest = terrain_idx == gt.FOREST_CLASS_IDX
    rocky_idx = gt.TERRAIN_NAMES.index("rocky")
    rocky = terrain_idx == rocky_idx

    # 8-connected dilation of the rocky mask: a cell is "rocky-adjacent" if
    # any of its 8 neighbours is rocky.
    pad = np.pad(rocky, 1, mode="constant", constant_values=False)
    rocky_adj = np.zeros_like(rocky)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rocky_adj |= pad[1 + dr : 1 + dr + rocky.shape[0],
                             1 + dc : 1 + dc + rocky.shape[1]]

    eligible = forest & rocky_adj
    rng = np.random.RandomState(seed + 12345)
    berry_mask = np.zeros_like(forest, dtype=bool)
    berry_mask[eligible] = rng.random(int(eligible.sum())) < BERRY_PROB_FOREST_NEAR_ROCKY
    return berry_mask


def generate_classic_heightmap(seed: int) -> np.ndarray:
    """Raw heightmap at GEN_SIZE using grassland-style sink_mode=0 and a
    circular island falloff so the landmass is round rather than square."""
    return gt.generate_island(
        size=gt.GEN_SIZE,
        seed=seed,
        sink_mode=CLASSIC_SINK_MODE,
        filtering=CLASSIC_FILTERING,
    )


def build_classic_map(seed: int):
    """Return (rgb, bordered_heightmap, terrain_idx, berry_mask) for one map."""
    raw = generate_classic_heightmap(seed)
    cropped = gt.center_crop(raw, gt.CROP_SIZE)

    terrain_idx_full = gt._terrain_idx(cropped, CLASSIC_BIOME)
    berry_mask = sample_classic_berry_mask(terrain_idx_full, seed)

    terrain_idx = terrain_idx_full.astype(np.int8)
    terrain_idx[0, :] = -1
    terrain_idx[-1, :] = -1
    terrain_idx[:, 0] = -1
    terrain_idx[:, -1] = -1

    bordered = gt.paint_deadly_border(cropped).astype(np.float32)
    rgb = gt.colorize_gradient(bordered, CLASSIC_BIOME, berry_mask)
    return rgb, bordered, terrain_idx, berry_mask


def _terrain_fractions_classic(hm: np.ndarray) -> dict[str, float]:
    idx = gt._terrain_idx(hm, CLASSIC_BIOME)
    n = idx.size
    return {name: float((idx == i).sum()) / n
            for i, name in enumerate(gt.TERRAIN_NAMES)}


# ── Preview ─────────────────────────────────────────────────────────────────

def save_preview(seeds: list[int], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(seeds)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)

    for i, seed in enumerate(seeds):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        rgb, bordered, _, berry_mask = build_classic_map(seed)
        f = _terrain_fractions_classic(bordered)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(
            f"seed={seed}  berries={int(berry_mask.sum())}\n"
            f"grass={f['grassland']:.0%} forest={f['forest']:.0%} "
            f"rocky={f['rocky']:.0%} mtn={f['mountains']:.0%}",
            fontsize=8,
        )
        ax.set_axis_off()

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"Classic maps — balanced thresholds, sink_mode=0, "
        f"berries on forest∩rocky-adj ({n} seeds)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Preview saved: {output_path}")


# ── Dataset ─────────────────────────────────────────────────────────────────

def build_split(n: int, base_seed: int, num_workers: int | None):
    rgbs, hms, tidxs, masks, biomes, seeds = [], [], [], [], [], []
    seed = base_seed
    for _ in range(n):
        rgb, hm, tidx, mask = build_classic_map(seed)
        rgbs.append(rgb)
        hms.append(hm)
        tidxs.append(tidx)
        masks.append(mask)
        biomes.append(CLASSIC_BIOME)
        seeds.append(seed)
        seed += 1

    heightmaps = np.stack(hms)
    print(f"  precomputing visibility LUTs ({n} × 128·128 cells)", flush=True)
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


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=8,
                        help="Number of preview seeds (preview mode only)")
    parser.add_argument("--preview", action="store_true",
                        help="Save preview grid PNG")
    parser.add_argument("--dataset", action="store_true",
                        help="Generate train/val/test .pt files")
    parser.add_argument("--train", type=int, default=16,
                        help="Train set size (dataset mode)")
    parser.add_argument("--val", type=int, default=4,
                        help="Val set size (dataset mode)")
    parser.add_argument("--test", type=int, default=4,
                        help="Test set size (dataset mode)")
    parser.add_argument("--output-dir", type=str, default="data/maps_classic")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.preview:
        seeds = list(range(args.base_seed, args.base_seed + args.count))
        save_preview(seeds, out_dir / "preview_classic.png")

    if args.dataset:
        seed = args.base_seed
        splits = {"train": args.train, "val": args.val, "test": args.test}
        for name, n in splits.items():
            print(f"\n{name}: {n} classic maps, seeds [{seed}, {seed + n})")
            split, seed = build_split(n, seed, args.num_workers)
            path = out_dir / f"{name}.pt"
            torch.save(split, path)
            mb = path.stat().st_size / 1e6
            print(f"  saved {path.name}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
