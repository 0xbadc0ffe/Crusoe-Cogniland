"""Generate island maps for the Survival Kit game.

Four biomes (balanced, archipelago, highland, grassland), each with its own
sink_mode + threshold profile. Each map is:
  1. Generated at GEN_SIZE x GEN_SIZE via generate_island
  2. Biome-modified (highland adds a ridge overlay)
  3. Center-cropped to CROP_SIZE x CROP_SIZE
  4. Seeded with berry tiles on ~BERRY_FRAC of pixels, concentrated in the
     tallest forest
  5. Painted with a 1-pixel lethal border (black on render)

Usage:
    python scripts/generate_maps.py --preview
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from cogniland_jax.mapgen.simplexnoise.noise import SimplexNoise, normalize

# ── Constants ───────────────────────────────────────────────────────────────

GEN_SIZE = 170
CROP_SIZE = 128
BERRY_FRAC = 0.02
DEADLY_VALUE = -1.0

TERRAIN_NAMES = [
    "ocean", "deep_water", "water", "beach", "sandy",
    "grassland", "forest", "rocky", "mountains",
]
DEFAULT_THRESHOLDS = np.array(
    [0.007, 0.025, 0.05, 0.06, 0.1, 0.25, 0.6, 0.7, 1.0]
)


def generate_island(
    size: int,
    seed: int,
    sink_mode: int = 1,
    scale: float = 0.33,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    filtering: str = "square",
) -> np.ndarray:
    """Generate a single island heightmap using simplex noise."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    sn = SimplexNoise(num_octaves=octaves, persistence=persistence, dimensions=2)
    hgrid = size * scale

    world = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            world[i, j] = normalize(sn.fractal(i, j, hgrid=hgrid, lacunarity=lacunarity))

    if sink_mode == 1:
        world = world ** 3
    elif sink_mode == 2:
        world = (2 * world) ** 2

    world = world / world.max()

    if filtering:
        center = size // 2
        grad = np.zeros((size, size), dtype=np.float64)
        for y in range(size):
            for x in range(size):
                dx = abs(x - center)
                dy = abs(y - center)
                if filtering == "circle":
                    dist = math.sqrt(dx * dx + dy * dy)
                elif filtering == "diamond":
                    dist = dx + dy
                elif filtering == "square":
                    dist = max(dx ** 2, dy ** 2)
                else:
                    raise ValueError(f"Unknown filtering: {filtering}")
                grad[y, x] = dist
        grad = grad / grad.max()
        grad = -(grad - 0.5) * 2.0
        grad[grad > 0] *= 20
        grad = grad / grad.max()

        world_noise = world * grad
        world_noise[world_noise > 0] *= 20
        world_noise = world_noise / world_noise.max()
        world = world_noise

    return world.astype(np.float32)

#                                   ocean   dw    water  beach  sandy  grass  forest rocky  mtn
THRESHOLDS_ARCHIPELAGO = np.array([0.015, 0.05,  0.15,  0.18,  0.22,  0.45,  0.75,  0.85,  1.0])
THRESHOLDS_GRASSLAND   = np.array([0.20,  0.25,  0.28,  0.34,  0.35,  0.75,  0.95,  0.98,  1.0])
THRESHOLDS_HIGHLAND    = np.array([0.20,  0.25,  0.28,  0.32,  0.36,  0.45,  0.65,  0.80,  1.0])

BIOME_THRESHOLDS: dict[str, np.ndarray] = {
    "balanced":    DEFAULT_THRESHOLDS,
    "archipelago": THRESHOLDS_ARCHIPELAGO,
    "grassland":   THRESHOLDS_GRASSLAND,
    "highland":    THRESHOLDS_HIGHLAND,
}

BIOME_SINK_MODE: dict[str, int] = {
    "balanced":    1,   # matches generate_dataset.py's default MapGenConfig
    "archipelago": 1,
    "grassland":   0,
    "highland":    0,
}

ALL_BIOMES = ["balanced", "archipelago", "highland", "grassland"]
FOREST_CLASS_IDX = TERRAIN_NAMES.index("forest")
BEACH_CLASS_IDX = TERRAIN_NAMES.index("beach")
BEACH_BERRY_PROB = 0.05

# Subtle gradient ramps anchored on the default terrain colors from
# configs/env/default.yaml. Each ramp is (color at class lower, color at class
# upper). Deltas are ±6 per channel — visually almost imperceptible, but
# distinguishable inside each class. Direction follows: shallower (t→1) is
# lighter for water/beach/sandy/grassland, and darker for forest/rocky/mountains.
CLASS_RAMPS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "ocean":      ((0, 25, 215),     (15, 45, 235)),
    "deep_water": ((15, 55, 215),    (35, 75, 235)),
    "water":      ((55, 95, 215),    (75, 115, 235)),
    "beach":      ((228, 204, 165),  (248, 224, 185)),
    "sandy":      ((200, 170, 130),  (220, 190, 150)),
    "grassland":  ((24, 129, 24),    (44, 149, 44)),
    "forest":     ((10, 110, 10),    (0, 90, 0)),
    "rocky":      ((149, 147, 147),  (129, 127, 127)),
    "mountains":  ((255, 255, 255),  (245, 240, 240)),
}
BERRY_COLOR = (155, 35, 60)
DEADLY_COLOR = (0, 0, 0)


# ── Per-biome terrain helpers ───────────────────────────────────────────────

def _terrain_idx(hm: np.ndarray, biome: str) -> np.ndarray:
    return np.searchsorted(BIOME_THRESHOLDS[biome], hm).clip(0, len(TERRAIN_NAMES) - 1)


def _terrain_fractions(hm: np.ndarray, biome: str) -> dict[str, float]:
    idx = _terrain_idx(hm, biome)
    n = idx.size
    return {name: float((idx == i).sum()) / n
            for i, name in enumerate(TERRAIN_NAMES)}


# ── Ridge overlay (highland only) ──────────────────────────────────────────

def _fbm_noise(size: int, scale: float, octaves: int = 6,
               persistence: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    result = np.zeros((size, size), dtype=np.float64)
    amplitude, freq = 1.0, 1.0
    for _ in range(octaves):
        noise = rng.uniform(-1, 1, (size, size))
        sigma = scale / freq
        if sigma > 0.5:
            noise = gaussian_filter(noise, sigma=sigma, mode="wrap")
        result += noise * amplitude
        amplitude *= persistence
        freq *= 2.0
    lo, hi = result.min(), result.max()
    if hi - lo > 1e-10:
        result = (result - lo) / (hi - lo)
    return result.astype(np.float32)


def _add_ridge(hm: np.ndarray, size: int, seed: int,
               rng: np.random.RandomState) -> None:
    angle = rng.uniform(0, math.pi)
    cx, cy = size / 2, size / 2
    Y, X = np.mgrid[:size, :size]
    dist = (X - cx) * math.sin(angle) - (Y - cy) * math.cos(angle)
    warp = _fbm_noise(size, scale=30, octaves=4, seed=seed + 8000)
    dist = np.abs(dist + (warp - 0.5) * 60)
    ridge = np.clip(1.0 - dist / 15, 0, 1) ** 1.3
    land = hm > 0.05
    hm[land] = hm[land] + ridge[land] * (0.98 - hm[land]) * 0.8
    np.clip(hm, 0, 0.99, out=hm)


# ── Generation pipeline stages ──────────────────────────────────────────────

def generate_raw_heightmap(seed: int, biome: str) -> np.ndarray:
    """Stage 1 — raw generate_island output at GEN_SIZE, pre-biome-mods."""
    return generate_island(size=GEN_SIZE, seed=seed, sink_mode=BIOME_SINK_MODE[biome])


def apply_biome_mods(hm: np.ndarray, biome: str, seed: int) -> np.ndarray:
    """Stage 2 — per-biome post-processing (highland adds a ridge)."""
    hm = hm.copy()
    if biome == "highland":
        rng = np.random.RandomState(seed)
        _add_ridge(hm, hm.shape[0], seed, rng)
    return hm


def center_crop(hm: np.ndarray, size: int) -> np.ndarray:
    """Stage 3a — center-crop to size x size."""
    h, w = hm.shape
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    return hm[r0:r0 + size, c0:c0 + size].copy()


def sample_berry_mask(hm: np.ndarray, biome: str, target_frac: float,
                      seed: int) -> np.ndarray:
    """Stage 3b — boolean mask of berry tiles.

    Only forest pixels are eligible. Probability per forest pixel is
    proportional to its local height within the forest class, scaled so the
    expected berry count equals ``target_frac`` of forest pixels.

    Each sampled seed is then dilated into a 2x2 patch (``(r,c)``,
    ``(r+1,c)``, ``(r,c+1)``, ``(r+1,c+1)``), clipped to eligible terrain
    (forest + beach) so berries never bleed onto water / rocks.
    """
    idx = _terrain_idx(hm, biome)
    forest_mask = idx == FOREST_CLASS_IDX
    beach_mask = idx == BEACH_CLASS_IDX
    berry_mask = np.zeros_like(forest_mask, dtype=bool)
    rng = np.random.RandomState(seed + 12345)

    forest_count = int(forest_mask.sum())
    if forest_count > 0:
        thresholds = BIOME_THRESHOLDS[biome]
        f_lo = thresholds[FOREST_CLASS_IDX - 1]
        f_hi = thresholds[FOREST_CLASS_IDX]
        local_h = np.clip((hm[forest_mask] - f_lo) / max(f_hi - f_lo, 1e-6), 0.0, 1.0)
        weights = local_h + 1e-3
        target_count = target_frac * forest_count
        k = target_count / weights.sum()
        probs = np.clip(weights * k, 0.0, 1.0)
        berry_mask[forest_mask] = rng.random(len(probs)) < probs

    beach_count = int(beach_mask.sum())
    if beach_count > 0:
        berry_mask[beach_mask] = rng.random(beach_count) < BEACH_BERRY_PROB

    # Dilate each berry seed into a 2x2 patch, clipped to forest+beach so
    # berries never leak onto ineligible terrain. Paints down-right from each
    # seed; seeds on the bottom/right edge simply paint fewer neighbours.
    eligible = forest_mask | beach_mask
    down = np.zeros_like(berry_mask)
    down[1:, :] = berry_mask[:-1, :]
    right = np.zeros_like(berry_mask)
    right[:, 1:] = berry_mask[:, :-1]
    diag = np.zeros_like(berry_mask)
    diag[1:, 1:] = berry_mask[:-1, :-1]
    dilated = berry_mask | (down & eligible) | (right & eligible) | (diag & eligible)

    return dilated


def paint_deadly_border(hm: np.ndarray, value: float = DEADLY_VALUE) -> np.ndarray:
    """Stage 3c — 1-pixel lethal border."""
    hm = hm.copy()
    hm[0, :] = value
    hm[-1, :] = value
    hm[:, 0] = value
    hm[:, -1] = value
    return hm


# ── Colorize ────────────────────────────────────────────────────────────────

def colorize_gradient(hm: np.ndarray, biome: str,
                      berry_mask: np.ndarray | None = None) -> np.ndarray:
    """Render hm with per-class gradient ramps, berry overlay, and deadly border."""
    thresholds = BIOME_THRESHOLDS[biome].astype(np.float32)
    names = TERRAIN_NAMES
    lo_colors = np.array([CLASS_RAMPS[n][0] for n in names], dtype=np.float32)
    hi_colors = np.array([CLASS_RAMPS[n][1] for n in names], dtype=np.float32)

    idx = _terrain_idx(hm, biome)
    lower = np.concatenate(([0.0], thresholds[:-1]))

    lo = lower[idx]
    hi = thresholds[idx]
    t = np.clip((hm - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    out = lo_colors[idx] * (1.0 - t[..., None]) + hi_colors[idx] * t[..., None]
    out = np.clip(out, 0, 255)

    if berry_mask is not None and berry_mask.any():
        out[berry_mask] = BERRY_COLOR

    deadly = hm <= DEADLY_VALUE / 2
    out[deadly] = DEADLY_COLOR
    return out.astype(np.uint8)


# ── Map dataclass & full generation ─────────────────────────────────────────

@dataclass
class MapData:
    heightmap: torch.Tensor     # [CROP_SIZE, CROP_SIZE], includes deadly border
    berry_mask: torch.Tensor    # [CROP_SIZE, CROP_SIZE] bool
    biome: str
    seed: int
    terrain_fractions: dict = field(default_factory=dict)


def generate_map(seed: int, biome: str) -> MapData:
    raw = generate_raw_heightmap(seed, biome)
    modified = apply_biome_mods(raw, biome, seed)
    cropped = center_crop(modified, CROP_SIZE)
    berry_mask = sample_berry_mask(cropped, biome, BERRY_FRAC, seed)
    bordered = paint_deadly_border(cropped)
    return MapData(
        heightmap=torch.from_numpy(bordered),
        berry_mask=torch.from_numpy(berry_mask),
        biome=biome,
        seed=seed,
        terrain_fractions=_terrain_fractions(cropped, biome),
    )


def generate_dataset(base_seed: int, count_per_biome: int = 3) -> list[MapData]:
    maps: list[MapData] = []
    for biome in ALL_BIOMES:
        for i in range(count_per_biome):
            m = generate_map(base_seed + i, biome=biome)
            maps.append(m)
            f = m.terrain_fractions
            berry_frac = m.berry_mask.float().mean().item()
            print(f"  {biome:>12} seed={m.seed} "
                  f"forest={f.get('forest', 0):.0%} rocky={f.get('rocky', 0):.0%} "
                  f"mtn={f.get('mountains', 0):.0%} berry={berry_frac:.1%}")
    return maps


# ── Preview ─────────────────────────────────────────────────────────────────

def _preview_grid(maps: list[MapData], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(ALL_BIOMES)
    nrows = max(sum(1 for m in maps if m.biome == b) for b in ALL_BIOMES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    biome_maps = {b: [m for m in maps if m.biome == b] for b in ALL_BIOMES}
    for col, biome in enumerate(ALL_BIOMES):
        for row, m in enumerate(biome_maps[biome]):
            ax = axes[row, col]
            rgb = colorize_gradient(m.heightmap.numpy(), m.biome, m.berry_mask.numpy())
            ax.imshow(rgb, interpolation="nearest")
            title = f"seed={m.seed}"
            if row == 0:
                title = f"{biome.upper()}\n{title}"
            ax.set_title(title, fontsize=9)
            ax.set_axis_off()
        for row in range(len(biome_maps[biome]), nrows):
            axes[row, col].set_visible(False)

    fig.suptitle("Maps", fontsize=14, y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Preview saved: {output_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=3, help="Maps per biome")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--output-dir", type=str, default="data/maps")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    print(f"Generating {args.count} maps per biome (seed={args.seed})")
    maps = generate_dataset(args.seed, count_per_biome=args.count)

    out_dir.mkdir(parents=True, exist_ok=True)
    heightmaps = torch.stack([m.heightmap for m in maps])
    berry_masks = torch.stack([m.berry_mask for m in maps])
    save_path = out_dir / f"maps_seed{args.seed}_n{len(maps)}.pt"
    torch.save({
        "maps": heightmaps,
        "berry_masks": berry_masks,
        "metadata": {
            "seed": args.seed,
            "count_per_biome": args.count,
            "biomes": [m.biome for m in maps],
            "gen_size": GEN_SIZE,
            "crop_size": CROP_SIZE,
            "berry_frac": BERRY_FRAC,
        },
    }, save_path)
    print(f"\nSaved: {save_path} ({len(maps)} maps)")

    if args.preview:
        _preview_grid(maps, out_dir / "preview.png")


if __name__ == "__main__":
    main()
