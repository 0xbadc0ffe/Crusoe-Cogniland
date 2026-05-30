"""Component-stamp map generation — visual exploration tool.

This is a *standalone preview generator*, not wired into the training
pipeline. Its job is to let you eyeball many candidate maps so you can
pick a look + hyperparameter set before we promote a recipe into
``cogniland.nav.mapgen``.

Pipeline
--------
1. **Component library** (``build_library``): a small, fixed set of
   structured terrain shapes rasterised once onto canonical patches —
   round mountains, irregular massifs, straight/curved ridge ranges,
   round lakes, meandering rivers. Each patch is a *height delta*
   (positive = raise toward rock, negative = carve toward water).
   Because the library is fixed and seedless, the *same* shapes get
   reused across every map; variety comes only from how they are
   placed.

2. **Heightmap** (``build_heightmap``): start from a low-frequency
   simplex field (organic large-scale undulation), then stamp a
   biome-dependent number of components onto it. Each stamp is the
   canonical patch put through a random **scale**, **rotation**, and
   **translation** before being added in. Mountains/ridges come from
   the "raise" set, lakes/rivers from the "carve" set; the biome
   weights decide how many of each.

3. **Biome thresholding** (``heightmap_to_terrain``): height above
   ``rock_thr`` -> ROCK, below ``water_thr`` -> WATER, otherwise land.
   A beach band of SAND is grown a few cells out from every water
   body via a distance transform.

4. **Artifacts**: two independent simplex fields sprinkle TREE
   (forest, in moist grass) and extra SAND (dry patches) onto land.

Biome semantics match the existing env:
  * ``lake``    -> water-dominant (raft is the useful build)
  * ``rocky``   -> rock-dominant  (harness is the useful build)
  * ``balanced``-> a bit of both

Output: a folder of grid PNGs (one grid per biome x preset, many
seeds each) plus a ``components.png`` reference sheet.

Run:
    python scripts/explore_mapgen.py --out mapgen_preview \
        --grid 6 --size 64
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import opensimplex
from scipy.ndimage import distance_transform_edt, rotate, zoom

# tile ids + palette — kept local so this script doesn't depend on import
# paths, but values mirror cogniland.nav.tiles exactly.
GRASS, DIRT, SAND, WATER, ROCK, TARGET, OOB, TREE, LAVA = range(9)
TILE_COLORS = np.array(
    [
        (110, 173, 86),   # grass
        (158, 122, 80),   # dirt
        (224, 198, 130),  # sand
        (61, 113, 184),   # water
        (110, 110, 110),  # rock
        (250, 220, 60),   # target
        (0, 0, 0),        # oob
        (50, 110, 50),    # tree
        (210, 60, 30),    # lava
    ],
    dtype=np.uint8,
)


# ───────────────────────────── component library ─────────────────────────


def _coord_grid(size: int):
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    return xx, yy


def _round_peak(size: int = 36, sigma: float = 7.0) -> np.ndarray:
    """Smooth radial bump, peak 1.0 at centre."""
    xx, yy = _coord_grid(size)
    return np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))


def _massif(size: int = 44, n_lobes: int = 4, sigma: float = 6.0, seed: int = 0) -> np.ndarray:
    """Irregular round mountain: a few overlapping bumps, normalised to 1."""
    rng = np.random.default_rng(seed)
    xx, yy = _coord_grid(size)
    field_ = np.zeros((size, size), dtype=np.float64)
    for _ in range(n_lobes):
        ox, oy = rng.uniform(-size / 6, size / 6, size=2)
        s = sigma * rng.uniform(0.7, 1.3)
        field_ += np.exp(-((xx - ox) ** 2 + (yy - oy) ** 2) / (2.0 * s**2))
    return field_ / field_.max()


def _ridge(
    length: float = 34.0,
    width: float = 4.5,
    size: int = 48,
    wiggle: float = 3.0,
    wiggle_period: float = 28.0,
    seed: int = 0,
) -> np.ndarray:
    """Elongated mountain range: a horizontal spine with a gentle sine
    meander, gaussian falloff perpendicular. Rotate at placement time to
    get ranges at any orientation."""
    rng = np.random.default_rng(seed)
    xx, yy = _coord_grid(size)
    phase = rng.uniform(0, 2 * math.pi)
    spine_y = wiggle * np.sin(2 * math.pi * xx / wiggle_period + phase)
    along = np.clip(xx, -length / 2, length / 2)
    # distance to the (wiggling) spine segment
    d_perp = yy - spine_y
    d_along = np.maximum(0.0, np.abs(xx) - length / 2)
    dist = np.sqrt(d_perp**2 + d_along**2)
    h = np.exp(-(dist**2) / (2.0 * width**2))
    return h / h.max()


def _round_lake(size: int = 40, radius: float = 11.0, soft: float = 2.5) -> np.ndarray:
    """Flat-bottomed disk (logistic edge), peak 1.0 -> used as a carve."""
    xx, yy = _coord_grid(size)
    r = np.sqrt(xx**2 + yy**2)
    return 1.0 / (1.0 + np.exp((r - radius) / soft))


def _river(
    size: int = 56,
    width: float = 3.0,
    amp: float = 12.0,
    period: float = 34.0,
    seed: int = 0,
) -> np.ndarray:
    """Meandering channel crossing the patch left-to-right, carve profile."""
    rng = np.random.default_rng(seed)
    xx, yy = _coord_grid(size)
    phase = rng.uniform(0, 2 * math.pi)
    path_y = amp * np.sin(2 * math.pi * xx / period + phase)
    dist = np.abs(yy - path_y)
    h = np.exp(-(dist**2) / (2.0 * width**2))
    return h / h.max()


@dataclass
class Component:
    name: str
    patch: np.ndarray   # height delta, normalised to peak magnitude 1.0
    kind: str           # "raise" (rock) | "carve" (water)


def build_library() -> list[Component]:
    """Fixed, seedless set of canonical shapes reused across all maps."""
    return [
        Component("round_peak", _round_peak(), "raise"),
        Component("massif", _massif(seed=1), "raise"),
        Component("massif_b", _massif(seed=2, n_lobes=5), "raise"),
        Component("ridge", _ridge(seed=1), "raise"),
        Component("long_ridge", _ridge(length=42, width=3.5, size=56, seed=3), "raise"),
        Component("round_lake", _round_lake(), "carve"),
        Component("big_lake", _round_lake(size=52, radius=16, soft=3.0), "carve"),
        Component("river", _river(seed=1), "carve"),
        Component("river_b", _river(seed=4, amp=16, period=42), "carve"),
    ]


# ───────────────────────────── stamping ──────────────────────────────────


def _stamp(canvas: np.ndarray, patch: np.ndarray, cr: int, cc: int, amp: float) -> None:
    """Add ``amp * patch`` onto ``canvas`` centred at ``(cr, cc)``, clipped
    to canvas bounds (in place)."""
    ph, pw = patch.shape
    H, W = canvas.shape
    r0 = cr - ph // 2
    c0 = cc - pw // 2
    # overlap region in canvas coords
    rr0, rr1 = max(0, r0), min(H, r0 + ph)
    cc0, cc1 = max(0, c0), min(W, c0 + pw)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    # corresponding patch region
    pr0, pr1 = rr0 - r0, rr1 - r0
    pc0, pc1 = cc0 - c0, cc1 - c0
    canvas[rr0:rr1, cc0:cc1] += amp * patch[pr0:pr1, pc0:pc1]


def _transform_patch(patch: np.ndarray, scale: float, angle: float) -> np.ndarray:
    p = patch
    if abs(scale - 1.0) > 1e-3:
        p = zoom(p, scale, order=1)
    if abs(angle) > 1e-3:
        p = rotate(p, angle, order=1, reshape=True, mode="constant", cval=0.0)
    return p


# ───────────────────────────── simplex helper ────────────────────────────


def _simplex_field(seed: int, size: int, scales) -> np.ndarray:
    """Multi-octave simplex on a size x size grid, weights given by
    ``scales`` ({feature_size: weight}), normalised to unit total weight."""
    if not isinstance(scales, dict):
        scales = {scales: 1.0}
    sim = opensimplex.OpenSimplex(seed=seed)
    out = np.zeros((size, size), dtype=np.float64)
    tot = 0.0
    for s, w in scales.items():
        coords = np.arange(size, dtype=np.float64) / s
        out += w * sim.noise2array(coords, coords)
        tot += w
    return out / tot if tot else out


# ───────────────────────────── hyperparameters ───────────────────────────


@dataclass
class Preset:
    name: str
    # base heightmap
    base_amp: float = 0.30
    base_scale: float = 18.0
    # per-component amplitude (how strongly one stamp pushes height)
    comp_amp: float = 1.0
    comp_scale_range: tuple[float, float] = (0.7, 1.5)
    # biome -> (n_raise, n_carve) component counts
    counts: dict = field(
        default_factory=lambda: {
            "lake": (1, 4),
            "rocky": (4, 1),
            "balanced": (2, 2),
        }
    )
    # thresholds on the (roughly [-1,1]) heightmap
    rock_thr: float = 0.45
    water_thr: float = -0.45
    sand_band: int = 2          # beach width in cells around water
    # artifacts
    forest_thr: float = 0.30    # moisture simplex cutoff for trees
    forest_density: float = 0.55
    desert_thr: float = 0.45    # dryness simplex cutoff for sand patches


def heightmap(preset: Preset, biome: str, seed: int, size: int, lib: list[Component]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = preset.base_amp * _simplex_field(seed * 13 + 1, size, {preset.base_scale: 1.0, preset.base_scale / 3: 0.3})

    raises = [c for c in lib if c.kind == "raise"]
    carves = [c for c in lib if c.kind == "carve"]
    n_raise, n_carve = preset.counts[biome]

    def place(pool, n, sign):
        for _ in range(n):
            comp = pool[rng.integers(len(pool))]
            scale = rng.uniform(*preset.comp_scale_range)
            angle = rng.uniform(0, 360)
            patch = _transform_patch(comp.patch, scale, angle)
            cr = int(rng.integers(0, size))
            cc = int(rng.integers(0, size))
            amp = sign * preset.comp_amp * rng.uniform(0.8, 1.2)
            _stamp(H, patch, cr, cc, amp)

    place(raises, n_raise, +1.0)
    place(carves, n_carve, -1.0)
    return H


def heightmap_to_terrain(H: np.ndarray, preset: Preset, seed: int) -> np.ndarray:
    size = H.shape[0]
    terrain = np.full((size, size), GRASS, dtype=np.int8)
    rock_mask = H > preset.rock_thr
    water_mask = H < preset.water_thr
    terrain[rock_mask] = ROCK
    terrain[water_mask] = WATER

    # beach: land cells within sand_band of any water
    if preset.sand_band > 0 and water_mask.any():
        dist_to_water = distance_transform_edt(~water_mask)
        beach = (dist_to_water <= preset.sand_band) & ~water_mask & ~rock_mask
        terrain[beach] = SAND

    land = (terrain == GRASS)
    # forest: moist clusters on grass
    moisture = _simplex_field(seed * 17 + 7, size, {9: 1.0, 4: 0.4})
    forest = land & (moisture > preset.forest_thr)
    forest &= np.random.default_rng(seed * 3 + 5).random((size, size)) < preset.forest_density
    terrain[forest] = TREE
    # desert: dry sand patches on remaining grass
    dryness = _simplex_field(seed * 19 + 11, size, {11: 1.0})
    desert = (terrain == GRASS) & (dryness > preset.desert_thr)
    terrain[desert] = SAND
    return terrain


def generate(preset: Preset, biome: str, seed: int, size: int, lib: list[Component]) -> np.ndarray:
    H = heightmap(preset, biome, seed, size, lib)
    return heightmap_to_terrain(H, preset, seed)


# ─────────────────── stereotypical single-component renders ───────────────


def _fit_centered(patch: np.ndarray, size: int, frac: float, angle: float) -> np.ndarray:
    """Scale ``patch`` to ~``frac`` of ``size`` (optionally rotated) and drop
    it into the centre of a ``size x size`` zero canvas."""
    if abs(angle) > 1e-3:
        patch = rotate(patch, angle, order=1, reshape=True, mode="constant", cval=0.0)
    s = (size * frac) / max(patch.shape)
    patch = zoom(patch, s, order=1)
    canvas = np.zeros((size, size), dtype=np.float64)
    _stamp(canvas, patch, size // 2, size // 2, 1.0)
    return canvas


@dataclass
class Archetype:
    name: str
    template: np.ndarray          # canonical height-delta patch
    mode: str                     # "lake" | "river" | "rock" | "forest"
    frac: float = 0.8             # how much of the tile the feature fills
    rotate: bool = False          # randomise orientation per variant


def render_archetype(arch: Archetype, seed: int, size: int) -> np.ndarray:
    """Render one stereotypical component as terrain tiles, with a touch of
    high-freq simplex noise on the field so the boundary looks natural."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 360) if arch.rotate else 0.0
    field_ = _fit_centered(arch.template, size, arch.frac, angle)
    # tiny edge noise — perturbs the threshold boundary, never the bulk
    noise = 0.18 * _simplex_field(seed * 23 + 3, size, {6: 1.0, 3: 0.5})
    f = field_ + noise

    terrain = np.full((size, size), GRASS, dtype=np.int8)
    if arch.mode in ("lake", "river"):
        water = f > 0.5
        terrain[water] = WATER
        if water.any():
            band = 2 if arch.mode == "lake" else 1
            d = distance_transform_edt(~water)
            terrain[(d <= band) & ~water] = SAND
    elif arch.mode == "rock":
        rock = f > 0.5
        terrain[rock] = ROCK
        # thin dirt/scree skirt for a more natural foot
        if rock.any():
            d = distance_transform_edt(~rock)
            terrain[(d <= 1) & ~rock] = DIRT
    elif arch.mode == "forest":
        # density blob -> clustered trees with noisy, broken edges
        dense = f > 0.45
        speckle = rng.random((size, size)) < (0.55 + 0.45 * np.clip(f, 0, 1))
        terrain[dense & speckle] = TREE
    return terrain


def build_archetypes() -> list[Archetype]:
    return [
        Archetype("round_lake", _round_lake(), "lake", frac=0.75),
        Archetype("river", _river(seed=1), "river", frac=0.95, rotate=True),
        Archetype("ridge", _ridge(seed=1), "rock", frac=0.85, rotate=True),
        Archetype("rocky_massif", _massif(seed=1), "rock", frac=0.8, rotate=True),
    ]


def save_archetype_sheet(arch: Archetype, out: Path, size: int, n: int, ncol: int) -> None:
    """A grid of ``n`` noisy variants of a single stereotypical component."""
    terrains = [render_archetype(arch, s, size) for s in range(n)]
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 1.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, terr in zip(axes, terrains):
        ax.imshow(render_rgb(terr), interpolation="nearest")
    fig.suptitle(f"{arch.name}  ({arch.mode}, +edge noise)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out / f"comp_{arch.name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def save_archetype_overview(archs: list[Archetype], out: Path, size: int) -> None:
    """One sheet: each archetype in a row, a few variants across the columns."""
    ncol = 5
    nrow = len(archs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 1.6))
    for i, arch in enumerate(archs):
        for j in range(ncol):
            ax = axes[i, j]
            ax.axis("off")
            ax.imshow(render_rgb(render_archetype(arch, j, size)), interpolation="nearest")
            if j == 0:
                ax.set_ylabel(arch.name, fontsize=9, rotation=0, ha="right", va="center")
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
    fig.suptitle("Stereotypical components (rows) x variants (cols)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / "components_terrain.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


# ───────────────────────────── rendering ─────────────────────────────────


def render_rgb(terrain: np.ndarray) -> np.ndarray:
    return TILE_COLORS[terrain]


def save_grid(terrains: list[np.ndarray], seeds: list[int], title: str, path: Path, ncol: int) -> None:
    nrow = math.ceil(len(terrains) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.7, nrow * 1.7))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, terr, s in zip(axes, terrains, seeds):
        ax.imshow(render_rgb(terr), interpolation="nearest")
        ax.set_title(f"seed {s}", fontsize=6, pad=1)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_component_sheet(lib: list[Component], path: Path) -> None:
    n = len(lib)
    ncol = 3
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 2.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, comp in zip(axes, lib):
        cmap = "Reds" if comp.kind == "raise" else "Blues"
        ax.imshow(comp.patch, cmap=cmap, interpolation="nearest")
        ax.set_title(f"{comp.name}\n({comp.kind})", fontsize=8)
    fig.suptitle("Component library (reused via translate / rotate / scale)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ───────────────────────────── presets ───────────────────────────────────


def default_presets() -> list[Preset]:
    return [
        Preset("baseline"),
        Preset(
            "dense",
            comp_amp=1.1,
            counts={"lake": (2, 6), "rocky": (6, 2), "balanced": (4, 4)},
        ),
        Preset(
            "big_features",
            comp_scale_range=(1.2, 2.2),
            counts={"lake": (1, 3), "rocky": (3, 1), "balanced": (2, 2)},
        ),
        Preset(
            "sparse_clean",
            base_amp=0.18,
            comp_amp=1.2,
            counts={"lake": (1, 2), "rocky": (2, 1), "balanced": (1, 1)},
            forest_thr=0.40,
            desert_thr=0.55,
            sand_band=3,
        ),
        Preset(
            "noisy_organic",
            base_amp=0.45,
            base_scale=12.0,
            rock_thr=0.40,
            water_thr=-0.40,
            forest_thr=0.20,
            forest_density=0.7,
        ),
    ]


# ───────────────────────────── main ──────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="mapgen_preview", help="output folder")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--grid", type=int, default=6, help="grid is grid x grid maps")
    ap.add_argument("--seed", type=int, default=0, help="base seed offset")
    ap.add_argument(
        "--biomes",
        nargs="+",
        default=["lake", "rocky", "balanced"],
    )
    ap.add_argument("--comp-size", type=int, default=40, help="tile size for component renders")
    ap.add_argument("--comp-variants", type=int, default=16, help="variants per component sheet")
    ap.add_argument("--components-only", action="store_true",
                    help="only render the stereotypical-component sheets, skip the full-map grids")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lib = build_library()
    save_component_sheet(lib, out / "components.png")

    # stereotypical components rendered as terrain, with edge noise
    archs = build_archetypes()
    save_archetype_overview(archs, out, args.comp_size)
    for arch in archs:
        save_archetype_sheet(arch, out, args.comp_size, args.comp_variants, ncol=4)

    if args.components_only:
        print(f"\nDone. Component sheets in {out}/")
        return

    n = args.grid * args.grid
    presets = default_presets()
    for preset in presets:
        for biome in args.biomes:
            seeds = list(range(args.seed, args.seed + n))
            terrains = [generate(preset, biome, s, args.size, lib) for s in seeds]
            nr, nc = preset.counts[biome]
            title = (
                f"{biome} | {preset.name}  "
                f"(raise={nr}, carve={nc}, amp={preset.comp_amp}, "
                f"rock>{preset.rock_thr}, water<{preset.water_thr})"
            )
            path = out / f"{biome}__{preset.name}.png"
            save_grid(terrains, seeds, title, path, ncol=args.grid)
            print(f"wrote {path}")

    print(f"\nDone. {1 + len(presets) * len(args.biomes)} PNGs in {out}/")


if __name__ == "__main__":
    main()
