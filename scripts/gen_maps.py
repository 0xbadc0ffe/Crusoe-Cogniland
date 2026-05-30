"""Compositional map generation — preview.

Builds full maps in three layers:

1. **Grass base.**
2. **Component layer** — analytic atoms (ridge, river, lake, round_mountain)
   placed under per-biome constraints and thresholded to ROCK / WATER masks.
3. **Texture layer** — independent perlin (simplex) fields sprinkle forest
   (TREE) plus sparse dirt (DIRT) / sand (SAND) onto the open grass.

The agent travels from a **spawn** (bottom-left) to a **target** (top-right).

Biome recipes (the spec):

* ``rocky``    — a **ridge** + a **lake**. The ridge has **one end pinned to
                 the top-right corner** and runs across toward the lower-left;
                 the ridge wins overlaps. Extra DIRT collar around the ridge.
* ``river``    — a **river** + a **round_mountain**. The river has **one end
                 pinned to the top-right corner**; the river wins overlaps.
                 Extra SAND banded + sprinkled along the river.
* ``balanced`` — split the map along the anti-diagonal (top-left half vs
                 bottom-right half). One half gets a WATER component
                 {lake, river}, the other a ROCK component {ridge,
                 round_mountain}. Extra forest fills the middle grass corridor.

In every biome the target (and spawn) keep a small grass clearance: rock/water
may pass *nearby* the target but never cover it.

Output: ``maps_grid.png`` — biomes on rows, seeds on columns.

Run:
    python scripts/gen_maps.py --out mapgen_preview --variants 6 --size 64
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import opensimplex
from scipy.ndimage import distance_transform_edt

# tile ids + palette — mirror cogniland.nav.tiles / crafter_in_cogniland.constants
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


# ───────────────────────────── primitives ────────────────────────────────


def _grid(size: int):
    rr = np.arange(size, dtype=np.float64)[:, None]
    cc = np.arange(size, dtype=np.float64)[None, :]
    return rr, cc


def _logistic(x: np.ndarray, soft: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x / soft))


def _simplex_field(seed: int, size: int, scales) -> np.ndarray:
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


def _edge_noise(size: int, seed: int, amp: float) -> np.ndarray:
    return amp * _simplex_field(seed, size, {5.0: 1.0, 2.5: 0.5})


def capsule_field(rr, cc, p0, p1, *, half_width, soft, wiggle, period, phase,
                  waist_frac=None, waist_pos_frac=0.5, waist_sigma_frac=0.13):
    """Meandering capsule between endpoints ``p0`` and ``p1`` (row, col), with
    rounded caps. ``waist_frac`` (if set) pinches the half-width to an hourglass
    at ``waist_pos_frac`` along the spine — used for the ridge; left ``None``
    gives a constant width — used for the river.
    """
    r0, c0 = p0
    r1, c1 = p1
    dr, dc = r1 - r0, c1 - c0
    L = math.hypot(dr, dc) + 1e-9
    ur, uc = dr / L, dc / L                 # unit along the spine
    u = (rr - r0) * ur + (cc - c0) * uc     # projection along
    v = (rr - r0) * (-uc) + (cc - c0) * ur  # perpendicular
    spine = wiggle * np.sin(2.0 * math.pi * u / period + phase)
    if waist_frac is not None:
        wp = waist_pos_frac * L
        sig = max(1.0, waist_sigma_frac * L)
        notch = (1.0 - waist_frac) * np.exp(-((u - wp) ** 2) / (2.0 * sig**2))
        ripple = 0.06 * np.sin(2.0 * math.pi * u / (L / 7.0) + phase)
        hw = half_width * np.clip(1.0 - notch + ripple, waist_frac * 0.8, 1.1)
    else:
        hw = half_width
    over = np.maximum(np.maximum(0.0, -u), u - L)   # 0 inside [0, L], else caps
    dist = np.sqrt(over**2 + (v - spine) ** 2)
    return _logistic(hw - dist, soft)


def blob_field(rr, cc, *, center, radius, soft, lobes, rng):
    """Irregular flat-topped disk — lake / round_mountain."""
    d = np.sqrt((rr - center[0]) ** 2 + (cc - center[1]) ** 2)
    f = _logistic(radius - d, soft)
    for _ in range(int(lobes)):
        oy, ox = rng.uniform(-radius * 0.7, radius * 0.7, size=2)
        s = radius * rng.uniform(0.35, 0.6)
        f = f + 0.6 * np.exp(-((rr - center[0] - oy) ** 2 + (cc - center[1] - ox) ** 2) / (2.0 * s**2))
    return f / max(f.max(), 1e-9)


def _mask(field: np.ndarray, size: int, rng) -> np.ndarray:
    return (field + _edge_noise(size, int(rng.integers(2**31)), amp=0.13)) > 0.5


# ───────────────────────────── component builders ────────────────────────


def _band_field(rr, cc, size, rng, p0, p1, *, kind):
    """A river (constant thin width, meandering) or a ridge (hourglass)."""
    if kind == "river":
        return capsule_field(
            rr, cc, p0, p1,
            half_width=size * rng.uniform(0.045, 0.06), soft=0.9,
            wiggle=size * rng.uniform(0.10, 0.18), period=size * rng.uniform(0.5, 0.8),
            phase=rng.uniform(0, 2 * math.pi),
        )
    return capsule_field(   # ridge
        rr, cc, p0, p1,
        half_width=size * rng.uniform(0.06, 0.085), soft=1.1,
        wiggle=size * rng.uniform(0.05, 0.09), period=size * rng.uniform(0.6, 0.9),
        phase=rng.uniform(0, 2 * math.pi),
        waist_frac=rng.uniform(0.28, 0.42), waist_pos_frac=rng.uniform(0.30, 0.70),
        waist_sigma_frac=rng.uniform(0.10, 0.16),
    )


def _blob(rr, cc, size, rng, center, *, kind):
    radius = size * (rng.uniform(0.12, 0.18) if kind == "lake" else rng.uniform(0.10, 0.15))
    lobes = rng.integers(3, 6) if kind == "lake" else rng.integers(2, 4)
    soft = 1.6 if kind == "lake" else 1.4
    return blob_field(rr, cc, center=center, radius=radius, soft=soft, lobes=lobes, rng=rng)


def _place_blob_nonoverlap(rr, cc, size, rng, kind, avoid_mask, *,
                           lo=0.22, hi=0.78, gap=2, tries=60):
    """Place a lake / mountain blob *after* the band, rejecting any pose that
    touches the band (within a ``gap``-cell halo). Falls back to clipping the
    band out if no clear spot is found."""
    blocked = distance_transform_edt(~avoid_mask) <= gap if avoid_mask.any() else avoid_mask
    last = None
    for _ in range(tries):
        center = (rng.uniform(lo, hi) * size, rng.uniform(lo, hi) * size)
        m = _mask(_blob(rr, cc, size, rng, center, kind=kind), size, rng)
        last = m
        if not (m & blocked).any():
            return m
    return last & ~avoid_mask


def _rand_point_in_half(half: str, size: int, rng, margin: float = 0.16):
    """A point biased into the top-left (``TL``: r+c < size) or bottom-right
    (``BR``: r+c > size) half, kept off the anti-diagonal corridor."""
    while True:
        r = rng.uniform(margin, 1 - margin) * size
        c = rng.uniform(margin, 1 - margin) * size
        if half == "TL" and (r + c) < size * 0.86:
            return (r, c)
        if half == "BR" and (r + c) > size * 1.14:
            return (r, c)


def _rand_point_near_diag(half: str, size: int, rng, lo: float = 0.05, hi: float = 0.16):
    """A point hugging the anti-diagonal, offset just into ``half`` — pulls
    balanced lakes / mountains toward the centre corridor."""
    s = rng.uniform(0.20, 0.80) * size
    off = rng.uniform(lo, hi) * size
    sign = 1.0 if half == "BR" else -1.0
    m = 0.10 * size
    r = float(np.clip(s + sign * off, m, size - 1 - m))
    c = float(np.clip((size - s) + sign * off, m, size - 1 - m))
    return (r, c)


# ───────────────────────────── biome recipes ─────────────────────────────


def build_map(map_type: str, size: int, seed: int):
    rng = np.random.default_rng(seed)
    rr, cc = _grid(size)
    zone = max(4, size // 4)
    spawn = (int(rng.integers(size - zone, size)), int(rng.integers(0, zone)))
    target = (int(rng.integers(0, zone)), int(rng.integers(size - zone, size)))
    # one end pinned to the top-right corner (small jitter)
    corner = (rng.uniform(0, 3), size - 1 - rng.uniform(0, 3))

    rock_mask = np.zeros((size, size), dtype=bool)
    water_mask = np.zeros((size, size), dtype=bool)
    forest_corridor = None

    if map_type == "rocky":
        far = (rng.uniform(0.50, 0.92) * size, rng.uniform(0.0, 0.42) * size)
        rock_mask = _mask(_band_field(rr, cc, size, rng, corner, far, kind="ridge"), size, rng)
        # lake placed after the ridge, not overlapping it
        water_mask = _place_blob_nonoverlap(rr, cc, size, rng, "lake", rock_mask)
        dirt_w, sand_w, sand_extra = 3, 2, 2

    elif map_type == "river":
        far = (rng.uniform(0.50, 0.92) * size, rng.uniform(0.0, 0.42) * size)
        water_mask = _mask(_band_field(rr, cc, size, rng, corner, far, kind="river"), size, rng)
        # mountain placed after the river, not overlapping it
        rock_mask = _place_blob_nonoverlap(rr, cc, size, rng, "round_mountain", water_mask)
        dirt_w, sand_w, sand_extra = 0, 3, 4

    elif map_type == "balanced":
        water_half = str(rng.choice(["TL", "BR"]))
        rock_half = "BR" if water_half == "TL" else "TL"
        water_name = str(rng.choice(["lake", "river"]))
        rock_name = str(rng.choice(["ridge", "round_mountain"]))

        def place(name, half):
            if name == "ridge":
                # one end pinned to the target (top-right) corner, the other
                # reaching deep into the assigned half (longest sampled span)
                best, best_d = corner, -1.0
                for _ in range(40):
                    p1 = _rand_point_in_half(half, size, rng)
                    d = math.hypot(p1[0] - corner[0], p1[1] - corner[1])
                    if d > best_d:
                        best, best_d = p1, d
                    if d > 0.60 * size:
                        break
                return _mask(_band_field(rr, cc, size, rng, corner, best, kind="ridge"), size, rng)
            if name == "river":
                # keep the longest of several candidate spans -> longer bands
                p0 = _rand_point_in_half(half, size, rng)
                best, best_d = p0, -1.0
                for _ in range(40):
                    p1 = _rand_point_in_half(half, size, rng)
                    d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                    if d > best_d:
                        best, best_d = p1, d
                    if d > 0.60 * size:
                        break
                return _mask(_band_field(rr, cc, size, rng, p0, best, kind="river"), size, rng)
            # lake / mountain: hug the centre diagonal
            ctr = _rand_point_near_diag(half, size, rng)
            return _mask(_blob(rr, cc, size, rng, ctr, kind=name), size, rng)

        water_mask = place(water_name, water_half)
        rock_mask = place(rock_name, rock_half)
        water_mask &= ~rock_mask
        # denser forest along the middle (anti-diagonal) grass corridor
        forest_corridor = np.exp(-((rr + cc - size) ** 2) / (2.0 * (size * 0.12) ** 2))
        dirt_w, sand_w, sand_extra = 2, 2, 2
    else:
        raise ValueError(f"unknown map_type: {map_type!r}")

    # ── never cover the target (or spawn): clear a small grass bubble ──
    clr = size * 0.065
    for (pr, pc) in (target, spawn):
        bubble = np.sqrt((rr - pr) ** 2 + (cc - pc) ** 2) < clr
        rock_mask &= ~bubble
        water_mask &= ~bubble

    feature = rock_mask | water_mask
    open_land = ~feature
    terrain = np.full((size, size), GRASS, dtype=np.int8)

    # ── texture layer: forest (clustered, organic) + sparse dirt / sand ──
    # ``cluster`` is a low-freq simplex so trees form blobs rather than uniform
    # speckle. Balanced maps concentrate clusters in the diagonal corridor and
    # scatter a few more around the whole map; rocky/river get a light sprinkle.
    cluster = _simplex_field(seed * 31 + 9, size, {size / 5.0: 1.0, size / 12.0: 0.5})
    forest = open_land & (cluster > 0.60) & (rng.random((size, size)) < 0.55)  # natural patches
    if forest_corridor is not None:
        forest |= open_land & (cluster > 0.42) & (forest_corridor > 0.55)       # clustered corridor
    else:
        forest |= open_land & (cluster > 0.52) & (rng.random((size, size)) < 0.22)  # a touch more
    terrain[forest] = TREE
    dry = _simplex_field(seed * 19 + 5, size, {size / 10.0: 1.0})
    terrain[open_land & (terrain == GRASS) & (dry > 0.72)] = DIRT
    sandn = _simplex_field(seed * 23 + 7, size, {size / 11.0: 1.0})
    terrain[open_land & (terrain == GRASS) & (sandn > 0.72)] = SAND

    # ── collars: dirt around rock (ridge), sand around ALL water (beaches) ──
    # the band hugging the feature stays solid; the outer sprinkle is kept light.
    if dirt_w > 0 and rock_mask.any():
        d = distance_transform_edt(~rock_mask)
        terrain[(d >= 1) & (d <= dirt_w) & open_land] = DIRT
        terrain[(d > dirt_w) & (d <= dirt_w + 2) & open_land
                & (rng.random((size, size)) < 0.18)] = DIRT
    if sand_w > 0 and water_mask.any():
        d = distance_transform_edt(~water_mask)
        terrain[(d >= 1) & (d <= sand_w) & open_land] = SAND
        terrain[(d > sand_w) & (d <= sand_w + sand_extra) & open_land
                & (rng.random((size, size)) < 0.16)] = SAND

    # ── components on top, then the target marker ──
    terrain[water_mask] = WATER
    terrain[rock_mask] = ROCK
    terrain[target] = TARGET
    return terrain, spawn, target


# ───────────────────────────── rendering ─────────────────────────────────


def render_rgb(terrain: np.ndarray) -> np.ndarray:
    return TILE_COLORS[terrain]


def save_grid(out: Path, size: int, variants: int) -> Path:
    biomes = ["rocky", "river", "balanced"]
    nrow, ncol = len(biomes), variants
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.8, nrow * 1.8))
    axes = np.atleast_2d(axes)
    for i, biome in enumerate(biomes):
        for j in range(ncol):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            terrain, spawn, target = build_map(biome, size, seed=100 * i + j)
            ax.imshow(render_rgb(terrain), interpolation="nearest")
            ax.plot(spawn[1], spawn[0], "o", ms=5, mfc="white", mec="black", mew=0.8)
            ax.plot(target[1], target[0], "*", ms=8, mfc="red", mec="black", mew=0.6)
            if i == 0:
                ax.set_title(f"seed {j}", fontsize=8, pad=3)
            if j == 0:
                ax.set_ylabel(biome, fontsize=11, rotation=90, va="center")
    fig.suptitle("Composed maps — biomes (rows) x seeds (cols)   "
                 "[o spawn  ★ target]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / "maps_grid.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="mapgen_preview")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--variants", type=int, default=6, help="columns (seeds per biome)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = save_grid(out, args.size, args.variants)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
