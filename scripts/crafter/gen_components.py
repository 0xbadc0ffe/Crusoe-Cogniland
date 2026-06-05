"""Compositional terrain components — definition + preview.

A tiny vocabulary of analytic terrain *shapes*, each evaluated directly on
the grid as a field in roughly ``[0, 1]`` (no rasterised stamps), so the
placement transform — **rotation** + **translation** (+ scale) — is just a
change of coordinates baked into the formula. A composed map is then a sum
of a few of these placed at random poses; this script only previews the
*atoms*.

Four atoms (the spec):

* ``ridge``          — finite, gently-meandering thick rock capsule -> ROCK
* ``river``          — thin (5-6 tiles), high-amplitude big-S band -> WATER
* ``lake``           — irregular round water body -> WATER
* ``round_mountain`` — round, slightly lobed rock dome -> ROCK

Each atom gets a touch of high-frequency simplex noise added to its field
*before* thresholding, so the boundary is ragged rather than analytic-clean.

Output: ``components_grid.png`` — atoms on rows, noisy variations on columns,
each rendered in the env tile palette.

Run:
    python scripts/crafter/gen_components.py --out mapgen_preview --variants 6 --size 48
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
    """Smooth 0->1 step centred at x==0; ``soft`` is the edge width."""
    return 1.0 / (1.0 + np.exp(-x / soft))


def _edge_noise(size: int, seed: int, amp: float) -> np.ndarray:
    """High-frequency simplex used only to ragged-ify a threshold boundary."""
    sim = opensimplex.OpenSimplex(seed=seed)
    coords_a = np.arange(size, dtype=np.float64) / 5.0
    coords_b = np.arange(size, dtype=np.float64) / 2.5
    n = sim.noise2array(coords_a, coords_a) + 0.5 * sim.noise2array(coords_b, coords_b)
    return amp * n / 1.5


# ───────────────────────────── component fields ──────────────────────────
#
# Bands (ridge / river) are built between two explicit endpoints so they can be
# *corner-anchored*: in a real map one end is pinned to the top-right (target)
# corner and the band runs toward the lower-left. The preview mirrors that — the
# ridge/river atoms always touch the top-right corner. Blobs (lake / mountain)
# have no such rule and stay centred.


def capsule_field(rr, cc, p0, p1, *, half_width, soft, wiggle, period, phase,
                  waist_frac=None, waist_pos_frac=0.5, waist_sigma_frac=0.13):
    """Meandering capsule between endpoints ``p0`` and ``p1`` (row, col), with
    rounded caps. ``waist_frac`` (if set) pinches the half-width to an hourglass
    at ``waist_pos_frac`` along the spine — the ridge; left ``None`` gives a
    constant width — the river.
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


def round_blob(rr, cc, *, center, radius, soft, lobes, rng):
    """Irregular flat-topped disk: a base logistic disk plus a few offset
    gaussian lobes so the outline is bumpy rather than a perfect circle.

    Used by ``lake`` (carve->water) and ``round_mountain`` (raise->rock).
    """
    d = np.sqrt((rr - center[0]) ** 2 + (cc - center[1]) ** 2)
    f = _logistic(radius - d, soft)
    for _ in range(int(lobes)):
        oy, ox = rng.uniform(-radius * 0.7, radius * 0.7, size=2)
        s = radius * rng.uniform(0.35, 0.6)
        f = f + 0.6 * np.exp(-((rr - center[0] - oy) ** 2 + (cc - center[1] - ox) ** 2) / (2.0 * s**2))
    return f / max(f.max(), 1e-9)


# ───────────────────────────── atoms ─────────────────────────────────────
#
# An "atom" binds a field-maker to per-variation pose/shape sampling and a
# render mode. Variation ``i`` is fully determined by ``(name, i)``.

ATOMS = ["ridge", "river", "lake", "round_mountain"]


def make_atom(name: str, size: int, variant: int) -> np.ndarray:
    """Render one noisy variation of ``name`` as a terrain-tile array."""
    rng = np.random.default_rng(hash((name, variant)) % (2**31))
    rr, cc = _grid(size)
    c = (size - 1) / 2.0
    # one end pinned to the top-right (target) corner; the other in lower-left
    corner = (rng.uniform(0, 2), size - 1 - rng.uniform(0, 2))
    far = (rng.uniform(0.55, 1.0) * size, rng.uniform(0.0, 0.45) * size)

    if name == "ridge":
        field = capsule_field(
            rr, cc, corner, far,
            half_width=size * rng.uniform(0.10, 0.13),   # thick ends
            soft=1.1,
            wiggle=size * rng.uniform(0.05, 0.09),       # gentle bend
            period=size * rng.uniform(0.6, 0.9),
            phase=rng.uniform(0, 2 * math.pi),
            waist_frac=rng.uniform(0.28, 0.42),          # bottleneck ~1/3 of ends
            waist_pos_frac=rng.uniform(0.30, 0.70),      # anywhere along the spine
            waist_sigma_frac=rng.uniform(0.10, 0.16),
        )
        return _threshold_render(field, size, rng, mode="rock")

    if name == "river":
        field = capsule_field(
            rr, cc, corner, far,
            half_width=size * rng.uniform(0.05, 0.065),  # ~5-6 tiles thick
            soft=0.9,
            wiggle=size * rng.uniform(0.12, 0.20),       # high amplitude (big S)
            period=size * rng.uniform(0.5, 0.8),
            phase=rng.uniform(0, 2 * math.pi),
        )
        return _threshold_render(field, size, rng, mode="water")

    if name == "lake":
        field = round_blob(
            rr, cc,
            center=(c + rng.uniform(-4, 4), c + rng.uniform(-4, 4)),
            radius=size * rng.uniform(0.20, 0.30),
            soft=1.6, lobes=rng.integers(3, 6), rng=rng,
        )
        return _threshold_render(field, size, rng, mode="water")

    if name == "round_mountain":
        field = round_blob(
            rr, cc,
            center=(c + rng.uniform(-3, 3), c + rng.uniform(-3, 3)),
            radius=size * rng.uniform(0.18, 0.26),
            soft=1.4, lobes=rng.integers(2, 4), rng=rng,
        )
        return _threshold_render(field, size, rng, mode="rock")

    raise ValueError(f"unknown atom: {name!r}")


def _threshold_render(field: np.ndarray, size: int, rng, *, mode: str) -> np.ndarray:
    """Add edge noise, threshold at 0.5, and paint tiles with a thin skirt
    (SAND beach for water, DIRT scree for rock)."""
    f = field + _edge_noise(size, int(rng.integers(2**31)), amp=0.14)
    mask = f > 0.5
    terrain = np.full((size, size), GRASS, dtype=np.int8)
    if not mask.any():
        return terrain
    if mode == "water":
        terrain[mask] = WATER
        d = distance_transform_edt(~mask)
        terrain[(d <= 2) & ~mask] = SAND
    else:  # rock
        terrain[mask] = ROCK
        d = distance_transform_edt(~mask)
        terrain[(d <= 1) & ~mask] = DIRT
    return terrain


# ───────────────────────────── rendering ─────────────────────────────────


def render_rgb(terrain: np.ndarray) -> np.ndarray:
    return TILE_COLORS[terrain]


def save_grid(out: Path, size: int, variants: int) -> Path:
    nrow, ncol = len(ATOMS), variants
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.7, nrow * 1.7))
    axes = np.atleast_2d(axes)
    for i, name in enumerate(ATOMS):
        for j in range(ncol):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(render_rgb(make_atom(name, size, j)), interpolation="nearest")
            if name in ("ridge", "river"):   # one end is pinned to this corner
                ax.plot(size - 1, 0, "*", ms=8, mfc="red", mec="black", mew=0.6)
            if i == 0:
                ax.set_title(f"var {j}", fontsize=8, pad=3)
            if j == 0:
                ax.set_ylabel(name, fontsize=10, rotation=90, va="center")
    fig.suptitle("Terrain atoms (rows) x noisy variations (cols)   "
                 "[★ = target corner the ridge/river anchor to]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / "components_grid.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="mapgen_preview", help="output folder")
    ap.add_argument("--size", type=int, default=48, help="tiles per component canvas")
    ap.add_argument("--variants", type=int, default=6, help="columns (variations per atom)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = save_grid(out, args.size, args.variants)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
