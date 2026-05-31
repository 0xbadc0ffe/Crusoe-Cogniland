"""Procedural map generation for zebra_nav (natural-only).

Only the ``"natural"`` orientation is supported now: open procedural terrain
with overlaid lakes / mountains / ridges thresholded to ``water_frac`` water and
``rock_frac`` rock, plus a few impassable TREE patches. The agent spawns at the
centre of the left edge and wins by reaching the goal on the right wall (a
central door by default, ``goal_half``). There is no obsidian — every lake/ridge
can be bridged / mined OR walked around; TREE is the only inviolable obstacle.

The retired stripe orientations (``"diagonal"`` / ``"vertical"``) used obsidian
walls + cue tiles; they have been dropped along with the obsidian/cue tile
vocabulary. ``generate_zebra_map(orientation=...)`` raises ``ValueError`` for
anything other than ``"natural"``; the stripe-only kwargs are kept as ignored
no-ops so existing callers don't break.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .tiles import DIRT, GRASS, ROCK, SAND, TARGET, TREE, WATER


@dataclass
class MapRecord:
    terrain: np.ndarray             # (H, W) int8
    spawn: tuple[int, int]
    target: tuple[int, int]
    seed: int
    orientation: str = "natural"
    # natural maps: every cell on the goal wall/door is a target (touch to win)
    goal_cells: list[tuple[int, int]] = field(default_factory=list)


ORIENTATIONS = ("natural",)


def generate_zebra_map(
    size: int = 32,
    seed: int = 0,
    n_stripes: int = 4,           # ignored (retired stripe param)
    thick_half: int = 3,          # ignored (retired stripe param)
    thin_half: int = 1,           # ignored (retired stripe param)
    obsidian_half: int = 1,       # ignored (retired stripe param)
    window_h: int = 3,            # ignored (retired stripe param)
    orientation: str = "natural",
    width: int | None = None,     # map width; height = size (default square)
    water_frac: float = 0.14,     # fraction of map that is water (lakes)
    rock_frac: float = 0.14,      # fraction of map that is rock (mountains)
    tree_frac: float = 0.03,      # fraction of grass turned to impassable tree patches
    goal_half: int | None = None, # None ⇒ whole right wall is goal; N ⇒ central door
) -> MapRecord:
    """Build one natural map.

    ``"natural"`` — open procedural terrain (``_build_natural``): a domain-warped
    fractal heightmap with overlaid lakes (round), mountains, and ridges,
    thresholded to ``water_frac`` water / ``rock_frac`` rock, plus a few
    impassable TREE patches biased toward the top/bottom edges. The agent spawns
    at the centre of the left edge and wins by reaching the goal on the right
    wall; every lake/ridge can be crossed (bridge / mine) OR walked around.

    ``height = size``, ``width = width or size``. Deterministic given the args.
    The stripe-only kwargs (``n_stripes`` / ``thick_half`` / ``thin_half`` /
    ``obsidian_half`` / ``window_h``) are accepted but ignored.
    """
    if orientation != "natural":
        raise ValueError(
            f"orientation must be 'natural' (stripe orientations retired), got {orientation!r}")
    H, W = int(size), int(width) if width is not None else int(size)
    return _build_natural(H, W, seed, water_frac, rock_frac, tree_frac, goal_half=goal_half)


# ───────────────────────── natural (open) terrain ─────────────────────────


def _simplex_field_rect(sim, H, W, sizes, normalize=True):
    """Multi-octave 2D simplex on an ``H×W`` grid (``opensimplex.noise2array``
    returns ``(len(ys), len(xs))`` = ``(rows, cols)``)."""
    if not isinstance(sizes, dict):
        sizes = {sizes: 1.0}
    out = np.zeros((H, W), dtype=np.float64)
    tw = 0.0
    for s, w in sizes.items():
        xs = np.arange(W, dtype=np.float64) / s
        ys = np.arange(H, dtype=np.float64) / s
        out += float(w) * sim.noise2array(xs, ys)
        tw += float(w)
    if normalize and tw > 0.0:
        out /= tw
    return out


def _disk_field(rr, cc, cr, cc0, radius, soft):
    """Flat-topped logistic disk — round lake / rocky knoll."""
    d = np.sqrt((rr - cr) ** 2 + (cc - cc0) ** 2)
    return 1.0 / (1.0 + np.exp((d - radius) / soft))


def _capsule_field(rr, cc, cr, cc0, length, width, angle, wiggle=0.0, period=1.0):
    """Elongated, optionally meandering band — ridge / mountain range / river."""
    dy = rr - cr
    dx = cc - cc0
    ca, sa = math.cos(angle), math.sin(angle)
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    if wiggle:
        v = v - wiggle * np.sin(2.0 * math.pi * u / period)
    over = np.maximum(0.0, np.abs(u) - length / 2.0)
    dist = np.sqrt(over ** 2 + v ** 2)
    return np.exp(-(dist ** 2) / (2.0 * width ** 2))


def _massif_field(rr, cc, cr, cc0, sigma, n_lobes, rng):
    """Irregular round mountain — a few overlapping bumps, peak normalised to 1."""
    f = np.zeros(np.broadcast(rr, cc).shape, dtype=np.float64)
    for _ in range(int(n_lobes)):
        oy, ox = rng.uniform(-sigma, sigma, size=2)
        s = sigma * rng.uniform(0.7, 1.3)
        f += np.exp(-((rr - cr - oy) ** 2 + (cc - cc0 - ox) ** 2) / (2.0 * s ** 2))
    return f / max(f.max(), 1e-9)


def _build_natural(H, W, seed, water_frac, rock_frac, tree_frac=0.06, edge_band=10,
                   goal_half=None):
    """Open procedural terrain. Domain-warped fractal heightmap + overlaid lakes
    (round), mountains, and ridges; thresholded so ~``water_frac`` of cells are
    WATER (low), ~``rock_frac`` are ROCK (high), the rest GRASS. A few small
    **impassable TREE patches** (~``tree_frac``) are sprinkled on the grass
    (walk-around-only), heavily biased toward the top & bottom walls so the
    along-the-wall route to the centre door gets clogged with forest. WATER is
    fringed with cosmetic SAND, ROCK with DIRT (both behave like grass). The
    left and right ``edge_band`` columns are kept obstacle-free. Spawn = centre
    of the left edge; the goal is on the right wall (touch it to win). A guard
    thins tree patches if they ever block the goal. Obstacle sizes are
    deliberately varied (mostly small, a few large) and on the small side."""
    import opensimplex
    from scipy.ndimage import map_coordinates, binary_dilation

    rng = np.random.default_rng(seed * 911 + 17)
    sim_h = opensimplex.OpenSimplex(seed=int(seed) * 7 + 11)
    sim_wr = opensimplex.OpenSimplex(seed=int(seed) * 7 + 12)
    sim_wc = opensimplex.OpenSimplex(seed=int(seed) * 7 + 13)
    # higher base frequency than before ⇒ smaller terrain features; extra octave
    # adds size variation.
    base = max(6.0, max(H, W) / 4.5)
    octaves = {base: 1.0, base / 2: 0.5, base / 4: 0.25, base / 8: 0.125, base / 16: 0.0625}

    height0 = _simplex_field_rect(sim_h, H, W, octaves, normalize=True)
    warp_r = _simplex_field_rect(sim_wr, H, W, {base: 1.0, base / 2: 0.5})
    warp_c = _simplex_field_rect(sim_wc, H, W, {base: 1.0, base / 2: 0.5})
    rr = np.arange(H, dtype=np.float64)[:, None]
    cc = np.arange(W, dtype=np.float64)[None, :]
    RR = np.broadcast_to(rr, (H, W)).astype(np.float64)
    CC = np.broadcast_to(cc, (H, W)).astype(np.float64)
    aw = max(H, W) * 0.10
    hf = map_coordinates(
        height0,
        [np.clip(RR + aw * warp_r, 0, H - 1), np.clip(CC + aw * warp_c, 0, W - 1)],
        order=1, mode="reflect",
    )

    big = float(max(H, W))

    def rand_center(margin=0.14):
        return (rng.uniform(margin, 1 - margin) * H, rng.uniform(margin, 1 - margin) * W)

    def rand_radius(lo, hi):
        # squaring biases toward the small end → mostly small features, a few big
        return big * (lo + (hi - lo) * rng.random() ** 2)

    # overlay stereotypical components — sparser + smaller (mostly small, a rare big)
    for _ in range(int(rng.integers(2, 5))):            # lakes (carve down)
        r, c = rand_center()
        R = rand_radius(0.02, 0.095)
        f = (_massif_field(rr, cc, r, c, R * 0.6, rng.integers(3, 6), rng)
             if rng.random() < 0.5 else _disk_field(rr, cc, r, c, R, max(1.5, R * 0.3)))
        hf = hf - 1.3 * f
    for _ in range(int(rng.integers(2, 5))):            # mountains / knolls (raise)
        r, c = rand_center()
        R = rand_radius(0.02, 0.095)
        f = (_massif_field(rr, cc, r, c, R * 0.6, rng.integers(3, 6), rng)
             if rng.random() < 0.5 else _disk_field(rr, cc, r, c, R, max(1.5, R * 0.3)))
        hf = hf + 1.3 * f
    for _ in range(int(rng.integers(0, 2))):            # a rare thin ridge (raise)
        r, c = rand_center()
        hf = hf + 1.3 * _capsule_field(
            rr, cc, r, c, big * rng.uniform(0.15, 0.32),
            big * rng.uniform(0.03, 0.045), rng.uniform(0, math.pi),
            wiggle=big * 0.06, period=big * 0.45,
        )

    # threshold by quantile so coverage matches water_frac / rock_frac
    terrain = np.full((H, W), GRASS, dtype=np.int8)
    water_level = float(np.quantile(hf, water_frac))
    rock_level = float(np.quantile(hf, 1.0 - rock_frac))
    water_mask = hf < water_level
    rock_mask = hf > rock_level
    terrain[water_mask] = WATER
    terrain[rock_mask] = ROCK

    # cosmetic fringes (walkable look-alikes of grass): a *sparse* 1-cell SAND
    # speckle around water, DIRT around rock — just for visual variety.
    sand_fringe = binary_dilation(water_mask, iterations=1) & (terrain == GRASS)
    terrain[sand_fringe & (rng.random((H, W)) < 0.30)] = SAND
    dirt_fringe = binary_dilation(rock_mask, iterations=1) & (terrain == GRASS)
    terrain[dirt_fringe & (rng.random((H, W)) < 0.25)] = DIRT

    # obstacle-free bands on the left and right edges
    eb = int(edge_band)
    terrain[:, :eb] = GRASS
    terrain[:, W - eb:] = GRASS

    rmid = H // 2
    spawn = (rmid, 0)

    # goal on the right wall. ``goal_half=None`` (default) ⇒ the WHOLE wall is a
    # goal (touch it anywhere to win) — this gives diverse endpoints / multiple
    # paths. A positive ``goal_half`` instead makes only a central door of that
    # half-height the goal (funnels the agent to the centre).
    if goal_half is None or (2 * int(goal_half) + 1) >= H:
        goal_rows = range(H)
    else:
        gh = int(goal_half)
        goal_rows = range(max(0, rmid - gh), min(H, rmid + gh + 1))
    for r in goal_rows:
        terrain[r, W - 1] = TARGET
    goal_cells = [(r, W - 1) for r in goal_rows]
    target = (rmid, W - 1)

    # sprinkle small IMPASSABLE tree patches on grass in the obstacle band only.
    from collections import deque

    def _reachable(terr) -> bool:
        seen = np.zeros((H, W), dtype=bool)
        seen[spawn] = True
        q = deque([spawn])
        while q:
            r, c = q.popleft()
            if (r, c) == target:
                return True
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] \
                        and terr[nr, nc] != TREE:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        return False

    target_trees = int(tree_frac * H * W)
    patches, placed, attempts = [], 0, 0
    while placed < target_trees and attempts < 100:
        attempts += 1
        # STRONG edge bias for tree rows: forests cluster hard against the top &
        # bottom walls and stay sparse in the vertical middle, so wall-hugging to
        # the centre door is blocked by forest. arcsine = U-shaped (dense near 0
        # and 1); blended 85/15 with uniform for a heavy edge bias. Patches may
        # land right against the top/bottom edge rows (pr ∈ [0, H-1]).
        # Heavy edge bias: pick a small distance-from-the-nearest-wall and snap
        # to the top or bottom edge. ``d = u**3`` concentrates the mass near 0
        # (the wall) so most patches sit in the outer ~15% of rows; a few stray
        # deeper. This makes the top/bottom walls a dense forest band while the
        # vertical middle stays largely open.
        u = rng.random()
        d = u ** 3 * 0.5                                 # 0 (at wall) .. 0.5 (centre)
        frac = d if rng.random() < 0.5 else 1.0 - d
        pr = int(round(frac * (H - 1)))
        pr = min(max(pr, 0), H - 1)
        pc = int(rng.integers(eb, W - eb))
        rad = int(rng.integers(1, 3))               # small forests (radius 1–2)
        patch = (((rr - pr) ** 2 + (cc - pc) ** 2 <= rad * rad) & (terrain == GRASS)
                 & (cc >= eb) & (cc < W - eb))      # keep trees out of the edge bands
        n = int(patch.sum())
        if n == 0:
            continue
        terrain[patch] = TREE
        patches.append(patch)
        placed += n
    while patches and not _reachable(terrain):
        terrain[patches.pop()] = GRASS

    return MapRecord(terrain, spawn, target, int(seed), "natural", goal_cells)


# ──────────────────────────────────────────────────────────────────────────


def is_reachable(rec: MapRecord) -> bool:
    """BFS treating in-bounds cells as passable unless inviolable (TREE).
    Confirms that *with* mining/bridging the agent can reach the target — used
    as a contract test (no tree wall isolates spawn from target)."""
    from .tiles import TREE
    H, W = rec.terrain.shape
    sr, sc = rec.spawn
    tr, tc = rec.target
    seen = np.zeros((H, W), dtype=bool)
    seen[sr, sc] = True
    stack = [(sr, sc)]
    while stack:
        r, c = stack.pop()
        if (r, c) == (tr, tc):
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc]:
                if rec.terrain[nr, nc] != TREE:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return False


__all__ = ["MapRecord", "generate_zebra_map", "is_reachable"]
