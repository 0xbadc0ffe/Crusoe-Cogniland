"""Procedural map generation for zebra_nav.

Two orientations are supported (``generate_zebra_map(orientation=...)``); both
are kept so an agent can be trained / probed on either or a mix:

* ``"diagonal"`` — spawn bottom-left ``(H-1, 0)``, target top-right ``(0, W-1)``;
  walls are diagonal bands perpendicular to the BL→TR path (``t = r-c = C``),
  windows flank the path's ``s = r+c = S_mid``.
* ``"vertical"`` — spawn mid-left ``(H//2, 0)``, target mid-right ``(H//2, W-1)``;
  walls are full-height vertical bands (``c = C``), windows flank the centre row
  ``R_mid``.

Shared mechanic (obsidian walls with two crossing windows)
----------------------------------------------------------
Each *zebra stripe* is a solid **obsidian wall** spanning the map end-to-end so
it cannot be skirted around the border. Each wall has exactly two crossing
**windows** flanking the path centre:

* WATER window (cross by PLACE) — inner cells within ``water_half`` of the wall
  centre are WATER.
* ROCK window (cross by MINE) — inner cells within ``rock_half`` are ROCK.
* a central obsidian divider (half-width ``obsidian_half``) separates them.

The agent meets the wall head-on at the centre and must commit to a window
*locally* (one direction = water, the other = rock). The windows sit
symmetrically about the path, so the only cost difference is the crossing width.

Thick vs thin window
--------------------
One window is thick (``2·thick_half + 1`` cells to cross, default 7) and one
thin (``2·thin_half + 1``, default 3); a fair coin per wall decides whether
water or rock is the thin one. Crossing the thin window needs fewer PLACE / MINE
actions, so the optimal policy reads the cue and threads the thin window. A CUE
tile on the grass just before each wall reveals which window is thin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .tiles import (
    CUE_ROCK_THIN, CUE_WATER_THIN, DIRT, GRASS, OBSIDIAN, ROCK, SAND, TARGET,
    TREE, WATER, is_walkable,
)


@dataclass
class MapRecord:
    terrain: np.ndarray             # (H, W) int8
    spawn: tuple[int, int]
    target: tuple[int, int]
    # per-stripe metadata (ordered along the path; empty for natural maps)
    stripe_centers: list[int]                  # diagonal: t=r-c centre; vertical: column
    stripe_thinner: list[str]                  # "water" or "rock" per stripe
    cue_positions: list[tuple[int, int, int]]  # (r, c, cue_tile_id)
    seed: int
    orientation: str = "diagonal"             # "diagonal" | "vertical" | "natural"
    # natural maps: every cell on the goal wall is a target (touch the wall to win)
    goal_cells: list[tuple[int, int]] = field(default_factory=list)


ORIENTATIONS = ("diagonal", "vertical", "natural")


def _diag_centers(size: int, n_stripes: int, margin: int = 7) -> list[int]:
    """Evenly spaced ``t = r-c`` centres for diagonal walls (BL→TR path)."""
    lo, hi = -(size - 1) + margin, (size - 1) - margin
    return [int(round(x)) for x in np.linspace(hi, lo, n_stripes)]


def _vert_centers(size: int, n_stripes: int, margin: int = 6) -> list[int]:
    """Evenly spaced column centres for vertical walls (mid-L→mid-R path)."""
    lo, hi = margin, (size - 1) - margin
    return [int(round(x)) for x in np.linspace(lo, hi, n_stripes)]


def generate_zebra_map(
    size: int = 32,
    seed: int = 0,
    n_stripes: int = 4,
    thick_half: int = 3,          # thick side: 2*thick_half + 1 = 7 cells to cross
    thin_half: int = 1,           # thin  side: 2*thin_half  + 1 = 3 cells to cross
    obsidian_half: int = 1,       # central obsidian divider half-width (cells)
    window_h: int = 3,            # diagonal-only: size of each crossing window
    orientation: str = "diagonal",
    width: int | None = None,     # map width; height = size (default square)
    water_frac: float = 0.14,     # natural-only: fraction of map that is water (lakes)
    rock_frac: float = 0.14,      # natural-only: fraction of map that is rock (mountains)
    tree_frac: float = 0.03,      # natural-only: fraction of grass turned to impassable tree patches
    goal_half: int | None = None, # natural-only: None ⇒ whole right wall is goal; N ⇒ central door
) -> MapRecord:
    """Build one zebra-stripe / natural map. ``orientation`` selects the layout:

    * ``"diagonal"`` — diagonal walls perpendicular to a BL→TR path; each wall is
      a solid obsidian barrier with a WATER and a ROCK crossing window flanking
      ``s=S_mid`` (``_build_diagonal``). Square only.
    * ``"vertical"`` — full-height vertical walls crossed mid-L→mid-R; each wall
      is WATER (top) + OBSIDIAN divider + ROCK (bottom), filling to the edges so
      it can't be skirted (``_build_vertical``). Supports rectangular maps.
    * ``"natural"`` — open procedural terrain (``_build_natural``): a domain-warped
      fractal heightmap with overlaid lakes (round), mountains, and ridges,
      thresholded to ``water_frac`` water / ``rock_frac`` rock. The agent spawns
      at the centre of the left edge and wins by **touching the opposite (right)
      wall**; there is no obsidian, so every lake/ridge can be crossed
      (bridge / mine) OR walked around — the agent chooses per obstacle.

    Stripe maps: one crossing is thick (``2·thick_half+1``) and one thin
    (``2·thin_half+1``), 50/50 per wall, revealed by a cue. ``height = size``,
    ``width = width or size``. Deterministic given the args.
    """
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {orientation!r}")
    H, W = int(size), int(width) if width is not None else int(size)
    if orientation == "vertical":
        return _build_vertical(H, W, seed, n_stripes, thick_half, thin_half, obsidian_half)
    if orientation == "natural":
        return _build_natural(H, W, seed, water_frac, rock_frac, tree_frac, goal_half=goal_half)
    return _build_diagonal(H, W, seed, n_stripes, thick_half, thin_half, obsidian_half, window_h)


def _build_diagonal(H, W, seed, n_stripes, thick_half, thin_half, obsidian_half, window_h):
    """Diagonal walls perpendicular to the BL→TR path; windows flank ``s=S_mid``.

    A wall is the t-band ``|t-C| ≤ thick_half`` (full anti-diagonal). The WATER
    window sits at ``s`` just below ``S_mid`` and the ROCK window just above,
    separated by a central obsidian divider — the agent meets the wall at the
    centre and picks up (water) or down (rock)."""
    rng = np.random.default_rng(seed)
    terrain = np.full((H, W), GRASS, dtype=np.int8)
    S_mid = (H + W - 2) / 2.0
    rr = np.arange(H, dtype=np.int32)[:, None]
    cc = np.arange(W, dtype=np.int32)[None, :]
    t_grid = rr - cc
    s_grid = rr + cc

    centers = _diag_centers(max(H, W), n_stripes)
    thinner_choice: list[str] = []
    cue_positions: list[tuple[int, int, int]] = []

    for C in centers:
        thinner = "water" if rng.random() < 0.5 else "rock"
        thinner_choice.append(thinner)
        water_half = thin_half if thinner == "water" else thick_half
        rock_half = thin_half if thinner == "rock" else thick_half

        abs_t = np.abs(t_grid - C)
        in_t = abs_t <= thick_half
        water_win = in_t & (s_grid <= S_mid - obsidian_half - 1) \
            & (s_grid >= S_mid - obsidian_half - window_h)
        rock_win = in_t & (s_grid >= S_mid + obsidian_half + 1) \
            & (s_grid <= S_mid + obsidian_half + window_h)
        terrain[in_t] = OBSIDIAN
        terrain[water_win] = GRASS
        terrain[water_win & (abs_t <= water_half)] = WATER
        terrain[rock_win] = GRASS
        terrain[rock_win & (abs_t <= rock_half)] = ROCK

        cue_t = C + thick_half + 2
        cue_r = int(round((S_mid + cue_t) / 2.0))
        cue_c = int(round((S_mid - cue_t) / 2.0))
        if not (0 <= cue_r < H and 0 <= cue_c < W):
            continue
        if not is_walkable(int(terrain[cue_r, cue_c])):
            for d in (1, -1, 2, -2):
                rr2, cc2 = cue_r + d, cue_c + d
                if 0 <= rr2 < H and 0 <= cc2 < W and is_walkable(int(terrain[rr2, cc2])):
                    cue_r, cue_c = rr2, cc2
                    break
            else:
                continue
        cue_tile = CUE_WATER_THIN if thinner == "water" else CUE_ROCK_THIN
        terrain[cue_r, cue_c] = cue_tile
        cue_positions.append((cue_r, cue_c, int(cue_tile)))

    spawn, target = (H - 1, 0), (0, W - 1)
    _clear_bubble(terrain, spawn, target, H, W)
    terrain[target] = TARGET
    return MapRecord(terrain, spawn, target, list(centers), thinner_choice,
                     cue_positions, int(seed), "diagonal")


def _build_vertical(H, W, seed, n_stripes, thick_half, thin_half, obsidian_half, window_h=None):
    """Full-height vertical walls crossed mid-L→mid-R.

    Each wall at column ``C`` is **WATER above the centre row, ROCK below**, with
    a central **obsidian divider** between them — and that divider is the *only*
    obsidian. Water/rock fill all the way to the top / bottom edges, so column
    ``C`` is water→obsidian→rock top-to-bottom and can't be skirted, yet there is
    no obsidian on the top/bottom sides. The WATER column-width is
    ``2·water_half+1`` and ROCK ``2·rock_half+1`` (one thick, one thin); the
    agent meets the divider at the centre and goes up (bridge water) or down
    (mine rock)."""
    rng = np.random.default_rng(seed)
    terrain = np.full((H, W), GRASS, dtype=np.int8)
    R_mid = float(H // 2)           # integer centre row → symmetric divider
    rr = np.arange(H, dtype=np.int32)[:, None]
    cc = np.arange(W, dtype=np.int32)[None, :]

    centers = _vert_centers(W, n_stripes)
    thinner_choice: list[str] = []
    cue_positions: list[tuple[int, int, int]] = []

    for C in centers:
        thinner = "water" if rng.random() < 0.5 else "rock"
        thinner_choice.append(thinner)
        water_half = thin_half if thinner == "water" else thick_half
        rock_half = thin_half if thinner == "rock" else thick_half

        abs_c = np.abs(cc - C)
        # central obsidian divider spans the full wall width (so the grass
        # approach lanes can't slip past it); water above it, rock below — both
        # reaching the map edges (no obsidian on the top/bottom sides).
        divider = (np.abs(rr - R_mid) <= obsidian_half) & (abs_c <= thick_half)
        water_reg = (rr <= R_mid - obsidian_half - 1) & (abs_c <= water_half)
        rock_reg = (rr >= R_mid + obsidian_half + 1) & (abs_c <= rock_half)
        terrain[np.broadcast_to(divider, terrain.shape)] = OBSIDIAN
        terrain[np.broadcast_to(water_reg, terrain.shape)] = WATER
        terrain[np.broadcast_to(rock_reg, terrain.shape)] = ROCK

        cue_r, cue_c = int(round(R_mid)), C - thick_half - 2
        if not (0 <= cue_r < H and 0 <= cue_c < W):
            continue
        if not is_walkable(int(terrain[cue_r, cue_c])):
            for d in (1, -1, 2, -2):
                rr2 = cue_r + d
                if 0 <= rr2 < H and is_walkable(int(terrain[rr2, cue_c])):
                    cue_r = rr2
                    break
            else:
                continue
        cue_tile = CUE_WATER_THIN if thinner == "water" else CUE_ROCK_THIN
        terrain[cue_r, cue_c] = cue_tile
        cue_positions.append((cue_r, cue_c, int(cue_tile)))

    rmid = int(round(R_mid))
    spawn, target = (rmid, 0), (rmid, W - 1)
    _clear_bubble(terrain, spawn, target, H, W)
    terrain[target] = TARGET
    return MapRecord(terrain, spawn, target, list(centers), thinner_choice,
                     cue_positions, int(seed), "vertical")


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
    (walk-around-only). WATER is fringed with cosmetic SAND, ROCK with DIRT
    (both behave like grass). The left and right ``edge_band`` columns are kept
    obstacle-free. Spawn = centre of the left edge; the goal is the **whole right
    wall** (touch it to win). A guard thins tree patches if they ever block the
    goal. Obstacle sizes are deliberately varied (mostly small, a few large) and
    on the small side."""
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
                        and terr[nr, nc] != OBSIDIAN and terr[nr, nc] != TREE:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        return False

    target_trees = int(tree_frac * H * W)
    patches, placed, attempts = [], 0, 0
    while placed < target_trees and attempts < 100:
        attempts += 1
        pr = int(rng.integers(2, H - 2))
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

    return MapRecord(terrain, spawn, target, [], [], [], int(seed),
                     "natural", goal_cells)


def _clear_bubble(terrain, spawn, target, H, W):
    """Clear any water/rock in the 3×3 neighbourhood of spawn/target (not obsidian)."""
    for (sr, sc) in (spawn, target):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = sr + dr, sc + dc
                if 0 <= r < H and 0 <= c < W and terrain[r, c] in (WATER, ROCK):
                    terrain[r, c] = GRASS


# ──────────────────────────────────────────────────────────────────────────


def is_reachable(rec: MapRecord) -> bool:
    """BFS treating in-bounds cells as passable unless inviolable (OBSIDIAN /
    TREE). Confirms that *with* mining/bridging the agent can reach the target —
    used as a contract test (no inviolable wall isolates spawn from target)."""
    from .tiles import OBSIDIAN, TREE
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
                if rec.terrain[nr, nc] != OBSIDIAN and rec.terrain[nr, nc] != TREE:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return False


__all__ = ["MapRecord", "generate_zebra_map", "is_reachable"]
