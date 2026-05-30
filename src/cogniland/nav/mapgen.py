"""Procedural map generation for Cogniland navigation.

Two generators, selected by ``generate_map(..., generator=...)``:

* ``"components"`` (default) — composes a map from structured analytic
  shapes (round/elongated lakes & rivers, big mountains & ranges) via
  translation / rotation / scale, plus a ragged simplex coastline octave
  and forest/sand artifacts. Every map is built to offer **two roughly
  equivalent routes**; the biome only flavours which terrain dominates:

    - ``lake``  → a WATER barrier you can raft across OR walk around
                  for ~equal cost (harness is worse).
    - ``rocky`` → a ROCK barrier: harness-across ≈ walk-around (raft worse).
    - ``balanced`` → one of: *split* (water on one side / rock on the
      other → raft ≈ harness), *cross_around* (either material), or
      *mild* (scattered features, no-skill simply best).

  See ``_plan_map`` / ``_build_material_components`` / ``_validate_plan``.

* ``"simplex"`` — the legacy Crafter-style noise terrain (spawn bubbles +
  low-frequency water/mountain fields) with the *strict* single-best-skill
  ordering ``raft < no_skill < harness`` (or its rocky/balanced
  counterparts). Kept as a fallback. The two simplex map families differ
  only in how the two non-grass materials are
mapped to in-game tile ids:

* **Lake** maps:  ``mountain → ROCK``,  ``water → WATER`` → raft correct
* **Rocky** maps: ``mountain → WATER``, ``water → ROCK``  → harness correct

(The user spec: "the other type of maps is the same but with rocks and
water flipped".)

For navigation correctness we add a small Gaussian bias along the
spawn-target line on both the water and mountain fields. Without this,
Crafter terrain only places the big lake / mountain on the agent's path on
~10% of seeds, so rejection sampling becomes prohibitively slow. The bias
is small enough that the maps still look organic, but it concentrates the
useful barrier where it matters for the belief/commitment problem.

Every generated map is validated by running a Dijkstra oracle over the
exact per-(object, terrain) cost table from `skills.py`. Maps that fail
``lake → raft < no_skill < harness`` (or its rocky counterpart) plus a
5% margin are rejected and re-sampled up to ``max_retries`` times.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import opensimplex

from . import skills as sk
from .tiles import DIRT, GRASS, LAVA, ROCK, SAND, TARGET, TREE, WATER

MapType = Literal["lake", "rocky", "balanced", "random"]


class MapGenError(RuntimeError):
    pass


class _Retry(Exception):
    pass


# ----------------------------------------------------------- Dijkstra oracle


def _dijkstra_from(
    goal: tuple[int, int], cost_arr: np.ndarray
) -> np.ndarray:
    """Single-source shortest cost-to-go array from every cell to ``goal``.

    ``cost_arr[r, c]`` is the cost of stepping *onto* cell ``(r, c)``;
    ``inf`` for blocked cells. The cost of standing on ``goal`` is 0.
    Returns float64 ``[H, W]`` with ``inf`` for unreachable cells.
    """
    H, W = cost_arr.shape
    gr, gc = int(goal[0]), int(goal[1])
    dist = np.full((H, W), math.inf, dtype=np.float64)
    if not math.isfinite(cost_arr[gr, gc]):
        return dist
    dist[gr, gc] = 0.0
    pq: list[tuple[float, int, int]] = [(0.0, gr, gc)]
    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                step = float(cost_arr[nr, nc])
                if not math.isfinite(step):
                    continue
                nd = d + step
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(pq, (nd, nr, nc))
    return dist


def shortest_path_cost(
    terrain: np.ndarray, start: tuple[int, int], goal: tuple[int, int], obj: int
) -> float:
    """Expected *attempt-count* shortest path from ``start`` to ``goal`` under
    ``obj`` (slip-adjusted). ``inf`` if unreachable. Cost of entering
    ``start`` is omitted."""
    cost_arr = sk.expected_attempts_grid(obj, terrain)
    dist = _dijkstra_from(goal, cost_arr)
    sr, sc = int(start[0]), int(start[1])
    return float(dist[sr, sc])


def cost_to_go_unit(
    terrain: np.ndarray, goal: tuple[int, int], obj: int
) -> np.ndarray:
    """Per-cell cost-to-go in *cells* (unit cost) under ``obj``. Used by the
    env's PBRS shaping so the shaping reward measures geometric progress
    toward the target irrespective of slip."""
    cost_arr = sk.unit_cost_grid(obj, terrain)
    return _dijkstra_from(goal, cost_arr)


# ------------------------------------------------------ Crafter noise helper


def _simplex_field(
    sim: opensimplex.OpenSimplex,
    size: int,
    sizes: int | dict[int, float],
    normalize: bool = True,
) -> np.ndarray:
    """Multi-octave 2D simplex sampled on the ``size x size`` grid.

    Each call to ``noise2array`` is ~2× faster than ``noise3array``; we
    achieve per-channel decorrelation by handing each field its own
    seeded :class:`OpenSimplex` instance instead of reusing one with
    different z values.
    """
    if not isinstance(sizes, dict):
        sizes = {sizes: 1.0}
    out = np.zeros((size, size), dtype=np.float64)
    total_w = 0.0
    for s, w in sizes.items():
        xs = np.arange(size, dtype=np.float64) / s
        ys = np.arange(size, dtype=np.float64) / s
        # noise2array returns shape (len(y), len(x)) → (rows, cols)
        n = sim.noise2array(xs, ys)
        out += float(w) * n
        total_w += float(w)
    if normalize and total_w > 0.0:
        out /= total_w
    return out


# ---------------------------------------------------------------- generation


@dataclass
class MapRecord:
    terrain: np.ndarray
    spawn: np.ndarray
    target: np.ndarray
    map_type: str
    correct_object: int
    # expected-attempt costs (slip-adjusted) — for analysis / dataset metadata
    no_skill_cost: float
    raft_cost: float
    harness_cost: float
    constraints_passed: bool
    seed: int
    # geometric per-skill cost-to-go arrays (cells) — used by env for PBRS shaping.
    # Not stored when serialised; recomputed by env from terrain+target.
    ctg_none: np.ndarray | None = None
    ctg_raft: np.ndarray | None = None
    ctg_harness: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "terrain": self.terrain,
            "spawn": self.spawn,
            "target": self.target,
            "map_type": self.map_type,
            "correct_object": self.correct_object,
            "no_skill_cost": self.no_skill_cost,
            "raft_cost": self.raft_cost,
            "harness_cost": self.harness_cost,
            "constraints_passed": self.constraints_passed,
            "seed": self.seed,
        }


def _diagonal_bias(
    size: int, spawn: tuple[int, int], target: tuple[int, int], sigma: float
) -> np.ndarray:
    """Gaussian peak along the line from ``spawn`` to ``target``.

    Returns a float64 ``[size, size]`` array in roughly ``[0, 1]`` where
    cells on the spawn-target line have value 1.0 and falloff is Gaussian
    perpendicular to the line.
    """
    rr = np.arange(size, dtype=np.float64)[:, None]
    cc = np.arange(size, dtype=np.float64)[None, :]
    sr, sc = spawn
    tr, tc = target
    drow = tr - sr
    dcol = tc - sc
    length = math.hypot(drow, dcol)
    if length < 1e-6:
        return np.zeros((size, size), dtype=np.float64)
    perp = np.abs((rr - sr) * dcol - (cc - sc) * drow) / length
    return np.exp(-(perp**2) / (2.0 * sigma * sigma))


# ───────────────────────── structured components ─────────────────────────
#
# A small vocabulary of analytic terrain *shapes* evaluated directly on the
# global grid (no rasterised stamps → no scipy dependency; rotation is
# closed-form). Each returns a field in roughly ``[0, 1]`` that is added to a
# heightmap with a chosen sign: ``+`` raises toward ROCK, ``−`` carves toward
# WATER. Maps are then composed by a couple of simple rules (see
# ``_build_material_components``): one biome-typed *barrier* straddling the
# spawn→target line, a few *decorative* features off the path, and light
# simplex artifacts (forest / sand). The same shapes recur across every map;
# only their placement (translate / rotate / scale) varies by seed.


def _global_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    rr = np.arange(size, dtype=np.float64)[:, None]
    cc = np.arange(size, dtype=np.float64)[None, :]
    return rr, cc


def _disk_field(rr, cc, cr, cc0, radius, soft):
    """Flat-topped logistic disk — round lake / rocky knoll."""
    d = np.sqrt((rr - cr) ** 2 + (cc - cc0) ** 2)
    return 1.0 / (1.0 + np.exp((d - radius) / soft))


def _capsule_field(rr, cc, cr, cc0, length, width, angle, wiggle=0.0, period=1.0):
    """Elongated, optionally meandering band — ridge / mountain range / river.

    ``angle`` is the orientation of the spine; ``length`` its extent; ``width``
    the gaussian falloff perpendicular to it.
    """
    dy = rr - cr
    dx = cc - cc0
    ca, sa = math.cos(angle), math.sin(angle)
    u = dx * ca + dy * sa          # along the spine
    v = -dx * sa + dy * ca         # perpendicular
    if wiggle:
        v = v - wiggle * np.sin(2.0 * math.pi * u / period)
    over = np.maximum(0.0, np.abs(u) - length / 2.0)
    dist = np.sqrt(over**2 + v**2)
    return np.exp(-(dist**2) / (2.0 * width**2))


def _massif_field(rr, cc, cr, cc0, sigma, n_lobes, rng):
    """Irregular round mountain — a few overlapping bumps, peak normalised 1."""
    H = rr.shape[0]
    W = cc.shape[1]
    f = np.zeros((H, W), dtype=np.float64)
    for _ in range(int(n_lobes)):
        oy, ox = rng.uniform(-sigma, sigma, size=2)
        s = sigma * rng.uniform(0.7, 1.3)
        f += np.exp(-((rr - cr - oy) ** 2 + (cc - cc0 - ox) ** 2) / (2.0 * s**2))
    return f / max(f.max(), 1e-9)


def _segment_distance(rr, cc, p0, p1):
    """Per-cell Euclidean distance to the segment ``p0→p1`` (row, col)."""
    r0, c0 = p0
    r1, c1 = p1
    dr = r1 - r0
    dc = c1 - c0
    L2 = dr * dr + dc * dc + 1e-9
    t = np.clip(((rr - r0) * dr + (cc - c0) * dc) / L2, 0.0, 1.0)
    pr = r0 + t * dr
    pc = c0 + t * dc
    return np.sqrt((rr - pr) ** 2 + (cc - pc) ** 2)


def _build_material_components(
    size: int,
    biome: Literal["lake", "rocky", "balanced"],
    seed: int,
    spawn: tuple[int, int],
    target: tuple[int, int],
) -> np.ndarray:
    """Natural-looking terrain: a domain-warped fractal heightmap thresholded
    into water / grass / rock, with overlaid stereotypical components (lakes,
    rivers, mountains, ranges), beaches, and forest.

    There is **no** cost-inequality validation — the env is a POMDP, so we only
    care that the maps look natural and that the three biomes differ:

    * ``lake``     — water-dominant (low water level), lakes + rivers, few rocks.
    * ``rocky``    — rock-dominant (low rock level), mountains + ranges, little water.
    * ``balanced`` — moderate thresholds, a mix of lakes / rivers / mountains.
    """
    from scipy.ndimage import distance_transform_edt, map_coordinates

    rng = np.random.default_rng(seed * 911 + 17)
    rr, cc = _global_grid(size)
    sr, sc = spawn
    tr, tc = target

    # ── domain-warped fractal Brownian heightmap → organic rolling terrain ──
    sim_h = opensimplex.OpenSimplex(seed=seed * 7 + 11)
    sim_wr = opensimplex.OpenSimplex(seed=seed * 7 + 12)
    sim_wc = opensimplex.OpenSimplex(seed=seed * 7 + 13)
    base = max(10.0, size / 3.0)
    octaves = {base: 1.0, base / 2: 0.5, base / 4: 0.25,
               base / 8: 0.125, base / 16: 0.0625}
    height0 = _simplex_field(sim_h, size, octaves, normalize=True)
    warp_r = _simplex_field(sim_wr, size, {base: 1.0, base / 2: 0.5})
    warp_c = _simplex_field(sim_wc, size, {base: 1.0, base / 2: 0.5})
    aw = size * 0.10
    RR = np.broadcast_to(rr, (size, size)).astype(np.float64)
    CC = np.broadcast_to(cc, (size, size)).astype(np.float64)
    H = map_coordinates(
        height0,
        [np.clip(RR + aw * warp_r, 0, size - 1),
         np.clip(CC + aw * warp_c, 0, size - 1)],
        order=1, mode="reflect",
    )

    # ── biome flavour ──
    if biome == "lake":
        water_level, rock_level = -0.08, 0.52
        n_lake, n_river, n_mtn, n_range = (int(rng.integers(1, 3)),
                                           int(rng.integers(1, 3)), 0,
                                           int(rng.integers(0, 2)))
        forest_thr = 0.18
    elif biome == "rocky":
        water_level, rock_level = -0.52, 0.08
        n_lake, n_river, n_mtn, n_range = (int(rng.integers(0, 2)), 0,
                                           int(rng.integers(1, 3)),
                                           int(rng.integers(1, 3)))
        forest_thr = 0.32
    else:  # balanced
        water_level, rock_level = -0.30, 0.32
        n_lake, n_river, n_mtn, n_range = (int(rng.integers(0, 2)),
                                           int(rng.integers(0, 2)),
                                           int(rng.integers(0, 2)),
                                           int(rng.integers(0, 2)))
        forest_thr = 0.26

    def rand_center(margin: float = 0.14) -> tuple[float, float]:
        return (rng.uniform(margin, 1 - margin) * size,
                rng.uniform(margin, 1 - margin) * size)

    # ── overlay stereotypical components onto the heightmap ──
    for _ in range(n_lake):                                   # round-ish lakes
        r, c = rand_center()
        R = size * rng.uniform(0.08, 0.16)
        f = (_massif_field(rr, cc, r, c, R * 0.6, rng.integers(3, 6), rng)
             if rng.random() < 0.6 else _disk_field(rr, cc, r, c, R, max(2.0, R * 0.3)))
        H = H - 1.3 * f
    for _ in range(n_river):                                  # meandering rivers
        r, c = rand_center(0.3)
        H = H - 1.4 * _capsule_field(
            rr, cc, r, c, size * 1.6, size * rng.uniform(0.025, 0.045),
            rng.uniform(0, math.pi), wiggle=size * rng.uniform(0.15, 0.30),
            period=size * rng.uniform(0.35, 0.60),
        )
    for _ in range(n_mtn):                                    # big mountains
        r, c = rand_center()
        R = size * rng.uniform(0.09, 0.18)
        H = H + 1.3 * _massif_field(rr, cc, r, c, R * 0.6, rng.integers(3, 6), rng)
    for _ in range(n_range):                                  # mountain ranges
        r, c = rand_center()
        H = H + 1.3 * _capsule_field(
            rr, cc, r, c, size * rng.uniform(0.4, 0.8),
            size * rng.uniform(0.05, 0.08), rng.uniform(0, math.pi),
            wiggle=size * 0.08, period=size * 0.5,
        )

    # ── threshold to terrain ──
    material = np.full((size, size), GRASS, dtype=np.int8)
    rock_mask = H > rock_level
    water_mask = H < water_level
    material[rock_mask] = ROCK
    material[water_mask] = WATER
    if water_mask.any():                                      # beaches around water
        d = distance_transform_edt(~water_mask)
        material[(d <= 2) & ~water_mask & ~rock_mask] = SAND

    # ── moisture / dryness artifacts on grass ──
    sim_m = opensimplex.OpenSimplex(seed=seed * 7 + 14)
    moist = _simplex_field(sim_m, size, {base / 1.5: 1.0, base / 4: 0.5})
    forest = (material == GRASS) & (moist > forest_thr) & (rng.random((size, size)) < 0.7)
    material[forest] = TREE
    sim_d = opensimplex.OpenSimplex(seed=seed * 7 + 15)
    dry = _simplex_field(sim_d, size, {base / 1.3: 1.0})
    material[(material == GRASS) & (dry > 0.55)] = SAND

    # ── clear spawn / target bubbles to grass so the agent has room ──
    bubble = max(5.0, size / 12.0)
    near_bubble = (
        (np.sqrt((rr - sr) ** 2 + (cc - sc) ** 2) < bubble + 2)
        | (np.sqrt((rr - tr) ** 2 + (cc - tc) ** 2) < bubble + 2)
    )
    material[near_bubble] = GRASS
    return material


def _build_material_simplex(
    size: int,
    map_type: Literal["lake", "rocky", "balanced"],
    seed: int,
    spawn: tuple[int, int],
    target: tuple[int, int],
) -> np.ndarray:
    """Original Crafter-style simplex terrain (kept as a fallback generator)."""
    rng = np.random.default_rng(seed * 7 + 99)
    spawn_r, spawn_c = spawn
    target_r, target_c = target

    # one OpenSimplex per logical channel (start_a, start_b, water, mountain, sand)
    sim_sa = opensimplex.OpenSimplex(seed=seed * 7 + 1)
    sim_sb = opensimplex.OpenSimplex(seed=seed * 7 + 2)
    sim_w = opensimplex.OpenSimplex(seed=seed * 7 + 3)
    sim_m = opensimplex.OpenSimplex(seed=seed * 7 + 4)
    sim_sn = opensimplex.OpenSimplex(seed=seed * 7 + 5)

    # Slightly larger spawn / target bubbles than Crafter's default — we
    # need the agent to have room to manoeuvre before hitting terrain.
    bubble_radius = max(6.0, size / 10.0)

    rr = np.arange(size, dtype=np.float64)[:, None]
    cc = np.arange(size, dtype=np.float64)[None, :]
    dist_spawn = np.sqrt((rr - spawn_r) ** 2 + (cc - spawn_c) ** 2)
    start_a = bubble_radius - dist_spawn + 2.0 * _simplex_field(sim_sa, size, sizes=3)
    start_a = 1.0 / (1.0 + np.exp(-start_a))

    dist_target = np.sqrt((rr - target_r) ** 2 + (cc - target_c) ** 2)
    start_b = bubble_radius - dist_target + 2.0 * _simplex_field(sim_sb, size, sizes=3)
    start_b = 1.0 / (1.0 + np.exp(-start_b))

    start = np.maximum(start_a, start_b)

    # Map-type-specific terrain parameters.
    #   bias_strength : how strongly to push the main barrier along the
    #                   spawn-target line (0 for balanced — we want the
    #                   barrier to NOT cross the path).
    #   water_thr     : threshold above which a cell becomes water.
    #   mountain_thr  : threshold above which a cell becomes stone.
    if map_type == "balanced":
        bias_strength = 0.0
        water_thr_high = 0.70       # very little water (was 0.55 — most
                                    # balanced maps had small lakes, rocks
                                    # were scarce; we want the inverse)
        water_thr_sand_lo = 0.60
        water_thr_sand_hi = 0.70
        mountain_thr = 0.40         # more rocky outcrops (was 0.55)
    else:
        bias_strength = 0.40
        water_thr_high = 0.38      # slightly thinner lakes (was 0.30):
                                   # raises the water threshold so each
                                   # patch shrinks at the edges.
        water_thr_sand_lo = 0.33   # sand band shifts with water.
        water_thr_sand_hi = 0.38
        mountain_thr = 0.48        # slightly thinner rocks (was 0.40):
                                   # symmetric trim for rocky biome.

    bias_main = _diagonal_bias(
        size, (spawn_r, spawn_c), (target_r, target_c), sigma=size / 5.0
    )
    water = (
        _simplex_field(sim_w, size, sizes={15: 1.0, 5: 0.15}, normalize=False)
        + 0.1
        - 2.0 * start
        + bias_strength * bias_main
    )
    mountain = (
        _simplex_field(sim_m, size, sizes={15: 1.0, 5: 0.3}, normalize=True)
        - 4.0 * start
        - 0.3 * water
    )
    sand_noise = _simplex_field(sim_sn, size, sizes=9)

    material = np.full((size, size), GRASS, dtype=np.int8)
    start_grass = start > 0.5
    mountain_mask = (~start_grass) & (mountain > mountain_thr)
    sand_mask = (
        (~start_grass)
        & (~mountain_mask)
        & (water > water_thr_sand_lo)
        & (water <= water_thr_sand_hi)
        & (sand_noise > -0.2)
    )
    water_mask = (
        (~start_grass) & (~mountain_mask) & (~sand_mask) & (water > water_thr_high)
    )

    if map_type == "lake":
        material[mountain_mask] = ROCK
        material[sand_mask] = SAND
        material[water_mask] = WATER
        correct = sk.RAFT
    elif map_type == "rocky":
        material[mountain_mask] = WATER
        material[sand_mask] = SAND
        material[water_mask] = ROCK
        correct = sk.HARNESS
    else:  # balanced — small amounts of both, no advantage to either skill
        material[mountain_mask] = ROCK
        material[sand_mask] = SAND
        material[water_mask] = WATER
        correct = sk.NONE

    # Cosmetic-but-blocking flourishes: scattered trees on grass and lava
    # pockets in the stone region. Both are always impassable, so we keep
    # the densities low and avoid the spawn/target neighbourhoods.
    tree_noise = _simplex_field(sim_sn, size, sizes=7)  # reuse the sand sim
    near_spawn_or_target = (
        (np.sqrt((rr - spawn_r) ** 2 + (cc - spawn_c) ** 2) < bubble_radius + 2)
        | (np.sqrt((rr - target_r) ** 2 + (cc - target_c) ** 2) < bubble_radius + 2)
    )
    tree_mask = (
        (material == GRASS)
        & (tree_noise > 0.45)
        & (~near_spawn_or_target)
        & (rng.random((size, size)) > 0.75)
    )
    material[tree_mask] = TREE

    lava_mask = (mountain > 0.45) & (tree_noise > 0.30) & (~near_spawn_or_target)
    if map_type == "lake":
        # Lava replaces some ROCK cells (visually cool, no nav impact since
        # ROCK is already blocked for no-skill).
        material[lava_mask & (material == ROCK)] = LAVA
    else:
        # In rocky maps mountain → WATER. Replacing some of those with lava
        # blocks the raft on those cells, but it's a small fraction.
        material[lava_mask & (material == WATER)] = LAVA
    return material


# ───────────────────────── composed generator ─────────────────────────────
#
# A second structured generator (prototyped in ``scripts/gen_maps.py``). It
# composes each map from a tiny vocabulary of analytic atoms placed under
# per-biome constraints, then sprinkles perlin forest / dirt / sand:
#
#   * env ``rocky``  → a corner-anchored **ridge** (rock barrier, harness) + a
#                      non-overlapping lake, dirt collar around the ridge.
#   * env ``lake``   → a corner-anchored **river** (water barrier, raft) + a
#                      non-overlapping round mountain, sand banks along the river.
#   * env ``balanced``→ anti-diagonal split: one half a water atom, the other a
#                      rock atom, forest clustered in the middle corridor. A
#                      ridge there is corner-anchored to the target like above;
#                      lakes / mountains instead hug the centre diagonal.
#
# "Corner-anchored" = one end pinned to the top-right corner (= the target
# corner); the target keeps a small grass clearance so the barrier passes
# *nearby* it but never covers it. Spawn (bottom-left) is cleared likewise.


def _comp_logistic(x: np.ndarray, soft: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x / soft))


def _comp_simplex(seed: int, size: int, scales: dict[float, float]) -> np.ndarray:
    sim = opensimplex.OpenSimplex(seed=seed)
    out = np.zeros((size, size), dtype=np.float64)
    tot = 0.0
    for s, w in scales.items():
        coords = np.arange(size, dtype=np.float64) / s
        out += w * sim.noise2array(coords, coords)
        tot += w
    return out / tot if tot else out


def _comp_edge_noise(size: int, seed: int, amp: float) -> np.ndarray:
    return amp * _comp_simplex(seed, size, {5.0: 1.0, 2.5: 0.5})


def _comp_capsule(rr, cc, p0, p1, *, half_width, soft, wiggle, period, phase,
                  waist_frac=None, waist_pos_frac=0.5, waist_sigma_frac=0.13):
    """Meandering capsule between endpoints (row, col) with rounded caps;
    optional hourglass waist (ridge) vs constant width (river)."""
    r0, c0 = p0
    r1, c1 = p1
    dr, dc = r1 - r0, c1 - c0
    L = math.hypot(dr, dc) + 1e-9
    ur, uc = dr / L, dc / L
    u = (rr - r0) * ur + (cc - c0) * uc
    v = (rr - r0) * (-uc) + (cc - c0) * ur
    spine = wiggle * np.sin(2.0 * math.pi * u / period + phase)
    if waist_frac is not None:
        wp = waist_pos_frac * L
        sig = max(1.0, waist_sigma_frac * L)
        notch = (1.0 - waist_frac) * np.exp(-((u - wp) ** 2) / (2.0 * sig**2))
        ripple = 0.06 * np.sin(2.0 * math.pi * u / (L / 7.0) + phase)
        hw = half_width * np.clip(1.0 - notch + ripple, waist_frac * 0.8, 1.1)
    else:
        hw = half_width
    over = np.maximum(np.maximum(0.0, -u), u - L)
    dist = np.sqrt(over**2 + (v - spine) ** 2)
    return _comp_logistic(hw - dist, soft)


def _comp_blob(rr, cc, *, center, radius, soft, lobes, rng):
    """Irregular flat-topped disk — lake / round mountain."""
    d = np.sqrt((rr - center[0]) ** 2 + (cc - center[1]) ** 2)
    f = _comp_logistic(radius - d, soft)
    for _ in range(int(lobes)):
        oy, ox = rng.uniform(-radius * 0.7, radius * 0.7, size=2)
        s = radius * rng.uniform(0.35, 0.6)
        f = f + 0.6 * np.exp(-((rr - center[0] - oy) ** 2 + (cc - center[1] - ox) ** 2) / (2.0 * s**2))
    return f / max(f.max(), 1e-9)


def _comp_mask(field, size, rng):
    return (field + _comp_edge_noise(size, int(rng.integers(2**31)), amp=0.13)) > 0.5


def _comp_band(rr, cc, size, rng, p0, p1, *, kind):
    if kind == "river":
        return _comp_capsule(
            rr, cc, p0, p1,
            half_width=size * rng.uniform(0.045, 0.06), soft=0.9,
            wiggle=size * rng.uniform(0.10, 0.18), period=size * rng.uniform(0.5, 0.8),
            phase=rng.uniform(0, 2 * math.pi),
        )
    return _comp_capsule(   # ridge
        rr, cc, p0, p1,
        half_width=size * rng.uniform(0.06, 0.085), soft=1.1,
        wiggle=size * rng.uniform(0.05, 0.09), period=size * rng.uniform(0.6, 0.9),
        phase=rng.uniform(0, 2 * math.pi),
        waist_frac=rng.uniform(0.28, 0.42), waist_pos_frac=rng.uniform(0.30, 0.70),
        waist_sigma_frac=rng.uniform(0.10, 0.16),
    )


def _comp_blob_named(rr, cc, size, rng, center, *, kind):
    radius = size * (rng.uniform(0.12, 0.18) if kind == "lake" else rng.uniform(0.10, 0.15))
    lobes = rng.integers(3, 6) if kind == "lake" else rng.integers(2, 4)
    soft = 1.6 if kind == "lake" else 1.4
    return _comp_blob(rr, cc, center=center, radius=radius, soft=soft, lobes=lobes, rng=rng)


def _comp_place_blob_nonoverlap(rr, cc, size, rng, kind, avoid_mask, *,
                                lo=0.22, hi=0.78, gap=2, tries=60):
    from scipy.ndimage import distance_transform_edt
    blocked = distance_transform_edt(~avoid_mask) <= gap if avoid_mask.any() else avoid_mask
    last = None
    for _ in range(tries):
        center = (rng.uniform(lo, hi) * size, rng.uniform(lo, hi) * size)
        m = _comp_mask(_comp_blob_named(rr, cc, size, rng, center, kind=kind), size, rng)
        last = m
        if not (m & blocked).any():
            return m
    return last & ~avoid_mask


def _comp_rand_point_in_half(half, size, rng, margin=0.16):
    while True:
        r = rng.uniform(margin, 1 - margin) * size
        c = rng.uniform(margin, 1 - margin) * size
        if half == "TL" and (r + c) < size * 0.86:
            return (r, c)
        if half == "BR" and (r + c) > size * 1.14:
            return (r, c)


def _comp_rand_point_near_diag(half, size, rng, lo=0.05, hi=0.16):
    """A point hugging the anti-diagonal, offset just into ``half`` — used to
    pull balanced lakes / mountains toward the centre corridor."""
    s = rng.uniform(0.20, 0.80) * size           # position along the diagonal
    off = rng.uniform(lo, hi) * size             # perpendicular offset into half
    sign = 1.0 if half == "BR" else -1.0         # BR raises r+c, TL lowers it
    m = 0.10 * size
    r = float(np.clip(s + sign * off, m, size - 1 - m))
    c = float(np.clip((size - s) + sign * off, m, size - 1 - m))
    return (r, c)


def _build_material_composed(
    size: int,
    map_type: Literal["lake", "rocky", "balanced"],
    seed: int,
    spawn: tuple[int, int],
    target: tuple[int, int],
) -> np.ndarray:
    """Composed terrain (see scripts/gen_maps.py). Returns the material array
    with spawn/target cells left as GRASS (the caller stamps TARGET)."""
    from scipy.ndimage import distance_transform_edt

    rng = np.random.default_rng(seed * 977 + 23)
    rr, cc = _global_grid(size)
    # env biome -> recipe biome: env "lake" is the water-barrier (raft) map,
    # which our recipe realises as a river.
    recipe = {"lake": "river", "rocky": "rocky", "balanced": "balanced"}[map_type]
    corner = (rng.uniform(0, 3), size - 1 - rng.uniform(0, 3))

    rock_mask = np.zeros((size, size), dtype=bool)
    water_mask = np.zeros((size, size), dtype=bool)
    forest_corridor = None

    if recipe == "rocky":
        far = (rng.uniform(0.50, 0.92) * size, rng.uniform(0.0, 0.42) * size)
        rock_mask = _comp_mask(_comp_band(rr, cc, size, rng, corner, far, kind="ridge"), size, rng)
        water_mask = _comp_place_blob_nonoverlap(rr, cc, size, rng, "lake", rock_mask)
        dirt_w, sand_w, sand_extra = 3, 2, 2
    elif recipe == "river":
        far = (rng.uniform(0.50, 0.92) * size, rng.uniform(0.0, 0.42) * size)
        water_mask = _comp_mask(_comp_band(rr, cc, size, rng, corner, far, kind="river"), size, rng)
        rock_mask = _comp_place_blob_nonoverlap(rr, cc, size, rng, "round_mountain", water_mask)
        dirt_w, sand_w, sand_extra = 0, 3, 4
    else:  # balanced
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
                    p1 = _comp_rand_point_in_half(half, size, rng)
                    d = math.hypot(p1[0] - corner[0], p1[1] - corner[1])
                    if d > best_d:
                        best, best_d = p1, d
                    if d > 0.60 * size:
                        break
                return _comp_mask(_comp_band(rr, cc, size, rng, corner, best, kind="ridge"), size, rng)
            if name == "river":
                p0 = _comp_rand_point_in_half(half, size, rng)
                best, best_d = p0, -1.0
                for _ in range(40):
                    p1 = _comp_rand_point_in_half(half, size, rng)
                    d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                    if d > best_d:
                        best, best_d = p1, d
                    if d > 0.60 * size:
                        break
                return _comp_mask(_comp_band(rr, cc, size, rng, p0, best, kind="river"), size, rng)
            # lake / mountain: hug the centre diagonal
            ctr = _comp_rand_point_near_diag(half, size, rng)
            return _comp_mask(_comp_blob_named(rr, cc, size, rng, ctr, kind=name), size, rng)

        water_mask = place(water_name, water_half)
        rock_mask = place(rock_name, rock_half)
        water_mask &= ~rock_mask
        forest_corridor = np.exp(-((rr + cc - size) ** 2) / (2.0 * (size * 0.12) ** 2))
        dirt_w, sand_w, sand_extra = 2, 2, 2

    # never cover the target (or spawn): clear a small grass bubble
    clr = size * 0.065
    for (pr, pc) in (target, spawn):
        bubble = np.sqrt((rr - pr) ** 2 + (cc - pc) ** 2) < clr
        rock_mask &= ~bubble
        water_mask &= ~bubble

    open_land = ~(rock_mask | water_mask)
    material = np.full((size, size), GRASS, dtype=np.int8)

    # texture: clustered forest + sparse dirt / sand
    cluster = _comp_simplex(seed * 31 + 9, size, {size / 5.0: 1.0, size / 12.0: 0.5})
    forest = open_land & (cluster > 0.60) & (rng.random((size, size)) < 0.55)
    if forest_corridor is not None:
        forest |= open_land & (cluster > 0.42) & (forest_corridor > 0.55)
    else:
        forest |= open_land & (cluster > 0.52) & (rng.random((size, size)) < 0.22)
    material[forest] = TREE
    dry = _comp_simplex(seed * 19 + 5, size, {size / 10.0: 1.0})
    material[open_land & (material == GRASS) & (dry > 0.72)] = DIRT
    sandn = _comp_simplex(seed * 23 + 7, size, {size / 11.0: 1.0})
    material[open_land & (material == GRASS) & (sandn > 0.72)] = SAND

    # collars: dirt around rock (ridge), sand around all water (beaches). The
    # band hugging the feature stays solid; the outer sprinkle is kept light.
    if dirt_w > 0 and rock_mask.any():
        d = distance_transform_edt(~rock_mask)
        material[(d >= 1) & (d <= dirt_w) & open_land] = DIRT
        material[(d > dirt_w) & (d <= dirt_w + 2) & open_land
                 & (rng.random((size, size)) < 0.18)] = DIRT
    if sand_w > 0 and water_mask.any():
        d = distance_transform_edt(~water_mask)
        material[(d >= 1) & (d <= sand_w) & open_land] = SAND
        material[(d > sand_w) & (d <= sand_w + sand_extra) & open_land
                 & (rng.random((size, size)) < 0.16)] = SAND

    # components on top
    material[water_mask] = WATER
    material[rock_mask] = ROCK

    # guarantee a clean grass pad at spawn / target (the caller stamps TARGET)
    for (pr, pc) in (spawn, target):
        material[np.sqrt((rr - pr) ** 2 + (cc - pc) ** 2) <= 2.0] = GRASS
    return material


_CORRECT_OBJECT = {"lake": sk.RAFT, "rocky": sk.HARNESS, "balanced": sk.NONE}


def _generate_one(
    size: int,
    map_type: Literal["lake", "rocky", "balanced"],
    seed: int,
    margin_frac: float,
    generator: str = "composed",
) -> MapRecord:
    rng = np.random.default_rng(seed)
    zone = max(3, size // 4)
    spawn_r = int(rng.integers(size - zone, size))
    spawn_c = int(rng.integers(0, zone))
    target_r = int(rng.integers(0, zone))
    target_c = int(rng.integers(size - zone, size))
    spawn = (spawn_r, spawn_c)
    target = (target_r, target_c)
    # nominal biome label (the env reward does not depend on it)
    correct = _CORRECT_OBJECT[map_type]

    if generator == "components":
        material = _build_material_components(size, map_type, seed, spawn, target)
    elif generator == "composed":
        material = _build_material_composed(size, map_type, seed, spawn, target)
    elif generator == "simplex":
        material = _build_material_simplex(size, map_type, seed, spawn, target)
    else:
        raise ValueError(f"unknown generator: {generator!r}")

    if material[target_r, target_c] != GRASS:
        raise _Retry("target cell collided with terrain")
    if material[spawn_r, spawn_c] != GRASS:
        raise _Retry("spawn cell collided with terrain")
    material[target_r, target_c] = TARGET

    # Geometric (unit-cost) ctg — the env needs this for PBRS shaping, so the
    # only feasibility guard we keep is that the target is reachable (water and
    # rock are walkable, so this rarely fails — trees/lava could wall it off).
    ctg_unit = cost_to_go_unit(material, target, sk.NONE)
    if not math.isfinite(ctg_unit[spawn]):
        raise _Retry("target unreachable")

    # Slip-adjusted attempt costs kept as metadata only (NOT used to accept or
    # reject the map — the env is partially observable, so optimal-path
    # inequalities between skills are not meaningful per map).
    no_skill_cost = shortest_path_cost(material, spawn, target, sk.NONE)
    raft_cost = shortest_path_cost(material, spawn, target, sk.RAFT) + 1.0
    harness_cost = shortest_path_cost(material, spawn, target, sk.HARNESS) + 1.0
    ctg_none = ctg_raft = ctg_harness = ctg_unit  # share the same array

    return MapRecord(
        terrain=material,
        spawn=np.array([spawn_r, spawn_c], dtype=np.int32),
        target=np.array([target_r, target_c], dtype=np.int32),
        map_type=map_type,
        correct_object=correct,
        no_skill_cost=float(no_skill_cost),
        raft_cost=float(raft_cost),
        harness_cost=float(harness_cost),
        constraints_passed=True,
        seed=seed,
        ctg_none=ctg_none.astype(np.float32),
        ctg_raft=ctg_raft.astype(np.float32),
        ctg_harness=ctg_harness.astype(np.float32),
    )


def generate_map(
    size: int,
    map_type: MapType = "random",
    seed: int = 0,
    max_retries: int = 200,
    margin_frac: float = 0.05,
    generator: str = "composed",
) -> MapRecord:
    """Generate one validated Cogniland map.

    ``generator="composed"`` (default) builds each map from a small vocabulary
    of analytic atoms placed under per-biome constraints (corner-anchored ridge
    / river barrier + a non-overlapping lake / mountain + perlin forest / dirt /
    sand); ``generator="components"`` is the older heightmap-threshold recipe;
    ``generator="simplex"`` falls back to Crafter-style noise terrain. The env
    is a POMDP, so no cost-inequality is enforced — the only feasibility guard
    is that the target stays reachable, so almost every seed passes.
    """
    rng = np.random.default_rng(seed)
    last_reason: str | None = None
    for _ in range(max_retries):
        sub_seed = int(rng.integers(0, 2**31))
        chosen_type: Literal["lake", "rocky", "balanced"]
        if map_type == "random":
            chosen_type = ("lake", "rocky", "balanced")[int(rng.integers(0, 3))]
        elif map_type in ("lake", "rocky", "balanced"):
            chosen_type = map_type  # type: ignore[assignment]
        else:
            raise ValueError(f"unknown map_type: {map_type!r}")
        try:
            return _generate_one(
                size=size,
                map_type=chosen_type,
                seed=sub_seed,
                margin_frac=margin_frac,
                generator=generator,
            )
        except _Retry as exc:
            last_reason = str(exc)
            continue
    raise MapGenError(
        f"generate_map: could not satisfy constraints for size={size} "
        f"map_type={map_type} after {max_retries} retries "
        f"(last reason: {last_reason})"
    )
