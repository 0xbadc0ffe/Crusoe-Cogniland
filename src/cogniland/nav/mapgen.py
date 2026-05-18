"""Crafter-style procedural map generation for Cogniland navigation.

We borrow the simplex-noise terrain pipeline from Crafter's
``worldgen.py``: a sigmoid "spawn bubble" of guaranteed grass, a low-
frequency water field, and a low-frequency mountain field. We then strip
out the things Crafter places that this env doesn't need — cows, zombies,
skeletons, trees, coal, iron, diamond, lava, and the carved dirt/path
tunnels — leaving only the four terrain materials we actually use:
``grass``, ``sand``, ``water``, ``stone``.

The two map families differ only in how the two non-grass materials are
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
from .tiles import GRASS, LAVA, ROCK, SAND, TARGET, TREE, WATER

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


def _generate_one(
    size: int,
    map_type: Literal["lake", "rocky"],
    seed: int,
    margin_frac: float,
) -> MapRecord:
    rng = np.random.default_rng(seed)
    zone = max(3, size // 4)

    spawn_r = int(rng.integers(size - zone, size))
    spawn_c = int(rng.integers(0, zone))
    target_r = int(rng.integers(0, zone))
    target_c = int(rng.integers(size - zone, size))

    # one OpenSimplex per logical channel (start_a, start_b, water, mountain,
    # sand, worm-width modulation)
    sim_sa = opensimplex.OpenSimplex(seed=seed * 7 + 1)
    sim_sb = opensimplex.OpenSimplex(seed=seed * 7 + 2)
    sim_w = opensimplex.OpenSimplex(seed=seed * 7 + 3)
    sim_m = opensimplex.OpenSimplex(seed=seed * 7 + 4)
    sim_sn = opensimplex.OpenSimplex(seed=seed * 7 + 5)
    sim_wn = opensimplex.OpenSimplex(seed=seed * 7 + 6)

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
    #
    # We now place water and rock as *ridge bands* around the zero
    # level-set of a zero-mean simplex field. A cell is "inside the
    # band" when ``band_width − |raw_noise| > 0``; the slope of the
    # noise at the level-set governs the band width, which gives
    # worm-shaped features. An auxiliary low-frequency noise modulates
    # the band per cell, so the width also *varies along the worm*.
    #
    #   bias_strength : how strongly to push the main barrier along
    #                   the spawn-target line (0 for balanced).
    #   *_band        : nominal worm half-thickness (in noise-units).
    if map_type == "balanced":
        bias_strength = 0.0
        water_band = 0.025           # narrow ponds / streams
        mountain_band = 0.06         # vein-like rocky outcrops
        bias_extra = 0.0
    else:
        bias_strength = 0.40
        water_band = 0.04            # thin off-path; thickened by bias_extra
        mountain_band = 0.05         # on the spawn-target line for the
        bias_extra = 0.36            # *correct* skill's terrain only.

    bias_main = _diagonal_bias(
        size, (spawn_r, spawn_c), (target_r, target_c), sigma=size / 5.0
    )

    # Width modulation: scale the worm band per cell so the same worm
    # narrows in places and widens elsewhere. Range ~[0.4, 1.6].
    width_mod = _simplex_field(sim_wn, size, sizes={15: 1.0, 6: 0.3}, normalize=True)
    width_mul = 1.0 + 0.6 * width_mod

    # Zero-mean noise fields; ridges sit at |raw| ≈ 0.
    raw_w = _simplex_field(sim_w, size, sizes={18: 1.0, 7: 0.30}, normalize=True)
    raw_m = _simplex_field(sim_m, size, sizes={18: 1.0, 7: 0.35}, normalize=True)

    # water > 0 inside the worm band; `bias_extra * bias_main` thickens it
    # along the spawn-target line for lake/rocky. `start` suppresses it
    # in the spawn/target bubbles.
    water = (
        water_band * width_mul - np.abs(raw_w)
        - 0.20 * start
        + bias_extra * bias_main
    )
    mountain = (
        mountain_band * width_mul - np.abs(raw_m)
        - 0.20 * start
        - 0.40 * np.clip(water, 0.0, None)
    )
    sand_noise = _simplex_field(sim_sn, size, sizes=9)

    # All thresholds now centre on 0 (inside the band). Sand sits in
    # a thin envelope just outside the water band, creating beaches.
    water_thr_high = 0.0
    water_thr_sand_lo = -0.04
    water_thr_sand_hi = 0.0
    mountain_thr = 0.0

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

    if material[target_r, target_c] != GRASS:
        raise _Retry("target cell collided with terrain")
    if material[spawn_r, spawn_c] != GRASS:
        raise _Retry("spawn cell collided with terrain")
    material[target_r, target_c] = TARGET

    spawn = (spawn_r, spawn_c)
    target = (target_r, target_c)

    # Geometric (unit-cost) ctg — used by the env for PBRS shaping. Now
    # that water/rock/trees are universally walkable, this is the same for
    # all three skills, so we store one array under each name.
    ctg_unit = cost_to_go_unit(material, target, sk.NONE)
    if not math.isfinite(ctg_unit[spawn]):
        raise _Retry("no-skill path infeasible")

    # Expected-attempt costs (slip-adjusted) — what the agent actually pays
    # in env steps. NONE pays no build, RAFT/HARNESS pay +1 for the build
    # action itself. All use the same shortest-path-cost routine.
    no_skill_cost = shortest_path_cost(material, spawn, target, sk.NONE)
    raft_cost = shortest_path_cost(material, spawn, target, sk.RAFT) + 1.0
    harness_cost = shortest_path_cost(material, spawn, target, sk.HARNESS) + 1.0
    if not all(math.isfinite(x) for x in (no_skill_cost, raft_cost, harness_cost)):
        raise _Retry("a path is infeasible")
    ctg_none = ctg_raft = ctg_harness = ctg_unit  # share the same array

    margin = margin_frac * no_skill_cost
    if map_type == "lake":
        if not (raft_cost < no_skill_cost - margin):
            raise _Retry("lake: raft not enough better than no-skill")
        if not (harness_cost > no_skill_cost + margin):
            raise _Retry("lake: harness not enough worse than no-skill")
    elif map_type == "rocky":
        if not (harness_cost < no_skill_cost - margin):
            raise _Retry("rocky: harness not enough better than no-skill")
        if not (raft_cost > no_skill_cost + margin):
            raise _Retry("rocky: raft not enough worse than no-skill")
    else:  # balanced — no-skill must be best, both items strictly worse
        if not (raft_cost > no_skill_cost + margin):
            raise _Retry("balanced: raft not enough worse than no-skill")
        if not (harness_cost > no_skill_cost + margin):
            raise _Retry("balanced: harness not enough worse than no-skill")

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
) -> MapRecord:
    """Generate one validated Cogniland map.

    Crafter-style terrain is naturally random; even with the diagonal bias
    we apply, only a fraction of seeds satisfy the strict cost inequality,
    so ``max_retries`` defaults higher than the structured-band generator.
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
            )
        except _Retry as exc:
            last_reason = str(exc)
            continue
    raise MapGenError(
        f"generate_map: could not satisfy constraints for size={size} "
        f"map_type={map_type} after {max_retries} retries "
        f"(last reason: {last_reason})"
    )
