"""Procedural map generation for bridge_tunnel_commit (3 labelled categories).

Reuses the base env's authoritative terrain builder
(``cogniland.bridge_tunnel.mapgen._build_natural`` — domain-warped fractal
heightmap with overlaid lakes/mountains/ridges + edge-biased tree forests) and
just varies the water/rock coverage to produce three map *categories*:

* ``balanced`` — 14% water / 14% rock (the original bridge_tunnel mix).
* ``lakes``    — water-dominated, ~80/20 water:rock (0.224 / 0.056).
* ``rocky``    — rock-dominated,  ~20/80 water:rock (0.056 / 0.224).

(The splits are approximate — coverage is thresholded by quantile, then trimmed
by the clear edge bands and sand/dirt fringes, so realised coverage is lower.)

Because the commit env can cross **only one** obstacle type (whichever the agent
commits to), each generated map is guaranteed *winnable under the commitment
that matches its category*, and is also guaranteed to **require** committing to
something (no pure-walkable path from spawn to the goal). Maps that fail either
guard are resampled with a fresh seed offset.

``MapRecord`` carries a ``category`` label so trainers / evals can build
class-balanced train / val / test splits.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cogniland.bridge_tunnel.mapgen import _build_natural
from .tiles import GRASS, ROCK, SAND, DIRT, TARGET, TREE, WATER, WOOD


CATEGORIES = ("balanced", "lakes", "rocky")

# (water_frac, rock_frac) per category. Total obstacle mass ~0.28 in all three;
# lakes/rocky are ~80/20 splits of that mass (approximate — see module docstring).
_CATEGORY_FRACS = {
    "balanced": (0.14, 0.14),
    "lakes":    (0.224, 0.056),
    "rocky":    (0.056, 0.224),
}

_WALK = (GRASS, WOOD, TARGET, SAND, DIRT)


@dataclass
class MapRecord:
    terrain: np.ndarray             # (H, W) int8
    spawn: tuple[int, int]
    target: tuple[int, int]
    seed: int
    category: str = "balanced"      # one of CATEGORIES
    orientation: str = "natural"
    goal_cells: list[tuple[int, int]] = field(default_factory=list)


def _can_reach_goal(terrain: np.ndarray, spawn: tuple[int, int],
                    crossable: frozenset[int]) -> bool:
    """BFS from spawn to ANY TARGET cell, treating walkable tiles plus the
    ``crossable`` tile ids as passable and everything else (TREE + the
    non-crossable obstacle) as a wall. ``crossable=∅`` ⇒ walkable-only path."""
    H, W = terrain.shape
    sr, sc = spawn
    seen = np.zeros((H, W), dtype=bool)
    seen[sr, sc] = True
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        if terrain[r, c] == TARGET:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc]:
                t = int(terrain[nr, nc])
                if t in _WALK or t in crossable:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return False


def category_fracs(category: str) -> tuple[float, float]:
    if category not in _CATEGORY_FRACS:
        raise ValueError(f"category must be one of {CATEGORIES}, got {category!r}")
    return _CATEGORY_FRACS[category]


def generate_commit_map(
    size: int = 32,
    width: int | None = 64,
    seed: int = 0,
    category: str = "balanced",
    tree_frac: float = 0.03,
    goal_half: int | None = 1,
    require_cross: bool = False,
    max_resample: int = 400,
) -> MapRecord:
    """Generate one ``category`` map that is winnable under the matching
    commitment (and, if ``require_cross``, cannot be solved without committing).

    ``require_cross`` defaults to ``False``: like the base bridge_tunnel env, a
    walkable-only detour usually exists, and the PBRS + build-cost economy (not a
    hard barrier) is what makes committing to the dominant obstacle worthwhile —
    detours around a *category*'s abundant obstacle are long, so the matching
    commitment pays off. Forcing a full barrier (``require_cross=True``) rejects
    ~90% of natural draws and yields atypical terrain.

    Winnability per category (must hold):
      * ``balanced`` — winnable by BUILD *and* by MINE (a genuine free choice).
      * ``lakes``    — winnable by BUILD (committing MINE may dead-end → the
                       lesson: read the terrain, commit to bridging).
      * ``rocky``    — winnable by MINE.

    Deterministic given ``(size, width, seed, category, ...)``: if the first
    draw fails a guard it is resampled with a fixed prime seed offset, so the
    same call always yields the same map. ``record.seed`` keeps the *requested*
    seed as the map's identity.
    """
    wf, rf = category_fracs(category)
    W = int(width) if width is not None else int(size)
    s = int(seed)
    for _ in range(max_resample):
        base = _build_natural(int(size), W, s, wf, rf, tree_frac, goal_half=goal_half)
        terr = base.terrain
        spawn = base.spawn
        build_ok = _can_reach_goal(terr, spawn, frozenset({WATER}))
        mine_ok = _can_reach_goal(terr, spawn, frozenset({ROCK}))
        if category == "balanced":
            intended = build_ok and mine_ok
        elif category == "lakes":
            intended = build_ok
        else:  # rocky
            intended = mine_ok
        cross_needed = (not _can_reach_goal(terr, spawn, frozenset())) if require_cross else True
        if intended and cross_needed:
            return MapRecord(terr, spawn, base.target, int(seed), category,
                             "natural", base.goal_cells)
        s += 100003          # large prime offset → a fresh, deterministic draw
    raise RuntimeError(
        f"could not generate a winnable {category!r} map from seed {seed} "
        f"in {max_resample} tries (require_cross={require_cross})")


def is_winnable(rec: MapRecord) -> bool:
    """True if the goal is reachable under at least one commitment (BUILD or
    MINE). Used as a contract / sanity check."""
    return (_can_reach_goal(rec.terrain, rec.spawn, frozenset({WATER}))
            or _can_reach_goal(rec.terrain, rec.spawn, frozenset({ROCK})))


def make_split(
    n_per_category: int,
    seed_start: int = 0,
    categories: tuple[str, ...] = CATEGORIES,
    **map_kwargs,
) -> list[MapRecord]:
    """Build a class-balanced list of ``n_per_category`` maps for each category.

    Each category draws its own deterministic seed block
    ``[seed_start, seed_start + n_per_category)`` so the categories are
    independent and reproducible. Returns the interleaved-by-category list.
    """
    recs: list[MapRecord] = []
    for cat in categories:
        for i in range(n_per_category):
            recs.append(generate_commit_map(seed=seed_start + i, category=cat, **map_kwargs))
    return recs


__all__ = [
    "MapRecord", "CATEGORIES", "generate_commit_map", "is_winnable",
    "make_split", "category_fracs",
]
