"""Hand-crafted behavioral test maps for Cogniland.

9 maps designed to probe specific agent behaviors.  Every map has an
optimal (reward-maximising) route that is NOT a straight line.

Occlusion maps (visibility-based planning):
  mountain_dome    — central mountain dome hides the far side; agent must go around
  hidden_ridge     — mountain wall hidden behind rocky foothills; visible only from high ground

Resource management maps:
  forest_corridor  — forest band across center: slow but heals; grassland gap is faster but drains
  fork_choice      — Y-junction: grassland route (fast/drain) vs forest route (slow/heals)
  desert_zigzag    — sandy island with forest oases in a zigzag; straight line = death

Routing / obstacle maps:
  lake_detour      — central lake blocks direct path; must circumnavigate
  island_hop       — three islands; middle island is off the direct line but has forest
  mountain_bridge  — two islands, mountain bridge (short/deadly) vs grassland bridge (long/safe)
  gauntlet         — diagonal journey through mixed terrain with obstacles
"""

from __future__ import annotations

import math

import numpy as np
import torch

# ── Terrain height values (midpoints of each threshold band) ──────────────
SIZE = 250

OCEAN      = 0.003
DEEP_WATER = 0.016
WATER      = 0.038
BEACH      = 0.055
SANDY      = 0.080
GRASSLAND  = 0.150
FOREST     = 0.400
ROCKY      = 0.650
MOUNTAINS  = 0.850

LAND_THRESHOLD = 0.050  # cells above this are land


# ── Canvas & brush primitives ─────────────────────────────────────────────

def _canvas() -> np.ndarray:
    """Fresh all-ocean canvas."""
    return np.full((SIZE, SIZE), OCEAN, dtype=np.float32)


def _dist(cy: float, cx: float) -> np.ndarray:
    Y, X = np.ogrid[:SIZE, :SIZE]
    return np.sqrt((Y.astype(np.float64) - cy) ** 2 + (X.astype(np.float64) - cx) ** 2).astype(np.float32)


def _circle(canvas: np.ndarray, cy: float, cx: float, r: float, val: float) -> None:
    canvas[_dist(cy, cx) < r] = val


def _ring(canvas: np.ndarray, cy: float, cx: float, r_in: float, r_out: float, val: float) -> None:
    d = _dist(cy, cx)
    canvas[(d >= r_in) & (d < r_out)] = val


def _rect(canvas: np.ndarray, r1: int, c1: int, r2: int, c2: int, val: float) -> None:
    canvas[r1:r2, c1:c2] = val


def _ellipse(canvas: np.ndarray, cy: float, cx: float, ry: float, rx: float, val: float) -> None:
    """Fill an axis-aligned ellipse."""
    Y, X = np.ogrid[:SIZE, :SIZE]
    mask = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 < 1.0
    canvas[mask] = val


def _paint_island(
    canvas: np.ndarray, cy: float, cx: float, r: float,
    layers: tuple[tuple[float, float], ...] | None = None,
) -> None:
    """Concentric-terrain island. layers = [(frac_of_r, terrain_val), ...] outermost first."""
    if layers is None:
        layers = (
            (1.00, BEACH),
            (0.88, SANDY),
            (0.74, GRASSLAND),
            (0.52, FOREST),
            (0.30, ROCKY),
            (0.14, MOUNTAINS),
        )
    d = _dist(cy, cx)
    for frac, val in layers:
        canvas[d < r * frac] = val


def _ocean_border(canvas: np.ndarray, fade: int = 22) -> None:
    """Smoothly fade edges toward ocean using per-pixel distance from nearest edge."""
    rows = np.arange(SIZE, dtype=np.float32)
    cols = np.arange(SIZE, dtype=np.float32)
    dist_r = np.minimum(rows, SIZE - 1 - rows)
    dist_c = np.minimum(cols, SIZE - 1 - cols)
    dist_edge = np.minimum(dist_r[:, None], dist_c[None, :])
    weight = np.clip(dist_edge / fade, 0.0, 1.0) ** 0.55
    canvas *= weight


def _to_tensor(canvas: np.ndarray, fade: int = 22) -> torch.Tensor:
    _ocean_border(canvas, fade=fade)
    return torch.from_numpy(canvas.copy())


def _clip_island(canvas: np.ndarray, cy: float, cx: float, r: float) -> None:
    """Clip everything outside island radius to ocean, with a beach rim."""
    d = _dist(cy, cx)
    canvas[d > r] = OCEAN
    canvas[(d > r - 4) & (d <= r)] = BEACH


# ── Map generators ────────────────────────────────────────────────────────
# Each returns (canvas, spawn_rc, target_rc).
# spawn/target are (row, col) tuples on land (value > LAND_THRESHOLD).

# ─── OCCLUSION MAPS ──────────────────────────────────────────────────────

def _map_mountain_dome() -> tuple[np.ndarray, tuple, tuple]:
    """Central mountain dome.  From grassland (vis=5) the far side is hidden.

    Climbing the mountain edge (vis=22) reveals an impenetrable dome of peaks.
    Optimal: walk around the dome on the grassland/forest ring.
    """
    c = _canvas()
    _paint_island(c, 125, 125, 112, layers=(
        (1.00, BEACH),
        (0.94, GRASSLAND),
    ))
    # Large mountain dome — radius 55, so ~110 cells wide
    _circle(c, 125, 125, 55, MOUNTAINS)
    _circle(c, 125, 125, 48, MOUNTAINS)   # solid interior
    # Rocky apron around dome (agent climbs here first, vis=10)
    _ring(c, 125, 125, 55, 62, ROCKY)
    # Forest ring outside rocky apron for resource recharging
    _ring(c, 125, 125, 62, 78, FOREST)
    # Spawn west, target east — direct line crosses the dome
    return c, (125, 28), (125, 222)


def _map_hidden_ridge() -> tuple[np.ndarray, tuple, tuple]:
    """Mountain wall hidden behind a rocky foothill belt.

    From grassland (vis=5) the agent sees only grass ahead.  Stepping onto the
    rocky foothills (vis=10) reveals a long N-S mountain wall behind them.
    From mountain-top (vis=22) the agent sees the wall extends far north and south.
    Must detour north via a grassland gap in the ridge.
    """
    c = _canvas()
    _paint_island(c, 125, 125, 112, layers=(
        (1.00, BEACH),
        (0.94, GRASSLAND),
    ))
    # Rocky foothills belt: cols 110-125 (agent hits this first walking east)
    _rect(c, 42, 110, 208, 125, ROCKY)
    # Mountain wall behind foothills: cols 125-155
    _rect(c, 42, 125, 208, 155, MOUNTAINS)
    # Northern gap in the wall: rows 30-60 → grassland (safe passage)
    _rect(c, 30, 110, 60, 155, GRASSLAND)
    # Forest patch near the gap for resource management
    _circle(c, 48, 132, 14, FOREST)
    # Forest patches on both sides of the wall for resources
    _circle(c, 125, 75, 16, FOREST)
    _circle(c, 125, 185, 16, FOREST)
    # Clip to island shape
    _clip_island(c, 125, 125, 112)
    # Spawn west, target east — direct path hits the wall
    return c, (125, 28), (125, 222)


# ─── RESOURCE MANAGEMENT MAPS ────────────────────────────────────────────

def _map_forest_corridor() -> tuple[np.ndarray, tuple, tuple]:
    """Thick forest band runs N-S across center with a grassland gap to the north.

    Through the forest: slow (cost=3.0) but gains resources (+5.0) and HP (+8.0).
    Around the gap: faster grassland but longer distance + resource drain.
    Optimal for well-provisioned agent: through the forest (net resource gain).
    """
    c = _canvas()
    _paint_island(c, 125, 125, 112, layers=(
        (1.00, BEACH),
        (0.94, GRASSLAND),
    ))
    # Dense forest band: cols 100-155, rows 65-210
    _rect(c, 65, 100, 210, 155, FOREST)
    # Northern grassland gap: rows 30-65
    _rect(c, 30, 100, 65, 155, GRASSLAND)
    # Some scattered forest patches outside the corridor
    _circle(c, 125, 55, 14, FOREST)
    _circle(c, 125, 195, 14, FOREST)
    _clip_island(c, 125, 125, 112)
    # Spawn west, target east — straight line crosses forest
    return c, (130, 28), (130, 222)


def _map_fork_choice() -> tuple[np.ndarray, tuple, tuple]:
    """Y-junction: west grassland route vs east forest route.

    Central mountain ridge forces a choice.
    Grassland route: fast (cost=2.25) but drains resources (-1.5/step).
    Forest route: slow (cost=3.0) but heals HP (+8.0) and gains resources (+5.0).
    Optimal: forest route — arrives healthy with full resources.
    """
    c = _canvas()
    _paint_island(c, 125, 125, 112, layers=(
        (1.00, BEACH),
        (0.94, GRASSLAND),
    ))
    # Central mountain ridge splitting the island N-S
    _rect(c, 55, 115, 195, 135, MOUNTAINS)
    _rect(c, 55, 108, 195, 115, ROCKY)
    _rect(c, 55, 135, 195, 142, ROCKY)
    # Western route: pure grassland (already the base)
    # Eastern route: forest corridor
    _rect(c, 55, 150, 195, 210, FOREST)
    # Merge zones at top and bottom
    _circle(c, 48, 125, 28, GRASSLAND)   # northern junction
    _circle(c, 202, 125, 28, GRASSLAND)  # southern junction
    # Forest patch at southern merge for resources
    _circle(c, 202, 155, 14, FOREST)
    _clip_island(c, 125, 125, 112)
    # Spawn south, target north
    return c, (210, 125), (40, 125)


def _map_desert_zigzag() -> tuple[np.ndarray, tuple, tuple]:
    """Sandy island with forest oases in a zigzag pattern.

    Straight line from SW to NE crosses pure sandy terrain: resource drain (-1.5/step)
    kills the agent before reaching the target.
    Optimal: zigzag between oases to recharge resources.
    """
    c = _canvas()
    _paint_island(c, 125, 125, 108, layers=(
        (1.00, BEACH),
        (0.92, SANDY),
    ))
    # Forest oases in a zigzag pattern from SW to NE
    oases = [
        (190, 60,  14),   # near spawn
        (165, 110, 13),
        (135, 65,  12),
        (110, 135, 14),
        (85,  80,  13),
        (60,  140, 14),   # near target
    ]
    for cy, cx, r in oases:
        _circle(c, cy, cx, r, FOREST)
    # Small grassland patches connecting oases for flavor
    for cy, cx, r in oases:
        _ring(c, cy, cx, r, r + 5, GRASSLAND)
    _clip_island(c, 125, 125, 108)
    # Spawn SW, target NE — straight line misses most oases
    return c, (192, 52), (58, 198)


# ─── ROUTING / OBSTACLE MAPS ─────────────────────────────────────────────

def _map_lake_detour() -> tuple[np.ndarray, tuple, tuple]:
    """Central lake blocks the direct E-W path.

    Circumnavigate north (grassland, shorter arc) or south (longer arc with
    forest patches for resource gain).  Swimming is possible but costly.
    Optimal: northern arc for speed, or southern arc if resources are low.
    """
    c = _canvas()
    _paint_island(c, 125, 125, 112, layers=(
        (1.00, BEACH),
        (0.94, GRASSLAND),
    ))
    # Central lake — large enough to force a real detour
    _circle(c, 125, 125, 48, WATER)
    _circle(c, 125, 125, 36, DEEP_WATER)
    _circle(c, 125, 125, 22, OCEAN)
    # Forest ring around lake (resource station)
    _ring(c, 125, 125, 48, 62, FOREST)
    # Rocky promontory extending south from lake (blocks south shortcut)
    _rect(c, 155, 110, 180, 140, ROCKY)
    _clip_island(c, 125, 125, 112)
    # Spawn west, target east
    return c, (125, 22), (125, 228)


def _map_island_hop() -> tuple[np.ndarray, tuple, tuple]:
    """Three islands.  Middle island is north of the direct line but has forest.

    Direct swim from island 1 to 3: crosses deep ocean, agent likely dies.
    Optimal: hop north to middle island (forest recharge), then continue east.
    Non-straight route tests planning ahead.
    """
    c = _canvas()
    # West island (spawn) — medium with forest core
    _paint_island(c, 145, 48, 38, layers=(
        (1.00, BEACH), (0.82, GRASSLAND), (0.50, FOREST),
    ))
    # Middle island — north of direct line, rich forest for recharging
    _paint_island(c, 85, 130, 32, layers=(
        (1.00, BEACH), (0.78, GRASSLAND), (0.42, FOREST),
    ))
    # East island (target) — medium with forest core
    _paint_island(c, 145, 210, 38, layers=(
        (1.00, BEACH), (0.82, GRASSLAND), (0.50, FOREST),
    ))
    # Shallow water stepping stones toward middle island (make hop feasible)
    _circle(c, 120, 85, 10, WATER)
    _circle(c, 105, 170, 10, WATER)
    # Spawn on west island, target on east island
    return c, (145, 28), (145, 225)


def _map_mountain_bridge() -> tuple[np.ndarray, tuple, tuple]:
    """Two islands connected by two bridges.

    North bridge: mountain (short, 3 cells wide — high cost + massive drain).
    South bridge: grassland (longer arc, manageable drain).
    Optimal: take the southern grassland bridge despite longer distance.
    """
    c = _canvas()
    # West island
    _paint_island(c, 125, 60, 52, layers=(
        (1.00, BEACH), (0.85, GRASSLAND), (0.55, FOREST),
    ))
    # East island
    _paint_island(c, 125, 190, 52, layers=(
        (1.00, BEACH), (0.85, GRASSLAND), (0.55, FOREST),
    ))
    # North bridge: mountains, rows 85-100, cols 108-142
    _rect(c, 85, 108, 100, 142, MOUNTAINS)
    _rect(c, 82, 108, 85, 142, ROCKY)
    _rect(c, 100, 108, 103, 142, ROCKY)
    # South bridge: grassland, rows 155-175, cols 108-142
    _rect(c, 155, 108, 175, 142, GRASSLAND)
    _rect(c, 152, 108, 155, 142, BEACH)
    _rect(c, 175, 108, 178, 142, BEACH)
    # Forest patches on south bridge for resources
    _circle(c, 165, 125, 8, FOREST)
    # Spawn on west island, target on east island
    return c, (125, 22), (125, 228)


def _map_gauntlet() -> tuple[np.ndarray, tuple, tuple]:
    """Diagonal gauntlet: all terrain types, multiple obstacles.

    SW → NE diagonal crosses:
    1. Sandy zone (resource drain)
    2. Mountain ridge (must detour around)
    3. Lake obstacle (must go around)
    4. Forest zone (recharge opportunity)
    Optimal route zigzags around obstacles, not a straight diagonal.
    """
    c = _canvas()
    _paint_island(c, 125, 125, 115, layers=(
        (1.00, BEACH),
        (0.92, GRASSLAND),
    ))
    Y, X = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing='ij')
    d = _dist(125, 125)
    # SW quadrant → sandy desert
    sw = (Y > 140) & (X < 110) & (d < 110)
    c[sw] = SANDY
    # Forest oases in sandy zone
    for cy, cx in [(165, 55), (190, 85), (155, 80)]:
        _circle(c, cy, cx, 11, FOREST)
    # Diagonal mountain ridge NW-SE through center
    for i in range(45):
        ry, rx = 85 + i, 85 + i
        _circle(c, ry, rx, 12, MOUNTAINS)
        _ring(c, ry, rx, 12, 18, ROCKY)
    # Gap in the ridge around row 115 (narrow passage)
    _rect(c, 110, 105, 122, 127, GRASSLAND)
    # NE lake obstacle
    _circle(c, 80, 175, 22, WATER)
    _circle(c, 80, 175, 14, DEEP_WATER)
    # NE forest zone (recharge after lake detour)
    _ring(c, 80, 175, 22, 35, FOREST)
    # Forest near target
    _circle(c, 55, 195, 14, FOREST)
    _clip_island(c, 125, 125, 115)
    # Spawn SW, target NE
    return c, (195, 52), (52, 200)


# ── Registry ──────────────────────────────────────────────────────────────

_GENERATORS: dict[str, object] = {
    "mountain_dome":    _map_mountain_dome,
    "hidden_ridge":     _map_hidden_ridge,
    "forest_corridor":  _map_forest_corridor,
    "fork_choice":      _map_fork_choice,
    "desert_zigzag":    _map_desert_zigzag,
    "lake_detour":      _map_lake_detour,
    "island_hop":       _map_island_hop,
    "mountain_bridge":  _map_mountain_bridge,
    "gauntlet":         _map_gauntlet,
}

_MAPS:    dict[str, torch.Tensor] | None = None
_SPAWNS:  dict[str, tuple[int, int]] | None = None
_TARGETS: dict[str, tuple[int, int]] | None = None


def _ensure_loaded() -> None:
    global _MAPS, _SPAWNS, _TARGETS
    if _MAPS is not None:
        return
    _MAPS, _SPAWNS, _TARGETS = {}, {}, {}
    for name, fn in _GENERATORS.items():
        canvas, spawn, target = fn()  # type: ignore[operator]
        _MAPS[name]    = _to_tensor(canvas)
        _SPAWNS[name]  = spawn
        _TARGETS[name] = target


def get_map(name: str) -> torch.Tensor:
    _ensure_loaded()
    if name not in _MAPS:  # type: ignore[operator]
        raise KeyError(
            f"Unknown custom map '{name}'. Available: {list(_GENERATORS.keys())}"
        )
    return _MAPS[name]  # type: ignore[index]


def get_spawn(name: str) -> tuple[int, int] | None:
    _ensure_loaded()
    return _SPAWNS.get(name)  # type: ignore[union-attr]


def get_target(name: str) -> tuple[int, int] | None:
    _ensure_loaded()
    return _TARGETS.get(name)  # type: ignore[union-attr]


def list_maps() -> list[str]:
    return list(_GENERATORS.keys())
