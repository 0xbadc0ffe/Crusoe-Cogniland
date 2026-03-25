"""Hand-crafted behavioral test maps for Cogniland.

16 maps designed to probe specific agent behaviors:
  open_grassland   — wide open island, baseline navigation
  forest_wall      — forest barrier: push through (heals) or detour around gap
  mountain_pass    — mountain range with two passes (rocky vs grassland)
  ring_island      — ring of land around central ocean: walk ring or swim through
  three_islands    — island chain requiring two water crossings
  peninsula        — long thin peninsula: walk along or swim beside
  fork_road        — Y-junction: fast-drain grassland vs slow-healing forest route
  desert_oasis     — sandy island with forest oases: resource scarcity
  lake_crossing    — central lake: circumnavigate or swim across
  archipelago      — scattered islands: multi-hop crossings
  mountain_dome    — central mountain: climb over or walk around grassland ring
  checkpoint_run   — long N-S corridor with forest recharge stations
  two_bridges      — two islands linked by mountain bridge and grassland bridge
  forest_detour    — forest island slightly off direct route: detour for resources?
  rocky_coast      — rocky coastline with grassland interior
  grand_circuit    — gauntlet: all terrain types, long diagonal
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


# ── Map generators ────────────────────────────────────────────────────────
# Each returns (canvas, spawn_rc, target_rc).
# spawn/target are (row, col) tuples on land (value > LAND_THRESHOLD = 0.05).


def _map_open_grassland() -> tuple[np.ndarray, tuple, tuple]:
    """Wide grassland island with scattered forest patches. Baseline navigation."""
    c = _canvas()
    _paint_island(c, 125, 125, 105, layers=(
        (1.00, BEACH),
        (0.90, SANDY),
        (0.78, GRASSLAND),
    ))
    # Forest patches for resource management
    for cy, cx in [(85, 85), (165, 85), (85, 165), (165, 165), (125, 125)]:
        _circle(c, cy, cx, 16, FOREST)
    return c, (125, 38), (125, 212)


def _map_forest_wall() -> tuple[np.ndarray, tuple, tuple]:
    """Forest wall across center with a northern gap.
    Push through wall (heals resources!) or detour around the gap."""
    c = _canvas()
    _paint_island(c, 125, 125, 110, layers=(
        (1.00, BEACH),
        (0.92, GRASSLAND),
    ))
    # Forest wall cols 97-153, rows 85-165
    _rect(c, 85, 97, 165, 153, FOREST)
    # Northern gap: clear rows 57-85 of forest
    _rect(c, 57, 97, 85, 153, GRASSLAND)
    # Extra grassland buffer around the gap
    _circle(c, 70, 125, 22, GRASSLAND)
    return c, (130, 32), (130, 218)


def _map_mountain_pass() -> tuple[np.ndarray, tuple, tuple]:
    """Mountain range with two passes: narrow rocky (north) vs wide grassland (south)."""
    c = _canvas()
    _paint_island(c, 125, 125, 112, layers=(
        (1.00, BEACH),
        (0.93, GRASSLAND),
    ))
    # Mountain range: thick vertical band cols 103-147
    _rect(c, 18, 103, 232, 147, MOUNTAINS)
    # Pass 1 (north): narrow rocky corridor rows 60-78
    _rect(c, 60, 103, 78, 147, ROCKY)
    # Pass 2 (south): wide grassland corridor rows 165-195
    _rect(c, 165, 103, 195, 147, GRASSLAND)
    # Forest on both sides for resources
    _circle(c, 125, 62, 18, FOREST)
    _circle(c, 125, 188, 18, FOREST)
    # Island mask (clip mountains outside island)
    d = _dist(125, 125)
    c[d > 112] = OCEAN
    c[(d > 108) & (d <= 112)] = BEACH
    return c, (125, 32), (125, 218)


def _map_ring_island() -> tuple[np.ndarray, tuple, tuple]:
    """Ring of grassland around a central ocean lake.
    Walk the ring (long, no water cost) or swim through center (boat cost + drain)."""
    c = _canvas()
    # Outer ocean (default)
    # Ring: grassland band r_in=52, r_out=82
    _ring(c, 125, 125, 52, 82, GRASSLAND)
    _ring(c, 125, 125, 72, 82, BEACH)     # outer edge beach
    _ring(c, 125, 125, 52, 60, SANDY)     # inner edge sandy
    # Central water body
    _circle(c, 125, 125, 52, WATER)
    _circle(c, 125, 125, 38, DEEP_WATER)
    _circle(c, 125, 125, 22, OCEAN)
    # Forest patches on ring at 4 compass points for resources
    for angle_deg in [45, 135, 225, 315]:
        rad = math.radians(angle_deg)
        ry = int(125 + 67 * math.sin(rad))
        rx = int(125 + 67 * math.cos(rad))
        _circle(c, ry, rx, 9, FOREST)
    return c, (125, 195), (125, 55)  # east to west, on the ring


def _map_three_islands() -> tuple[np.ndarray, tuple, tuple]:
    """Three islands in a chain. Must make two water crossings."""
    c = _canvas()
    _paint_island(c, 125, 45, 38, layers=(
        (1.00, BEACH), (0.82, GRASSLAND), (0.50, FOREST),
    ))
    _paint_island(c, 112, 125, 28, layers=(
        (1.00, BEACH), (0.78, GRASSLAND), (0.42, FOREST),
    ))
    _paint_island(c, 125, 205, 38, layers=(
        (1.00, BEACH), (0.82, GRASSLAND), (0.50, FOREST),
    ))
    return c, (125, 22), (125, 228)


def _map_peninsula() -> tuple[np.ndarray, tuple, tuple]:
    """Long thin peninsula. Walk its length or swim alongside (boat cost + water drain)."""
    c = _canvas()
    # Main body
    _paint_island(c, 125, 52, 52, layers=(
        (1.00, BEACH), (0.86, GRASSLAND), (0.52, FOREST),
    ))
    # Peninsula: narrow E-W strip rows 113-137
    _rect(c, 113, 95, 137, 228, GRASSLAND)
    _rect(c, 109, 95, 113, 228, BEACH)
    _rect(c, 137, 95, 141, 228, BEACH)
    # Forest checkpoints on peninsula
    for cx in [138, 170, 202]:
        _circle(c, 125, cx, 9, FOREST)
    return c, (125, 28), (125, 220)


def _map_fork_road() -> tuple[np.ndarray, tuple, tuple]:
    """Y-junction: grassland fork (fast, high drain) vs forest fork (slow, heals).
    Spawn south, target northwest. Tests resource-aware route selection."""
    c = _canvas()
    # Shared island base
    _paint_island(c, 105, 125, 108, layers=(
        (1.00, BEACH), (0.92, GRASSLAND),
    ))
    # Stem going south
    _rect(c, 170, 108, 228, 142, GRASSLAND)
    _rect(c, 170, 104, 228, 108, BEACH)
    _rect(c, 170, 142, 228, 146, BEACH)
    # West fork: all grassland (fast, drains resources)
    _rect(c, 45, 42, 172, 112, GRASSLAND)
    # East fork: all forest (slow, heals resources, same distance)
    _rect(c, 45, 138, 172, 208, FOREST)
    _ring(c, 45, 125, 0, 83, GRASSLAND)  # reconnect at top
    return c, (218, 125), (48, 75)  # south → northwest (grassland side)


def _map_desert_oasis() -> tuple[np.ndarray, tuple, tuple]:
    """Sandy island (high resource drain) with forest oases. Resource scarcity challenge."""
    c = _canvas()
    _paint_island(c, 125, 125, 105, layers=(
        (1.00, BEACH), (0.90, SANDY),
    ))
    # Forest oases: grid + center
    for cy, cx in [(75, 75), (75, 125), (75, 175),
                   (125, 62), (125, 125), (125, 188),
                   (175, 75), (175, 125), (175, 175)]:
        _circle(c, cy, cx, 13, FOREST)
    return c, (68, 68), (182, 182)


def _map_lake_crossing() -> tuple[np.ndarray, tuple, tuple]:
    """Island with large central lake. Circumnavigate north/south or swim across."""
    c = _canvas()
    _paint_island(c, 125, 125, 108, layers=(
        (1.00, BEACH), (0.93, GRASSLAND),
    ))
    # Forest ring around lake for resources
    _ring(c, 125, 125, 52, 68, FOREST)
    # Central lake
    _circle(c, 125, 125, 52, WATER)
    _circle(c, 125, 125, 38, DEEP_WATER)
    _circle(c, 125, 125, 24, OCEAN)
    return c, (125, 22), (125, 228)


def _map_archipelago() -> tuple[np.ndarray, tuple, tuple]:
    """Scattered island chain. Multi-hop water crossings required."""
    c = _canvas()
    islands = [
        (125, 25,  28),   # spawn island
        (88,  72,  22),
        (158, 78,  18),
        (108, 128, 24),
        (148, 172, 20),
        (88,  178, 18),
        (125, 225, 28),   # target island
    ]
    for cy, cx, r in islands:
        _paint_island(c, cy, cx, r, layers=(
            (1.00, BEACH), (0.78, GRASSLAND), (0.45, FOREST),
        ))
    return c, (125, 14), (125, 236)


def _map_mountain_dome() -> tuple[np.ndarray, tuple, tuple]:
    """Central mountain dome. Climb over (short, brutal cost + drain) or walk around."""
    c = _canvas()
    _paint_island(c, 125, 125, 108, layers=(
        (1.00, BEACH), (0.93, GRASSLAND),
    ))
    # Mountain dome
    _circle(c, 125, 125, 62, ROCKY)
    _circle(c, 125, 125, 50, MOUNTAINS)
    # Forest ring around mountain base
    _ring(c, 125, 125, 62, 76, FOREST)
    return c, (32, 125), (218, 125)


def _map_checkpoint_run() -> tuple[np.ndarray, tuple, tuple]:
    """Long narrow N-S corridor with forest recharge checkpoints every ~45 rows.
    Tests whether agent learns to use forest checkpoints strategically."""
    c = _canvas()
    _paint_island(c, 125, 125, 112, layers=(
        (1.00, BEACH), (0.93, GRASSLAND),
    ))
    # Narrow the island to a corridor (mask out east/west wings)
    _Y, X = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing='ij')
    c[np.abs(X - 125) > 42] = OCEAN
    c[(np.abs(X - 125) > 38) & (np.abs(X - 125) <= 42)] = BEACH
    # Forest checkpoints
    for row in [55, 98, 152, 197]:
        _circle(c, row, 125, 16, FOREST)
    return c, (26, 125), (224, 125)


def _map_two_bridges() -> tuple[np.ndarray, tuple, tuple]:
    """Two islands linked by two bridges: mountain (short) vs grassland (long).
    North bridge: mountain (cheap steps but huge resource drain).
    South bridge: grassland (longer but manageable drain)."""
    c = _canvas()
    _paint_island(c, 125, 58, 50, layers=(
        (1.00, BEACH), (0.83, GRASSLAND), (0.52, FOREST),
    ))
    _paint_island(c, 125, 192, 50, layers=(
        (1.00, BEACH), (0.83, GRASSLAND), (0.52, FOREST),
    ))
    # North bridge: mountains, rows 88-102, cols 106-144
    _rect(c, 88, 106, 102, 144, MOUNTAINS)
    _rect(c, 100, 106, 112, 144, ROCKY)
    # South bridge: grassland, rows 148-162, cols 106-144
    _rect(c, 148, 106, 162, 144, GRASSLAND)
    return c, (125, 22), (125, 228)


def _map_forest_detour() -> tuple[np.ndarray, tuple, tuple]:
    """Forest island slightly south of direct route.
    Direct path: straight grassland (resource drain). Detour south: forest to recharge.
    Tests resource-aware planning."""
    c = _canvas()
    _paint_island(c, 125, 125, 112, layers=(
        (1.00, BEACH), (0.92, GRASSLAND),
    ))
    # Forest island: south of direct route, around row 180
    _circle(c, 180, 125, 30, FOREST)
    _ring(c, 180, 125, 28, 36, BEACH)
    # Water channel between main island and forest island
    d_fi = _dist(180, 125)
    c[(d_fi >= 36) & (d_fi <= 48)] = WATER
    # Ensure main island is fully grassland along the direct path row 125
    _rect(c, 108, 22, 142, 228, GRASSLAND)
    return c, (125, 22), (125, 228)


def _map_rocky_coast() -> tuple[np.ndarray, tuple, tuple]:
    """Rocky coastline with grassland interior. Coastal vs inland routing."""
    c = _canvas()
    _paint_island(c, 125, 125, 105, layers=(
        (1.00, ROCKY),
        (0.85, GRASSLAND),
        (0.55, FOREST),
        (0.25, ROCKY),
        (0.12, MOUNTAINS),
    ))
    _ring(c, 125, 125, 100, 108, BEACH)
    return c, (48, 88), (202, 162)


def _map_grand_circuit() -> tuple[np.ndarray, tuple, tuple]:
    """Gauntlet: all terrain types in one map, long diagonal journey.
    SW zone: sandy desert. NE zone: forest. Central ridge: mountains.
    SE: lake obstacle."""
    c = _canvas()
    _paint_island(c, 125, 125, 112, layers=(
        (1.00, BEACH), (0.90, GRASSLAND),
    ))
    # SW quadrant → sandy
    Y, X = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing='ij')
    d = _dist(125, 125)
    sw = (Y > 138) & (X < 112) & (d < 108)
    c[sw] = SANDY
    # NE quadrant → forest
    ne = (Y < 112) & (X > 138) & (d < 108)
    c[ne] = FOREST
    # Diagonal mountain ridge NW→SE across center
    for i in range(50):
        ry, rx = 88 + i, 88 + i
        if 0 <= ry < SIZE and 0 <= rx < SIZE:
            _circle(c, ry, rx, 11, MOUNTAINS)
            _ring(c, ry, rx, 11, 17, ROCKY)
    # SE lake
    _circle(c, 175, 175, 22, WATER)
    _circle(c, 175, 175, 14, DEEP_WATER)
    # Oases in sandy zone
    for cy, cx in [(170, 58), (195, 92), (158, 82)]:
        _circle(c, cy, cx, 11, FOREST)
    return c, (200, 50), (50, 200)


# ── Registry ──────────────────────────────────────────────────────────────

_GENERATORS: dict[str, object] = {
    "open_grassland":  _map_open_grassland,
    "forest_wall":     _map_forest_wall,
    "mountain_pass":   _map_mountain_pass,
    "ring_island":     _map_ring_island,
    "three_islands":   _map_three_islands,
    "peninsula":       _map_peninsula,
    "fork_road":       _map_fork_road,
    "desert_oasis":    _map_desert_oasis,
    "lake_crossing":   _map_lake_crossing,
    "archipelago":     _map_archipelago,
    "mountain_dome":   _map_mountain_dome,
    "checkpoint_run":  _map_checkpoint_run,
    "two_bridges":     _map_two_bridges,
    "forest_detour":   _map_forest_detour,
    "rocky_coast":     _map_rocky_coast,
    "grand_circuit":   _map_grand_circuit,
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
