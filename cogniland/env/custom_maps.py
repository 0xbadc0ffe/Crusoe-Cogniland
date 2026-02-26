"""Hand-crafted maps with known dual-strategy structure for toy training experiments.

Each map is a 250×250 float32 tensor matching the world_map heightmap format.
Terrain encoding uses midpoint values from TERRAIN_LEVELS thresholds:

  ocean      < 0.007  → use 0.003
  deep_water < 0.025  → use 0.016
  water      < 0.05   → use 0.037
  beach      < 0.06   → use 0.055
  sandy      < 0.1    → use 0.080
  grassland  < 0.25   → use 0.150
  forest     < 0.6    → use 0.400
  rocky      < 0.7    → use 0.650
  mountains  < 1.0    → use 0.850
"""

from __future__ import annotations

import numpy as np
import torch

_SIZE = 250

# Terrain midpoint values
_OCEAN      = 0.003
_DEEP_WATER = 0.016
_WATER      = 0.037
_BEACH      = 0.055
_SANDY      = 0.080
_GRASSLAND  = 0.150
_FOREST     = 0.400
_ROCKY      = 0.650
_MOUNTAINS  = 0.850

# Registry: name → (world_map_fn, spawn, target)
_REGISTRY: dict[str, tuple] = {}


def _register(name: str, spawn: tuple[int, int], target: tuple[int, int]):
    """Decorator to register a map builder function."""
    def decorator(fn):
        _REGISTRY[name] = (fn, spawn, target)
        return fn
    return decorator


def _fill(arr: np.ndarray, r0: int, r1: int, c0: int, c1: int, val: float) -> None:
    """Fill rectangle [r0:r1, c0:c1] with val (end-exclusive)."""
    arr[r0:r1, c0:c1] = val


def _ellipse(arr: np.ndarray, cr: int, cc: int, ra: int, rb: int, val: float) -> None:
    """Fill an axis-aligned ellipse centred at (cr, cc) with semi-axes ra (rows), rb (cols)."""
    rr, cc_grid = np.ogrid[:_SIZE, :_SIZE]
    mask = ((rr - cr) ** 2 / ra ** 2 + (cc_grid - cc) ** 2 / rb ** 2) <= 1.0
    arr[mask] = val


# ---------------------------------------------------------------------------
# Map 1 — "the_strait"
# ---------------------------------------------------------------------------

@_register("the_strait", spawn=(125, 30), target=(125, 220))
def _build_the_strait() -> np.ndarray:
    """Two land masses separated by a deep channel; shallow northern crossing.

    Strategy A: Cross the deep channel directly (short swim, needs resources).
    Strategy B: Detour north through shallow crossing (more walking, less drain).
    """
    arr = np.full((_SIZE, _SIZE), _GRASSLAND, dtype=np.float32)

    # Ocean border
    _fill(arr, 0, 5, 0, _SIZE, _OCEAN)
    _fill(arr, _SIZE - 5, _SIZE, 0, _SIZE, _OCEAN)
    _fill(arr, 0, _SIZE, 0, 5, _OCEAN)
    _fill(arr, 0, _SIZE, _SIZE - 5, _SIZE, _OCEAN)

    # Deep ocean channel
    _fill(arr, 0, _SIZE, 100, 155, _DEEP_WATER)

    # Shallow northern crossing
    _fill(arr, 10, 35, 100, 155, _WATER)

    # Forest patches for resources
    _fill(arr, 60, 190, 10, 80, _FOREST)   # left island
    _fill(arr, 60, 190, 165, 235, _FOREST)  # right island

    return arr


# ---------------------------------------------------------------------------
# Map 2 — "forest_belt"
# ---------------------------------------------------------------------------

@_register("forest_belt", spawn=(210, 125), target=(40, 125))
def _build_forest_belt() -> np.ndarray:
    """A wide forest belt blocks the direct north-south path.

    Strategy A: Go straight through the forest (slow but resource-positive).
    Strategy B: Go around via left/right grassland edge (faster but no resource gain).
    """
    arr = np.full((_SIZE, _SIZE), _GRASSLAND, dtype=np.float32)

    # Rocky strips at very top and bottom
    _fill(arr, 0, 15, 0, _SIZE, _ROCKY)
    _fill(arr, 235, _SIZE, 0, _SIZE, _ROCKY)

    # Forest belt across the middle
    _fill(arr, 80, 170, 20, 230, _FOREST)

    # Left and right edge corridors remain grassland (already set by background)
    # cols 0-19 and 230-249 stay as grassland

    return arr


# ---------------------------------------------------------------------------
# Map 3 — "twin_peaks"
# ---------------------------------------------------------------------------

@_register("twin_peaks", spawn=(125, 20), target=(125, 230))
def _build_twin_peaks() -> np.ndarray:
    """Two mountain masses with a rocky gap between them.

    Strategy A: Navigate the rocky gap (direct, high movement cost).
    Strategy B: Go around north or south via open grassland corridors.
    """
    arr = np.full((_SIZE, _SIZE), _GRASSLAND, dtype=np.float32)

    # Left mountain mass
    _fill(arr, 40, 210, 60, 115, _MOUNTAINS)

    # Right mountain mass
    _fill(arr, 40, 210, 135, 190, _MOUNTAINS)

    # Rocky gap between peaks
    _fill(arr, 90, 165, 115, 135, _ROCKY)

    # Forest flanking the rocky gap (resources for mountain crossing)
    _fill(arr, 100, 155, 40, 60, _FOREST)
    _fill(arr, 100, 155, 190, 210, _FOREST)

    # Northern and southern corridors are open grassland (already set)
    # rows 0-35 and 215-250 span full width

    return arr


# ---------------------------------------------------------------------------
# Map 4 — "river_delta"
# ---------------------------------------------------------------------------

@_register("river_delta", spawn=(30, 30), target=(220, 220))
def _build_river_delta() -> np.ndarray:
    """A diagonal river from top-left to bottom-right; two ford crossings.

    Strategy A: Cross at the first ford (~row 70) early — direct path.
    Strategy B: Follow the river to the second ford (~row 180) — more forest time.
    """
    arr = np.full((_SIZE, _SIZE), _GRASSLAND, dtype=np.float32)

    # Diagonal river: band ~15px wide, slope ≈ 1
    for r in range(_SIZE):
        c_center = r  # diagonal from (0,0) to (249,249)
        c0 = max(0, c_center - 7)
        c1 = min(_SIZE, c_center + 8)
        arr[r, c0:c1] = _WATER

    # Forest strips along both banks (~8px wide each side)
    for r in range(_SIZE):
        c_center = r
        # Left bank
        c0 = max(0, c_center - 16)
        c1 = max(0, c_center - 7)
        arr[r, c0:c1] = _FOREST
        # Right bank
        c0 = min(_SIZE, c_center + 8)
        c1 = min(_SIZE, c_center + 17)
        arr[r, c0:c1] = _FOREST

    # Ford 1: rows 60-80 — beach (easier crossing)
    for r in range(60, 80):
        c_center = r
        c0 = max(0, c_center - 7)
        c1 = min(_SIZE, c_center + 8)
        arr[r, c0:c1] = _BEACH

    # Ford 2: rows 170-190 — beach
    for r in range(170, 190):
        c_center = r
        c0 = max(0, c_center - 7)
        c1 = min(_SIZE, c_center + 8)
        arr[r, c0:c1] = _BEACH

    return arr


# ---------------------------------------------------------------------------
# Map 5 — "archipelago"
# ---------------------------------------------------------------------------

@_register("archipelago", spawn=(60, 60), target=(200, 210))
def _build_archipelago() -> np.ndarray:
    """Four islands connected by land bridges across deep ocean.

    Strategy A: Follow land bridges island-to-island (safe, longer detour).
    Strategy B: Swim diagonally between islands (shorter, needs resource management).
    """
    arr = np.full((_SIZE, _SIZE), _DEEP_WATER, dtype=np.float32)

    # Island A (spawn): centre (60,60), semi-axes 40r×30c
    _ellipse(arr, 60, 60, 40, 30, _GRASSLAND)
    _ellipse(arr, 60, 60, 25, 18, _FOREST)

    # Island B: centre (110,130), semi-axes 35r×25c
    _ellipse(arr, 110, 130, 35, 25, _GRASSLAND)
    _ellipse(arr, 110, 130, 20, 14, _FOREST)

    # Island C: centre (165,175), semi-axes 30r×30c
    _ellipse(arr, 165, 175, 30, 30, _GRASSLAND)
    _ellipse(arr, 165, 175, 17, 17, _FOREST)

    # Island D (target): centre (200,210), semi-axes 35r×25c
    _ellipse(arr, 200, 210, 35, 25, _GRASSLAND)
    _ellipse(arr, 200, 210, 20, 14, _FOREST)

    # Land bridges (8px wide) via shallow water/beach
    def _bridge(r0, c0, r1, c1, val=_BEACH, width=8):
        """Draw a rectangular land bridge between two points."""
        steps = max(abs(r1 - r0), abs(c1 - c0))
        if steps == 0:
            return
        for s in range(steps + 1):
            t = s / steps
            r = int(round(r0 + t * (r1 - r0)))
            c = int(round(c0 + t * (c1 - c0)))
            hw = width // 2
            arr[max(0, r - hw):min(_SIZE, r + hw + 1),
                max(0, c - hw):min(_SIZE, c + hw + 1)] = val

    _bridge(60, 80, 110, 105)   # A → B
    _bridge(110, 155, 165, 150)  # B → C
    _bridge(165, 200, 200, 185)  # C → D

    return arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_maps() -> list[str]:
    """Return all registered map names."""
    return list(_REGISTRY.keys())


def get_map(name: str) -> torch.Tensor:
    """Return the [250×250] world_map tensor for a named map."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown map '{name}'. Available: {list_maps()}")
    fn, _spawn, _target = _REGISTRY[name]
    arr = fn()
    return torch.from_numpy(arr)


def get_spawn(name: str) -> tuple[int, int]:
    """Return the (row, col) spawn position for a named map."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown map '{name}'. Available: {list_maps()}")
    _fn, spawn, _target = _REGISTRY[name]
    return spawn


def get_target(name: str) -> tuple[int, int]:
    """Return the (row, col) target position for a named map."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown map '{name}'. Available: {list_maps()}")
    _fn, _spawn, target = _REGISTRY[name]
    return target
