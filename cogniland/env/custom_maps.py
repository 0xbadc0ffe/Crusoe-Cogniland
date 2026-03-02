"""Hand-crafted maps with known dual-strategy structure for toy training experiments.

Each map is a 20×20 float32 tensor matching the world_map heightmap format.
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

_SIZE = 20

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
    mask = ((rr - cr) ** 2 / max(ra, 1) ** 2 + (cc_grid - cc) ** 2 / max(rb, 1) ** 2) <= 1.0
    arr[mask] = val


# ---------------------------------------------------------------------------
# Map 1 — "the_strait"
# ---------------------------------------------------------------------------

@_register("the_strait", spawn=(10, 2), target=(10, 18))
def _build_the_strait() -> np.ndarray:
    """Two land masses separated by a deep channel; shallow northern crossing.

    Strategy A: Cross the deep channel directly (short swim, needs resources).
    Strategy B: Detour north through shallow crossing (more walking, less drain).
    """
    arr = np.full((_SIZE, _SIZE), _GRASSLAND, dtype=np.float32)

    # Ocean border (1 cell)
    _fill(arr, 0, 1, 0, _SIZE, _OCEAN)
    _fill(arr, _SIZE - 1, _SIZE, 0, _SIZE, _OCEAN)
    _fill(arr, 0, _SIZE, 0, 1, _OCEAN)
    _fill(arr, 0, _SIZE, _SIZE - 1, _SIZE, _OCEAN)

    # Deep ocean channel (cols 8-12)
    _fill(arr, 0, _SIZE, 8, 13, _DEEP_WATER)

    # Shallow northern crossing (rows 1-3)
    _fill(arr, 1, 3, 8, 13, _WATER)

    # Forest patches for resources
    _fill(arr, 5, 15, 1, 7, _FOREST)    # left island
    _fill(arr, 5, 15, 13, 19, _FOREST)   # right island

    return arr


# ---------------------------------------------------------------------------
# Map 2 — "forest_belt"
# ---------------------------------------------------------------------------

@_register("forest_belt", spawn=(17, 10), target=(3, 10))
def _build_forest_belt() -> np.ndarray:
    """A wide forest belt blocks the direct north-south path.

    Strategy A: Go straight through the forest (slow but resource-positive).
    Strategy B: Go around via left/right grassland edge (faster but no resource gain).
    """
    arr = np.full((_SIZE, _SIZE), _GRASSLAND, dtype=np.float32)

    # Rocky strips at top and bottom
    _fill(arr, 0, 1, 0, _SIZE, _ROCKY)
    _fill(arr, 19, _SIZE, 0, _SIZE, _ROCKY)

    # Forest belt across the middle (rows 6-14, cols 2-18)
    _fill(arr, 6, 14, 2, 18, _FOREST)

    # Left and right edge corridors remain grassland (cols 0-1 and 18-19)
    return arr


# ---------------------------------------------------------------------------
# Map 3 — "twin_peaks"
# ---------------------------------------------------------------------------

@_register("twin_peaks", spawn=(10, 2), target=(10, 18))
def _build_twin_peaks() -> np.ndarray:
    """Two mountain masses with a rocky gap between them.

    Strategy A: Navigate the rocky gap (direct, high movement cost).
    Strategy B: Go around north or south via open grassland corridors.
    """
    arr = np.full((_SIZE, _SIZE), _GRASSLAND, dtype=np.float32)

    # Left mountain mass (rows 3-17, cols 5-9)
    _fill(arr, 3, 17, 5, 9, _MOUNTAINS)

    # Right mountain mass (rows 3-17, cols 11-15)
    _fill(arr, 3, 17, 11, 15, _MOUNTAINS)

    # Rocky gap between peaks (rows 7-13, cols 9-11)
    _fill(arr, 7, 13, 9, 11, _ROCKY)

    # Forest flanking the rocky gap
    _fill(arr, 8, 12, 3, 5, _FOREST)
    _fill(arr, 8, 12, 15, 17, _FOREST)

    return arr


# ---------------------------------------------------------------------------
# Map 4 — "river_delta"
# ---------------------------------------------------------------------------

@_register("river_delta", spawn=(2, 2), target=(18, 18))
def _build_river_delta() -> np.ndarray:
    """A diagonal river from top-left to bottom-right; two ford crossings.

    Strategy A: Cross at the first ford early — direct path.
    Strategy B: Follow the river to the second ford — more forest time.
    """
    arr = np.full((_SIZE, _SIZE), _GRASSLAND, dtype=np.float32)

    # Diagonal river: 1px wide, slope ≈ 1
    for r in range(_SIZE):
        c_center = r
        c0 = max(0, c_center - 1)
        c1 = min(_SIZE, c_center + 2)
        arr[r, c0:c1] = _WATER

    # Forest strips along both banks (1px each side)
    for r in range(_SIZE):
        c_center = r
        cl = max(0, c_center - 2)
        arr[r, cl:max(0, c_center - 1)] = _FOREST
        cr0 = min(_SIZE, c_center + 2)
        cr1 = min(_SIZE, c_center + 3)
        if cr0 < cr1:
            arr[r, cr0:cr1] = _FOREST

    # Ford 1: rows 5-6 — beach
    for r in range(5, 7):
        c_center = r
        c0 = max(0, c_center - 1)
        c1 = min(_SIZE, c_center + 2)
        arr[r, c0:c1] = _BEACH

    # Ford 2: rows 14-15 — beach
    for r in range(14, 16):
        c_center = r
        c0 = max(0, c_center - 1)
        c1 = min(_SIZE, c_center + 2)
        arr[r, c0:c1] = _BEACH

    return arr


# ---------------------------------------------------------------------------
# Map 5 — "archipelago"
# ---------------------------------------------------------------------------

@_register("archipelago", spawn=(5, 5), target=(16, 17))
def _build_archipelago() -> np.ndarray:
    """Four islands connected by land bridges across deep ocean.

    Strategy A: Follow land bridges island-to-island (safe, longer detour).
    Strategy B: Swim diagonally between islands (shorter, needs resource management).
    """
    arr = np.full((_SIZE, _SIZE), _DEEP_WATER, dtype=np.float32)

    # Island A (spawn): centre (5,5), semi-axes 3r×2c
    _ellipse(arr, 5, 5, 3, 2, _GRASSLAND)
    _ellipse(arr, 5, 5, 2, 1, _FOREST)

    # Island B: centre (9,10), semi-axes 3r×2c
    _ellipse(arr, 9, 10, 3, 2, _GRASSLAND)
    _ellipse(arr, 9, 10, 2, 1, _FOREST)

    # Island C: centre (13,14), semi-axes 2r×2c
    _ellipse(arr, 13, 14, 2, 2, _GRASSLAND)
    _ellipse(arr, 13, 14, 1, 1, _FOREST)

    # Island D (target): centre (16,17), semi-axes 2r×2c
    _ellipse(arr, 16, 17, 2, 2, _GRASSLAND)
    _ellipse(arr, 16, 17, 1, 1, _FOREST)

    # Land bridges (1px wide) via beach
    def _bridge(r0, c0, r1, c1, val=_BEACH, width=1):
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

    _bridge(5, 7, 9, 8)    # A → B
    _bridge(9, 12, 13, 12)  # B → C
    _bridge(13, 16, 16, 15) # C → D

    return arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_maps() -> list[str]:
    """Return all registered map names."""
    return list(_REGISTRY.keys())


def get_map(name: str) -> torch.Tensor:
    """Return the [20×20] world_map tensor for a named map."""
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
