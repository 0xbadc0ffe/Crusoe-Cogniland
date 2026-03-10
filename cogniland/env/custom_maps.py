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
    """3-tile mountain column next to spawn hides a single forest tile behind it.

    The column at rows 9-11, col 1 is visible from spawn (distance ≤ 1.5) and
    points directly opposite the target.  It fully occludes (10,0) from every
    col≥2 vantage point via Bresenham raycasting:

        (10,2) spawn   → ray hits (10,1) = mountain ✓
        (9,2)  N-path  → ray hits (10,1) = mountain ✓
        (8,2)  N-path  → ray hits (9,1)  = mountain ✓  ← requires row 9
        (11,2) S-path  → ray hits (11,1) = mountain ✓
        (12,2) S-path  → ray hits (11,1) = mountain ✓  ← requires row 11
        (7,2)+ / (13,2)+ → distance > 3, invisible anyway

    The bypass is placed in the SOUTH (rows 17-19) so the sub-optimal agent
    goes south and never enters the visibility range of the hidden forest.
    The optimal agent explores SOUTH-WEST around the mountain base, discovers
    the forest at (12,0), wades 1 free step through the moat to (10,0), and
    collects resources before swimming.

    Both paths are exactly 30 steps; only resource management differs:

        Sub-optimal (south beach bypass, 30 steps):
            resources = 0 throughout → r_resource = −0.02×25×30 = −15.0
            total reward ≈ 12 − 3.0 − 15.0 = −6.0

        Optimal (forest detour + swim, 30 steps):
            resources peak at 9, exit swim with 3
            r_resource ≈ −11.6
            total reward ≈ 12 − 3.0 − 11.6 = −2.6

    Swimming without resources is fatal: the 6-wide deep-water channel (rows
    0-16, cols 7-12) deals 25 HP on steps 4-6, exhausting 75 HP exactly.

    Map layout:
        rows  9-11  col 1        = MOUNTAINS  (3-tile column; full visual wall)
        (10,0)                   = FOREST     (single tile; visible from (12,0))
        (9,0), (11,0)            = WATER      (shallow moat; 1 free water step)
        rows  0-16  cols 7-12   = DEEP_WATER  (fatal without resources)
        rows 17-19  cols 7-12   = BEACH       (sub-optimal south bypass)
    """
    arr = np.full((_SIZE, _SIZE), _GRASSLAND, dtype=np.float32)

    # Mountain column: 3 tiles, directly west of spawn — full visual wall
    _fill(arr, 9, 12, 1, 2, _MOUNTAINS)

    # Single forest tile hidden behind the column
    arr[10, 0] = _FOREST

    # Shallow-water moat surrounding the forest (north + south; east=mountain, west=edge)
    arr[9, 0]  = _WATER
    arr[11, 0] = _WATER

    # Deep-water channel: 6 tiles wide, full height except south bypass — fatal without resources
    _fill(arr, 0, 17, 7, 13, _DEEP_WATER)

    # Beach south bypass: safe crossing but equally long and resource-barren (sub-optimal)
    _fill(arr, 17, _SIZE, 7, 13, _BEACH)

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

@_register("river_delta", spawn=(10, 5), target=(18, 5))
def _build_river_delta() -> np.ndarray:
    """Horizontal deep-water river (rows 12-17) divides spawn (N) from target (S).

    Forest patch (rows 6-9, cols 1-4) lies north-west of spawn — the agent must
    backtrack to collect resources before attempting to swim.

    Sub-optimal strategy (local minima): detour right to the land bridge
    (cols 18-19) and walk around — reachable but ~34 steps with 0 resources
    throughout → heavy resource-deficit penalty → reward ≈ −5.6.

    Optimal strategy: walk back to forest (~6 steps), stay 3 turns to gain
    9 resources (+6 HP), return to spawn column, then swim straight south
    through the 6-row deep-water channel (3 free + 3 paid at 2 res each,
    exits with 3 resources remaining) → ~24 steps → reward ≈ +3.1.

    Swimming without resources is fatal: steps 4-6 deal 25 HP each,
    exhausting the starting 75 HP exactly on the 6th water tile.

    Map layout (cols 18-19 are grassland land-bridge throughout):
        rows  0-11  grassland (forest patch rows 6-9, cols 1-4)
        rows 12-17  DEEP_WATER cols 0-17  |  grassland cols 18-19
        row  18     grassland (target at col 5)
    """
    arr = np.full((_SIZE, _SIZE), _GRASSLAND, dtype=np.float32)

    # Deep-water river: 6 rows wide — fatal without resources (cols 0-17)
    _fill(arr, 12, 18, 0, 18, _DEEP_WATER)

    # Forest resource patch north-west of spawn (covers ~(7,2))
    _fill(arr, 6, 10, 1, 5, _FOREST)

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
