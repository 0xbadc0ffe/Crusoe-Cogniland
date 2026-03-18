"""Pathfinding on the island heightmap for eval path-efficiency metrics.

Two implementations:
- ``batch_astar``: original sequential Python heapq A* (kept for backward compat).
- ``batch_dijkstra_from_sources``: faster scipy Dijkstra, parallelized with threads.
  Each call computes distances from ONE source to ALL cells, so a single call per
  episode serves both path_efficiency (spawn→target) and directness (spawn→final).
"""

from __future__ import annotations

import heapq
import math

import numpy as np
import torch

from cogniland.env.constants import TERRAIN_THRESHOLDS, TERRAIN_COSTS


def _terrain_level(value: float, thresholds: list[float]) -> int:
    """Return terrain level index for a given heightmap value."""
    for i, t in enumerate(thresholds):
        if value < t:
            return i
    return len(thresholds) - 1


def astar_shortest_path(
    world_map: torch.Tensor,
    terrain_costs: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
) -> float:
    """A* on the grid using terrain movement costs as edge weights.

    Uses 4-connected grid (up/down/left/right) matching the agent's action space.
    Heuristic: L2 distance * min_terrain_cost (admissible).

    Args:
        world_map: [H, W] heightmap tensor.
        terrain_costs: [9] per-terrain-level movement costs.
        start: [2] (row, col) start position.
        goal: [2] (row, col) goal position.

    Returns:
        Total movement cost of the optimal path, or -1.0 if unreachable.
    """
    wm = world_map.cpu().numpy()
    thresholds = TERRAIN_THRESHOLDS.cpu().tolist()
    costs = terrain_costs.cpu().tolist()
    min_cost = min(costs)

    H, W = wm.shape
    sr, sc = int(start[0].item()), int(start[1].item())
    gr, gc = int(goal[0].item()), int(goal[1].item())

    if sr == gr and sc == gc:
        return 0.0

    # heuristic: L2 * min_cost (admissible)
    def h(r: int, c: int) -> float:
        return math.sqrt((r - gr) ** 2 + (c - gc) ** 2) * min_cost

    # (f, g, row, col)
    open_set: list[tuple[float, float, int, int]] = []
    heapq.heappush(open_set, (h(sr, sc), 0.0, sr, sc))
    g_best = {(sr, sc): 0.0}

    deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    while open_set:
        f, g, r, c = heapq.heappop(open_set)

        if r == gr and c == gc:
            return g

        # Skip if we already found a better path to this node
        if g > g_best.get((r, c), float("inf")):
            continue

        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                level = _terrain_level(wm[nr, nc], thresholds)
                edge_cost = costs[level]
                ng = g + edge_cost
                if ng < g_best.get((nr, nc), float("inf")):
                    g_best[(nr, nc)] = ng
                    heapq.heappush(open_set, (ng + h(nr, nc), ng, nr, nc))

    return -1.0


def batch_astar(
    world_map: torch.Tensor,
    terrain_costs: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
) -> torch.Tensor:
    """Run A* for each (start, goal) pair.

    Args:
        world_map: [H, W] shared or [B, H, W] per-env heightmap.
        terrain_costs: [9] costs per terrain level.
        starts: [B, 2] start positions.
        goals: [B, 2] goal positions.

    Returns:
        [B] tensor of optimal path costs (-1.0 for unreachable pairs).
    """
    B = starts.shape[0]
    per_env = world_map.dim() == 3
    results = torch.zeros(B)
    for i in range(B):
        wm_i = world_map[i] if per_env else world_map
        results[i] = astar_shortest_path(wm_i, terrain_costs, starts[i], goals[i])
    return results


# ---------------------------------------------------------------------------
# Dijkstra (scipy-based, ~10-50× faster than Python heapq A*)
# ---------------------------------------------------------------------------

def _build_grid_graph(wm: np.ndarray, costs_np: np.ndarray) -> "csr_matrix":
    """Build a 4-connected CSR sparse graph for Dijkstra.

    Edge cost = terrain cost of the **destination** cell (matching A* convention).
    The graph is directed because cost(A→B) != cost(B→A) in general.
    """
    from scipy.sparse import csr_matrix

    H, W = wm.shape
    thresholds_np = TERRAIN_THRESHOLDS.cpu().numpy()

    # Vectorized terrain level assignment: level = first threshold index where wm < threshold
    terrain = np.searchsorted(thresholds_np, wm.ravel(), side="left").reshape(H, W)
    terrain = np.clip(terrain, 0, len(thresholds_np) - 1).astype(np.int32)

    # Horizontal edges: (r,c) → (r,c+1) with cost of destination
    r_h, c_h = np.mgrid[0:H, 0:W-1]
    src_h = (r_h * W + c_h).ravel()
    dst_h = (r_h * W + c_h + 1).ravel()
    cost_h_fwd = costs_np[terrain[r_h, c_h + 1]].ravel()   # → right: cost of (r, c+1)
    cost_h_bwd = costs_np[terrain[r_h, c_h]].ravel()        # ← left:  cost of (r, c)

    # Vertical edges: (r,c) → (r+1,c) with cost of destination
    r_v, c_v = np.mgrid[0:H-1, 0:W]
    src_v = (r_v * W + c_v).ravel()
    dst_v = ((r_v + 1) * W + c_v).ravel()
    cost_v_fwd = costs_np[terrain[r_v + 1, c_v]].ravel()   # ↓ down: cost of (r+1, c)
    cost_v_bwd = costs_np[terrain[r_v, c_v]].ravel()        # ↑ up:   cost of (r, c)

    all_src  = np.concatenate([src_h, dst_h, src_v, dst_v])
    all_dst  = np.concatenate([dst_h, src_h, dst_v, src_v])
    all_data = np.concatenate([cost_h_fwd, cost_h_bwd, cost_v_fwd, cost_v_bwd])

    N = H * W
    return csr_matrix((all_data, (all_src, all_dst)), shape=(N, N))


def dijkstra_from_source(
    world_map: torch.Tensor,    # [H, W] CPU
    terrain_costs: torch.Tensor,
    source: torch.Tensor,       # [2] long
) -> np.ndarray:                # [H, W] float64, distance from source to every cell
    """Run scipy Dijkstra once from source. Returns full distance map.

    ~10-50× faster than Python heapq A*. scipy releases the GIL so multiple
    calls can be parallelized via ThreadPoolExecutor.
    """
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

    H, W = world_map.shape
    wm = world_map.cpu().numpy()
    costs_np = terrain_costs.cpu().numpy()

    graph = _build_grid_graph(wm, costs_np)

    sr, sc = int(source[0].item()), int(source[1].item())
    source_idx = sr * W + sc
    dist = scipy_dijkstra(graph, directed=True, indices=source_idx)  # [H*W]
    return dist.reshape(H, W)


def _run_dijkstra(args: tuple) -> np.ndarray:
    """Unpacking helper so pool.map can call dijkstra_from_source."""
    return dijkstra_from_source(*args)


def batch_dijkstra_from_sources(
    world_maps: torch.Tensor,   # [B, H, W]
    terrain_costs: torch.Tensor,
    sources: torch.Tensor,      # [B, 2]
    n_workers: int | None = None,
) -> list[np.ndarray]:          # list of B arrays, each [H, W]
    """Parallel Dijkstra across a batch using ThreadPoolExecutor.

    scipy releases the GIL, so threads genuinely parallelise the computation.
    Returns one full distance map per episode (from spawn to every cell), which
    serves both path_efficiency (spawn→target) and directness (spawn→final pos).
    """
    from concurrent.futures import ThreadPoolExecutor

    B = sources.shape[0]
    args = [
        (world_maps[i].cpu(), terrain_costs.cpu(), sources[i].cpu())
        for i in range(B)
    ]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        dist_maps = list(pool.map(_run_dijkstra, args))

    return dist_maps
