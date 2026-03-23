"""Pathfinding on the island heightmap for eval path-efficiency metrics.

Uses scipy Dijkstra from a single source to compute distances to all cells.
One call per episode serves both directness (spawn→final position) and
survival margin computations.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Dijkstra (scipy-based)
# ---------------------------------------------------------------------------

def _build_grid_graph(
    wm: np.ndarray,
    thresholds_np: np.ndarray,
    costs_np: np.ndarray,
) -> "csr_matrix":
    """Build a 4-connected CSR sparse graph for Dijkstra.

    Edge cost = terrain cost of the **destination** cell.
    The graph is directed because cost(A→B) != cost(B→A) in general.
    """
    from scipy.sparse import csr_matrix

    H, W = wm.shape

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
    terrain_thresholds: torch.Tensor | None = None,
) -> np.ndarray:                # [H, W] float64, distance from source to every cell
    """Run scipy Dijkstra once from source. Returns full distance map.

    scipy releases the GIL so multiple calls can be parallelized via ThreadPoolExecutor.
    """
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

    H, W = world_map.shape
    wm = world_map.cpu().numpy()
    costs_np = terrain_costs.cpu().numpy()

    if terrain_thresholds is not None:
        thresholds_np = terrain_thresholds.cpu().numpy()
    else:
        # Fallback: caller must provide thresholds
        raise ValueError("terrain_thresholds is required")

    graph = _build_grid_graph(wm, thresholds_np, costs_np)

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
    terrain_thresholds: torch.Tensor | None = None,
    n_workers: int | None = None,
) -> list[np.ndarray]:          # list of B arrays, each [H, W]
    """Parallel Dijkstra across a batch using ThreadPoolExecutor.

    scipy releases the GIL, so threads genuinely parallelise the computation.
    Returns one full distance map per episode (from spawn to every cell), which
    serves directness (spawn→final pos) and other per-episode metrics.
    """
    from concurrent.futures import ThreadPoolExecutor

    B = sources.shape[0]
    args = [
        (world_maps[i].cpu(), terrain_costs.cpu(), sources[i].cpu(), terrain_thresholds.cpu())
        for i in range(B)
    ]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        dist_maps = list(pool.map(_run_dijkstra, args))

    return dist_maps
