"""Pathfinding on the island heightmap.

Two graph types:
1. **Move-cost graph** — edge cost = terrain move_cost of destination.
   Used by eval metrics (path efficiency) and the time-ratio in the reward.
2. **Reward cost-to-go graph** — edge cost = τ(dest) + β_raft × 1_{land→water}.
   Used for the Dijkstra progress signal J_t in the reward function.
   Computed as a *reverse* Dijkstra from the target so that J(cell) = cost-to-go.

Both use scipy Dijkstra (releases the GIL → parallelisable via threads).
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Graph builders
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


def _build_reward_graph(
    wm: np.ndarray,
    thresholds_np: np.ndarray,
    costs_np: np.ndarray,
    res_rates_np: np.ndarray,      # [num_terrains] signed resource rate (negative = drain)
    is_water: np.ndarray,          # [num_terrains] bool
    beta_raft: float,
) -> "csr_matrix":
    """Build graph with c(s→s') = τ(s') − res_rate(s') + β_raft × 1_{land→water}.

    Edge cost = move_cost − res_rate: draining terrains become more expensive,
    resource-gaining terrains (forest) become cheaper.
    Used for the Dijkstra cost-to-go progress signal in the reward function.
    """
    from scipy.sparse import csr_matrix

    H, W = wm.shape

    terrain = np.searchsorted(thresholds_np, wm.ravel(), side="left").reshape(H, W)
    terrain = np.clip(terrain, 0, len(thresholds_np) - 1).astype(np.int32)

    water_mask = is_water[terrain]  # [H, W] bool

    # Combined edge base cost: move_cost − res_rate per terrain
    combined_np = costs_np - res_rates_np  # [num_terrains]

    # Horizontal edges
    r_h, c_h = np.mgrid[0:H, 0:W-1]
    src_h = (r_h * W + c_h).ravel()
    dst_h = (r_h * W + c_h + 1).ravel()
    # Forward: (r,c) → (r,c+1)
    cost_h_fwd = combined_np[terrain[r_h, c_h + 1]].ravel()
    raft_h_fwd = (~water_mask[r_h, c_h] & water_mask[r_h, c_h + 1]).ravel().astype(np.float64) * beta_raft
    # Backward: (r,c+1) → (r,c)
    cost_h_bwd = combined_np[terrain[r_h, c_h]].ravel()
    raft_h_bwd = (~water_mask[r_h, c_h + 1] & water_mask[r_h, c_h]).ravel().astype(np.float64) * beta_raft

    # Vertical edges
    r_v, c_v = np.mgrid[0:H-1, 0:W]
    src_v = (r_v * W + c_v).ravel()
    dst_v = ((r_v + 1) * W + c_v).ravel()
    cost_v_fwd = combined_np[terrain[r_v + 1, c_v]].ravel()
    raft_v_fwd = (~water_mask[r_v, c_v] & water_mask[r_v + 1, c_v]).ravel().astype(np.float64) * beta_raft
    cost_v_bwd = combined_np[terrain[r_v, c_v]].ravel()
    raft_v_bwd = (~water_mask[r_v + 1, c_v] & water_mask[r_v, c_v]).ravel().astype(np.float64) * beta_raft

    all_src  = np.concatenate([src_h, dst_h, src_v, dst_v])
    all_dst  = np.concatenate([dst_h, src_h, dst_v, src_v])
    all_data = np.concatenate([
        cost_h_fwd + raft_h_fwd, cost_h_bwd + raft_h_bwd,
        cost_v_fwd + raft_v_fwd, cost_v_bwd + raft_v_bwd,
    ])

    N = H * W
    return csr_matrix((all_data, (all_src, all_dst)), shape=(N, N))


# ---------------------------------------------------------------------------
# Dijkstra runners
# ---------------------------------------------------------------------------

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
    Returns one full distance map per episode (from spawn to every cell).
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


# ---------------------------------------------------------------------------
# Reverse Dijkstra for reward cost-to-go
# ---------------------------------------------------------------------------

def _reverse_dijkstra_single(args: tuple) -> np.ndarray:
    """Compute cost-to-go from every cell to a single target.

    Runs Dijkstra from the target on the *transposed* graph so that
    dist[r, c] = optimal cost from cell (r, c) to the target.
    """
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

    wm_np, thresholds_np, costs_np, res_rates_np, is_water_np, beta_raft, target_rc = args

    H, W = wm_np.shape
    graph = _build_reward_graph(wm_np, thresholds_np, costs_np, res_rates_np, is_water_np, beta_raft)

    tr, tc = int(target_rc[0]), int(target_rc[1])
    target_idx = tr * W + tc
    # Transpose: Dijkstra from target on G^T gives cost-to-go in G
    dist = scipy_dijkstra(graph.T, directed=True, indices=target_idx)  # [H*W]
    return dist.reshape(H, W)


def batch_reverse_dijkstra(
    world_maps: torch.Tensor,       # [B, H, W] CPU
    terrain_costs: torch.Tensor,    # [num_terrains]
    terrain_thresholds: torch.Tensor,
    is_water: torch.Tensor,         # [num_terrains] bool
    targets: torch.Tensor,          # [B, 2] long
    beta_raft: float,
    res_rates: torch.Tensor | None = None,  # [num_terrains] signed resource rate
    n_workers: int | None = None,
) -> list[np.ndarray]:
    """Parallel reverse Dijkstra: cost-to-go from every cell to each target.

    Edge cost = move_cost − res_rate + β_raft × 1_{land→water}.

    Returns list of B arrays, each [H, W] float64.
    """
    from concurrent.futures import ThreadPoolExecutor

    B = targets.shape[0]
    costs_np = terrain_costs.cpu().numpy()
    thresholds_np = terrain_thresholds.cpu().numpy()
    is_water_np = is_water.cpu().numpy()
    res_rates_np = res_rates.cpu().numpy() if res_rates is not None else np.zeros_like(costs_np)

    args = [
        (world_maps[i].cpu().numpy(), thresholds_np, costs_np, res_rates_np, is_water_np, beta_raft,
         (targets[i, 0].item(), targets[i, 1].item()))
        for i in range(B)
    ]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        dist_maps = list(pool.map(_reverse_dijkstra_single, args))

    return dist_maps


# ---------------------------------------------------------------------------
# Dijkstra path reconstruction (for eval metrics)
# ---------------------------------------------------------------------------

def reconstruct_dijkstra_path(
    dist_map: np.ndarray,       # [H, W] distance from source to every cell
    source: tuple[int, int],    # (row, col) start
    target: tuple[int, int],    # (row, col) goal
) -> set[tuple[int, int]]:
    """Trace the shortest path from source to target using a forward distance map.

    Greedy backtrack from target: at each cell, step to the 4-connected
    neighbor with the smallest dist_from_source value.  Returns the set of
    (row, col) cells on the path (including source and target).
    """
    H, W = dist_map.shape
    DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    path: set[tuple[int, int]] = set()
    r, c = target
    path.add((r, c))

    while (r, c) != source:
        best_d, best_rc = float("inf"), (r, c)
        for dr, dc in DELTAS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and dist_map[nr, nc] < best_d:
                best_d = dist_map[nr, nc]
                best_rc = (nr, nc)
        if best_rc == (r, c):
            break  # unreachable — shouldn't happen with valid maps
        r, c = best_rc
        path.add((r, c))

    return path
