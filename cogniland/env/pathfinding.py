"""A* pathfinding on the island heightmap for eval path-efficiency metrics."""

from __future__ import annotations

import heapq
import math

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
