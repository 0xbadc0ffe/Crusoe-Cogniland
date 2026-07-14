"""Min-action cost-to-go (Dijkstra) used by the PBRS potential φ = −ctg.

Shared by both variants of the env:

* ``bt``  uses a single field with both obstacles crossable.
* ``btc`` uses a 3-field stack indexed by the live commitment (none / build /
  mine): the non-committed obstacle becomes an impassable wall, and the field is
  **capped** at ``2·(H+W)`` so the per-step term ``(1−γ)·ctg`` on the (now common)
  unreachable cells stays bounded.
"""
from __future__ import annotations

import heapq

import numpy as np

from .tiles import ROCK, TARGET, TREE, WATER


def compute_ctg(terrain: np.ndarray, target: tuple[int, int], *,
                water_cross: bool = True, rock_cross: bool = True,
                cap: int | None = None, seeds=None) -> np.ndarray:
    """Entering walkable land costs 1; entering a *crossable* obstacle costs 2
    (build/mine + the move); TREE and any non-crossable obstacle are walls.
    Seeded from every TARGET cell by default; pass ``seeds`` (an iterable of
    ``(r, c)``) to seed from a specific subset instead — e.g. the fork_wall
    task's *correct* door only, so the PBRS potential doesn't pull the agent
    toward the decoy door. Unreachable → ``INF = H·W·4`` (then clamped to
    ``cap`` if given)."""
    H, W = terrain.shape
    INF = H * W * 4
    dist = np.full((H, W), INF, dtype=np.int32)
    seeds = list(map(tuple, seeds)) if seeds is not None \
        else list(map(tuple, np.argwhere(terrain == TARGET)))
    if not seeds:
        seeds = [tuple(target)]
    pq = []
    for (tr, tc) in seeds:
        dist[tr, tc] = 0
        heapq.heappush(pq, (0, tr, tc))
    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            t = terrain[nr, nc]
            if t == TREE:
                continue
            if t == WATER:
                if not water_cross:
                    continue
                step = 2
            elif t == ROCK:
                if not rock_cross:
                    continue
                step = 2
            else:
                step = 1
            nd = d + step
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(pq, (nd, nr, nc))
    if cap is not None:
        dist = np.minimum(dist, int(cap))
    return dist


def commit_ctg_stack(terrain: np.ndarray, target: tuple[int, int],
                     cap: int | None = None, seeds=None) -> np.ndarray:
    """(3, H, W) float32 commitment-indexed fields: [none, build, mine]."""
    H, W = terrain.shape
    if cap is None:
        cap = 2 * (H + W)
    none = compute_ctg(terrain, target, water_cross=True, rock_cross=True, cap=cap, seeds=seeds)
    build = compute_ctg(terrain, target, water_cross=True, rock_cross=False, cap=cap, seeds=seeds)
    mine = compute_ctg(terrain, target, water_cross=False, rock_cross=True, cap=cap, seeds=seeds)
    return np.stack([none, build, mine], axis=0).astype(np.float32)


__all__ = ["compute_ctg", "commit_ctg_stack"]
