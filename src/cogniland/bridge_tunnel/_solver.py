"""Hand-scripted BFS solver for the bridge_tunnel env.

Used as an end-to-end *solvability test* — proves a deterministic policy can
reach the target from the env's natural maps using only the in-game actions
(move / mine / place). The solver re-plans every step, treating TREE as
the only absolutely-impassable tile; rock and water are crossed by emitting
the corresponding MINE / PLACE action when the agent faces them.
"""
from __future__ import annotations

from collections import deque

from .env import (
    A_DOWN, A_LEFT, A_MINE, A_PLACE, A_RIGHT, A_UP,
    F_DOWN, F_LEFT, F_RIGHT, F_UP, BridgeTunnelEnv,
)
from .tiles import ROCK, TREE, WATER

_FACE_TO_MOVE = {F_UP: A_UP, F_DOWN: A_DOWN, F_LEFT: A_LEFT, F_RIGHT: A_RIGHT}


def _delta_to_facing(dr: int, dc: int) -> int:
    if dr < 0:
        return F_UP
    if dr > 0:
        return F_DOWN
    if dc < 0:
        return F_LEFT
    return F_RIGHT


def _bfs_path(terrain, start, goal):
    H, W = terrain.shape
    if start == goal:
        return []
    seen = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            path = []
            cur = goal
            while cur != start:
                path.append(cur)
                cur = seen[cur]
            return list(reversed(path))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen \
                    and terrain[nr, nc] != TREE:
                seen[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return None


def scripted_solve(env: BridgeTunnelEnv) -> tuple[int, bool]:
    """Run the BFS-replanning solver to completion (or max_steps). Returns
    ``(steps_taken, reached_target)``."""
    while env._step_count < env.max_steps:
        if env._pos == env._record.target:
            return env._step_count, True
        path = _bfs_path(env._terrain, env._pos, env._record.target)
        if not path:
            return env._step_count, False
        next_cell = path[0]
        dr = next_cell[0] - env._pos[0]
        dc = next_cell[1] - env._pos[1]
        face = _delta_to_facing(dr, dc)
        tile = int(env._terrain[next_cell])
        if tile == WATER:
            if env._facing != face:                     # turn first (blocked move)
                env.step(_FACE_TO_MOVE[face])
            else:
                env.step(A_PLACE)
        elif tile == ROCK:
            if env._facing != face:
                env.step(_FACE_TO_MOVE[face])
            else:
                env.step(A_MINE)
        else:                                           # walkable → just move
            env.step(_FACE_TO_MOVE[face])
    return env._step_count, (env._pos == env._record.target)


__all__ = ["scripted_solve"]
