"""Hand-scripted BFS solver — end-to-end solvability test for both variants.

* ``bt``: re-plan treating TREE as the only wall; cross water/rock by emitting
  PLACE/MINE when facing them.
* ``btc``: pick a tool by the map's category (lakes→build, rocky→mine, else
  whichever reaches the goal), then BFS treating only that obstacle as crossable;
  the first BUILD/MINE implicitly commits.
"""
from __future__ import annotations

from collections import deque

from .env import (
    A_BUILD, A_DOWN, A_LEFT, A_MINE, A_PLACE, A_RIGHT, A_UP,
    F_DOWN, F_LEFT, F_RIGHT, F_UP, BridgeTunnelEnv,
)
from .mapgen import _can_reach_goal
from .tiles import DIRT, GRASS, ROCK, SAND, TARGET, TREE, WATER, WOOD

_FACE_TO_MOVE = {F_UP: A_UP, F_DOWN: A_DOWN, F_LEFT: A_LEFT, F_RIGHT: A_RIGHT}
_WALK = (GRASS, WOOD, TARGET, SAND, DIRT)


def _delta_to_facing(dr, dc):
    if dr < 0:
        return F_UP
    if dr > 0:
        return F_DOWN
    if dc < 0:
        return F_LEFT
    return F_RIGHT


def _bfs_path(terrain, start, passable):
    """Shortest path start → any TARGET; ``passable(tile)`` decides traversal."""
    H, W = terrain.shape
    seen = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if terrain[r, c] == TARGET:
            path, cur = [], (r, c)
            while cur != start:
                path.append(cur); cur = seen[cur]
            return list(reversed(path))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen \
                    and passable(int(terrain[nr, nc])):
                seen[(nr, nc)] = (r, c); q.append((nr, nc))
    return None


def scripted_solve(env: BridgeTunnelEnv) -> tuple[int, bool]:
    """Run the solver to completion. Returns ``(steps_taken, reached_target)``."""
    if env.commit_enabled:
        cat = env._record.category
        build_ok = _can_reach_goal(env._terrain, env._pos, frozenset({WATER}))
        if cat == "rocky":
            cross_tile, cross_act = ROCK, A_MINE
        elif cat == "lakes" or build_ok:
            cross_tile, cross_act = WATER, A_BUILD
        else:
            cross_tile, cross_act = ROCK, A_MINE
        passable = lambda t: (t in _WALK) or (t == cross_tile)
    else:
        cross_tile = cross_act = None                 # cross whatever is faced
        passable = lambda t: t != TREE

    while env._step_count < env.max_steps:
        if env._terrain[env._pos] == TARGET:
            return env._step_count, True
        path = _bfs_path(env._terrain, env._pos, passable)
        if not path:
            return env._step_count, False
        nxt = path[0]
        face = _delta_to_facing(nxt[0] - env._pos[0], nxt[1] - env._pos[1])
        tile = int(env._terrain[nxt])
        if tile == WATER or tile == ROCK:             # an obstacle to cross
            act = cross_act if cross_act is not None else (A_PLACE if tile == WATER else A_MINE)
            env.step(_FACE_TO_MOVE[face] if env._facing != face else act)
        else:
            env.step(_FACE_TO_MOVE[face])
    return env._step_count, (env._terrain[env._pos] == TARGET)


__all__ = ["scripted_solve"]
