"""Hand-scripted solver for bridge_tunnel_commit — an end-to-end solvability test.

Proves a deterministic policy can win using only the in-game actions: it first
COMMITs to the tool that makes the goal reachable (preferring the category's
intended tool), then BFS-replans toward the goal, emitting the committed
BUILD/MINE action when it faces its crossable obstacle and walking around the
other (now-impassable) obstacle and trees.
"""
from __future__ import annotations

from collections import deque

from .env import (
    A_BUILD, A_COMMIT_BUILD, A_COMMIT_MINE, A_DOWN, A_LEFT, A_MINE, A_RIGHT,
    A_UP, COMMIT_BUILD, COMMIT_MINE, F_DOWN, F_LEFT, F_RIGHT, F_UP,
    BridgeTunnelCommitEnv,
)
from .mapgen import _can_reach_goal
from .tiles import GRASS, ROCK, SAND, DIRT, TARGET, TREE, WATER, WOOD

_FACE_TO_MOVE = {F_UP: A_UP, F_DOWN: A_DOWN, F_LEFT: A_LEFT, F_RIGHT: A_RIGHT}
_WALK = (GRASS, WOOD, TARGET, SAND, DIRT)


def _delta_to_facing(dr: int, dc: int) -> int:
    if dr < 0:
        return F_UP
    if dr > 0:
        return F_DOWN
    if dc < 0:
        return F_LEFT
    return F_RIGHT


def _bfs_path(terrain, start, cross_tile):
    """Shortest path from ``start`` to ANY TARGET cell, treating walkable tiles
    plus ``cross_tile`` as passable (everything else a wall)."""
    H, W = terrain.shape
    seen = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if terrain[r, c] == TARGET:
            path = []
            cur = (r, c)
            while cur != start:
                path.append(cur)
                cur = seen[cur]
            return list(reversed(path))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen:
                t = int(terrain[nr, nc])
                if t in _WALK or t == cross_tile:
                    seen[(nr, nc)] = (r, c)
                    q.append((nr, nc))
    return None


def scripted_solve(env: BridgeTunnelCommitEnv) -> tuple[int, bool]:
    """Run the commit-then-BFS solver to completion. Returns
    ``(steps_taken, reached_target)``."""
    terr = env._terrain
    spawn = env._pos
    # choose a tool that makes the goal reachable; prefer the intended one
    build_ok = _can_reach_goal(terr, spawn, frozenset({WATER}))
    cat = env._record.category
    if cat == "rocky":
        commit_action, cross_tile, cross_act = A_COMMIT_MINE, ROCK, A_MINE
    elif cat == "lakes":
        commit_action, cross_tile, cross_act = A_COMMIT_BUILD, WATER, A_BUILD
    elif build_ok:
        commit_action, cross_tile, cross_act = A_COMMIT_BUILD, WATER, A_BUILD
    else:
        commit_action, cross_tile, cross_act = A_COMMIT_MINE, ROCK, A_MINE
    env.step(commit_action)

    while env._step_count < env.max_steps:
        if env._terrain[env._pos] == TARGET:
            return env._step_count, True
        path = _bfs_path(env._terrain, env._pos, cross_tile)
        if not path:
            return env._step_count, False
        nxt = path[0]
        dr = nxt[0] - env._pos[0]
        dc = nxt[1] - env._pos[1]
        face = _delta_to_facing(dr, dc)
        tile = int(env._terrain[nxt])
        if tile == cross_tile:                          # face it, then build/mine
            if env._facing != face:
                env.step(_FACE_TO_MOVE[face])
            else:
                env.step(cross_act)
        else:                                           # walkable → step
            env.step(_FACE_TO_MOVE[face])
    return env._step_count, (env._terrain[env._pos] == TARGET)


__all__ = ["scripted_solve"]
