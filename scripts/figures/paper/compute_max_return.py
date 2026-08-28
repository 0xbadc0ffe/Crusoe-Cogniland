#!/usr/bin/env python3
"""The two anchors that normalise fork_wall return to a 0-100 per cent scale.

``r_optimal`` -- the maximum achievable episode return, and ``r_random`` -- the
mean return of a uniform-random policy. Figure 7.2 plots every agent as

    return_pct = 100 * (r - r_random) / (r_optimal - r_random)

so 0 per cent is a random policy and 100 per cent is optimal play. Raw returns
start near -8, so a plain r/r_optimal scale would push early training far below
-100 per cent and off the panel.

The fork_wall reward is
    -0.01                       per step                  (slack_penalty)
    +0.015 * (ctg_prev - ctg_curr)                        (PBRS, shaping_gamma=1)
    +3.0                        on the category-matching door (reach_bonus)
With ``shaping_gamma == 1`` the shaping term telescopes over an episode, so an
episode that ends on the correct door earns exactly

    R = 3.0 + 0.015 * ctg(spawn) - 0.01 * steps

and the best achievable return per map uses the shortest spawn -> correct-door
path. This script measures that per map over ``forkwall6k/train.pkl``, and
measures ``r_random`` by rolling a uniform-random policy on the same pool.

Why not ``_solver.scripted_solve``: its BFS stops at ANY ``TARGET`` tile, and on
fork_wall maps both doors are TARGET tiles, so it happily walks into the decoy
door. Here the decoy door cells are masked out of the BFS graph, so the path is
the shortest route to the REWARDED door. The scripted run is executed against
the real env, so the reported return is a genuinely achieved return, not an
arithmetic guess -- the closed-form value above is recomputed alongside it and
the two are cross-checked.

  python scripts/figures/paper/compute_max_return.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from cogniland.bridge_tunnel.env import (  # noqa: E402
    A_BUILD, A_DOWN, A_LEFT, A_MINE, A_RIGHT, A_UP, F_DOWN, F_LEFT, F_RIGHT,
    F_UP, BridgeTunnelEnv,
)
from cogniland.bridge_tunnel.map_pool import MapPool  # noqa: E402
from cogniland.bridge_tunnel.tiles import (  # noqa: E402
    DIRT, GRASS, ROCK, SAND, TARGET, TREE, WATER, WOOD,
)

sys.path.insert(0, str(REPO / "STORM_model"))
from cl.environments.bridge_tunnel import FORKWALL_KWARGS  # noqa: E402

FACE_TO_MOVE = {F_UP: A_UP, F_DOWN: A_DOWN, F_LEFT: A_LEFT, F_RIGHT: A_RIGHT}
WALK = (GRASS, WOOD, TARGET, SAND, DIRT)
OUT = REPO / "outputs/belief_report/max_return.json"


def facing(dr: int, dc: int) -> int:
    if dr < 0:
        return F_UP
    if dr > 0:
        return F_DOWN
    if dc < 0:
        return F_LEFT
    return F_RIGHT


def bfs_to(terrain: np.ndarray, start, goals: set) -> list | None:
    """Shortest cell path start -> any cell in ``goals``; TREE blocks."""
    H, W = terrain.shape
    seen = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur in goals:
            path = []
            while cur != start:
                path.append(cur)
                cur = seen[cur]
            return list(reversed(path))
        r, c = cur
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in seen \
                    and int(terrain[nxt]) != TREE:
                seen[nxt] = cur
                q.append(nxt)
    return None


def solve_one(env: BridgeTunnelEnv, rec) -> dict:
    """Drive ``env`` down the shortest correct-door path; report the return."""
    env._fixed_record = rec
    env.reset()
    correct = set(env._correct_cells or ())
    if not correct:
        return {"ok": False, "why": "no correct door"}
    wrong = (set(rec.top_goal_cells) | set(rec.bottom_goal_cells)) - correct

    # mask the decoy door out of the search graph so the path cannot cross it
    graph = env._terrain.copy()
    for cell in wrong:
        graph[cell] = TREE
    ctg_spawn = float(env._ctg[env._pos])
    path = bfs_to(graph, tuple(env._pos), correct)
    if path is None:
        return {"ok": False, "why": "unreachable correct door"}

    total = 0.0
    for cell in path:
        while tuple(env._pos) != cell:
            if env._step_count >= env.max_steps:
                return {"ok": False, "why": "step budget"}
            face = facing(cell[0] - env._pos[0], cell[1] - env._pos[1])
            tile = int(env._terrain[cell])
            if tile in (WATER, ROCK):
                # turn to face the obstacle, clear it, then walk in
                act = (FACE_TO_MOVE[face] if env._facing != face
                       else (A_BUILD if tile == WATER else A_MINE))
            else:
                act = FACE_TO_MOVE[face]
            _, rew, term, trunc, info = env.step(act)
            total += float(rew)
            if term or trunc:
                # TRUE door metric: the final cell must be a rewarded door cell
                # (repo convention -- never "return > 0", see CLAUDE.md)
                landed = tuple(env._pos) in correct
                return {"ok": bool(info["reached_target"]) and landed,
                        "why": "ended",
                        "ret": total, "steps": int(env._step_count),
                        "ctg_spawn": ctg_spawn, "path_len": len(path),
                        "category": rec.category}
    return {"ok": False, "why": "path exhausted without terminating"}


def random_return(env: BridgeTunnelEnv, pool, n_maps: int, seed: int = 0) -> dict:
    """Mean undiscounted return of a uniform-random policy on the same pool."""
    rng = np.random.default_rng(seed)
    n_actions = int(env.action_space.n)
    rets, steps, wins = [], [], 0
    stride = max(1, len(pool) // n_maps)
    for k in range(n_maps):
        env._fixed_record = pool.get(k * stride)
        env.reset()
        total = 0.0
        while True:
            _, rew, term, trunc, info = env.step(int(rng.integers(n_actions)))
            total += float(rew)
            if term or trunc:
                wins += int(bool(info["reached_target"]))
                break
        rets.append(total)
        steps.append(int(env._step_count))
    a = np.array(rets)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), "episodes": int(a.size),
            "mean_steps": float(np.mean(steps)),
            "correct_door_rate": wins / len(rets), "seed": seed,
            "n_actions": n_actions}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="maps to use (0 = all)")
    ap.add_argument("--random-maps", type=int, default=200,
                    help="maps to roll the uniform-random policy on")
    ap.add_argument("--pool", default="data/bridge_tunnel/forkwall6k/train.pkl")
    args = ap.parse_args()

    pool = MapPool(REPO / args.pool)
    env = BridgeTunnelEnv(**FORKWALL_KWARGS)
    n = len(pool) if args.limit <= 0 else min(args.limit, len(pool))

    rets, steps, ctgs, closed, fails = [], [], [], [], []
    per_cat: dict[str, list] = {}
    for i in range(n):
        rec = pool.get(i)
        res = solve_one(env, rec)
        if not res.get("ok"):
            fails.append({"idx": i, "why": res.get("why")})
            continue
        rets.append(res["ret"])
        steps.append(res["steps"])
        ctgs.append(res["ctg_spawn"])
        # closed form: telescoped PBRS + bonus - slack
        closed.append(3.0 + 0.015 * res["ctg_spawn"] - 0.01 * res["steps"])
        per_cat.setdefault(res["category"] or "none", []).append(res["ret"])

    rets_a = np.array(rets)
    closed_a = np.array(closed)
    resid = float(np.max(np.abs(rets_a - closed_a))) if rets else float("nan")

    rnd = random_return(env, pool, args.random_maps)
    r_opt, r_rnd = float(rets_a.mean()), rnd["mean"]

    out = {
        "metric": "maximum achievable undiscounted episode return, fork_wall",
        "r_optimal": r_opt,
        "r_random": r_rnd,
        "normalisation": (
            "return_pct = 100 * (r - r_random) / (r_optimal - r_random); "
            "0 per cent is a uniform-random policy, 100 per cent is optimal play"
        ),
        "random_policy": {
            **rnd,
            "method": (
                f"uniform-random actions over the {rnd['n_actions']} discrete "
                f"actions, one episode on each of {rnd['episodes']} maps taken "
                "at even stride through the pool, numpy default_rng(0)"
            ),
        },
        "method": (
            "For every map in the pool, a scripted optimal solver walks the "
            "shortest spawn->correct-door path (BFS with TREE blocking and the "
            "decoy door masked out) inside the real BridgeTunnelEnv, and the "
            "returns actually emitted by the env are summed. Cross-checked "
            "against the closed form 3.0 + 0.015*ctg(spawn) - 0.01*steps, "
            "valid because shaping_gamma=1 telescopes the PBRS term."
        ),
        "pool": args.pool,
        "maps_used": int(rets_a.size),
        "maps_failed": len(fails),
        "mean": float(rets_a.mean()),
        "std": float(rets_a.std()),
        "median": float(np.median(rets_a)),
        "min": float(rets_a.min()),
        "max": float(rets_a.max()),
        "p5": float(np.percentile(rets_a, 5)),
        "p95": float(np.percentile(rets_a, 95)),
        "mean_steps": float(np.mean(steps)),
        "mean_ctg_spawn": float(np.mean(ctgs)),
        "closed_form_mean": float(closed_a.mean()),
        "max_abs_closed_form_residual": resid,
        "per_category_mean": {k: float(np.mean(v)) for k, v in sorted(per_cat.items())},
        "per_category_n": {k: len(v) for k, v in sorted(per_cat.items())},
        "env_kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in FORKWALL_KWARGS.items()},
        "failures_sample": fails[:10],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("method", "env_kwargs", "failures_sample")}, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
