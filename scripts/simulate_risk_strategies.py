"""Simulate conservative vs risky pathfinding strategies.

Game: spawn → target on island maps with terrain barriers.
Forest tiles have a probability of spawning an apple (+heal HP).

Conservative strategy: avoid dangerous terrain, detour around barriers
Risky strategy: take shortest path, cut through barriers

Two reward functions:
  r_safe  = 1.0 if reached, else 0                    → conservative play
  r_fast  = (1 + speed_bonus) if reached, else 0       → risky speed play

Usage:
    python scripts/simulate_risk_strategies.py
    python scripts/simulate_risk_strategies.py --sweep
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cogniland.env.types import EnvConfig, MapGenConfig, _DEFAULT_TERRAINS

# ── Terrain setup ────────────────────────────────────────────────────────

TERRAIN_NAMES = [t.name for t in _DEFAULT_TERRAINS]
DEFAULT_THRESHOLDS = np.array([t.threshold for t in _DEFAULT_TERRAINS])

HP_DRAIN = {
    "ocean":      -4.0,
    "deep_water": -3.25,
    "water":      -2.5,
    "beach":      -0.40,
    "sandy":      -0.40,
    "grassland":  -0.40,
    "forest":     -0.60,
    "rocky":      -1.0,
    "mountains":  -2.0,
}

INIT_HP = 100.0
MAX_HP = 100.0
APPLE_HEAL = 20.0
APPLE_PROB = 0.15
MAX_STEPS = 500
SIZE = 150

# ── Map generation: island with barrier ──────────────────────────────────

def _make_island_with_ridge(seed: int) -> tuple[np.ndarray, tuple, tuple]:
    """Generate a 150x150 island with a mountain ridge across the middle.
    Spawn on left, target on right — forces detour vs cut-through decision."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.RandomState(seed)
    S = SIZE

    # Base: gentle heightmap for the island
    noise = rng.uniform(0, 1, (S, S)).astype(np.float32)
    hm = gaussian_filter(noise, sigma=12).astype(np.float32)
    hm = (hm - hm.min()) / (hm.max() - hm.min())

    # Island mask: circular with noisy border
    Y, X = np.mgrid[:S, :S]
    cx, cy = S/2, S/2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    border_noise = gaussian_filter(rng.uniform(0, 1, (S, S)), sigma=8)
    radius = S * 0.42 + border_noise * S * 0.08
    island = dist < radius
    hm[~island] = 0.0

    # Scale land heights to grassland/forest range (0.06-0.60)
    land = hm > 0
    if land.any():
        hm[land] = 0.06 + hm[land] * 0.50  # mostly grassland + forest

    # Mountain ridge across the middle (vertical, slightly wavy)
    ridge_col = S // 2
    warp = gaussian_filter(rng.uniform(-1, 1, S), sigma=10) * 15
    ridge_width = rng.randint(18, 30)  # wide ridge = big detour

    for r in range(S):
        col_center = int(ridge_col + warp[r])
        for dc in range(-ridge_width, ridge_width + 1):
            c = col_center + dc
            if 0 <= c < S and island[r, c]:
                # Mountain core, fading to rocky at edges
                dist_from_center = abs(dc) / ridge_width
                if dist_from_center < 0.5:
                    hm[r, c] = 0.80 + rng.uniform(0, 0.15)  # mountains
                elif dist_from_center < 0.75:
                    hm[r, c] = 0.65 + rng.uniform(0, 0.10)  # rocky
                else:
                    hm[r, c] = 0.55 + rng.uniform(0, 0.10)  # forest/rocky transition

    # Optionally add a gap in the ridge (30% chance) — makes some maps have a pass
    if rng.random() < 0.15:  # rare gap in ridge
        gap_row = rng.randint(S // 4, 3 * S // 4)
        gap_h = rng.randint(6, 15)
        for r in range(max(0, gap_row - gap_h), min(S, gap_row + gap_h)):
            col_center = int(ridge_col + warp[r])
            for dc in range(-ridge_width - 2, ridge_width + 3):
                c = col_center + dc
                if 0 <= c < S and island[r, c]:
                    hm[r, c] = 0.15 + rng.uniform(0, 0.10)  # grassland pass

    np.clip(hm, 0, 0.99, out=hm)

    # Spawn left side, target right side (both on land, away from ridge)
    left_land = np.where(island & (X < S * 0.3) & (hm > 0.05) & (hm < 0.60))
    right_land = np.where(island & (X > S * 0.7) & (hm > 0.05) & (hm < 0.60))

    if len(left_land[0]) == 0 or len(right_land[0]) == 0:
        # Fallback
        return hm, (S//2, S//4), (S//2, 3*S//4)

    li = rng.randint(len(left_land[0]))
    ri = rng.randint(len(right_land[0]))
    spawn = (int(left_land[0][li]), int(left_land[1][li]))
    target = (int(right_land[0][ri]), int(right_land[1][ri]))

    return hm, spawn, target


def _make_island_with_lake(seed: int) -> tuple[np.ndarray, tuple, tuple]:
    """Generate a 150x150 island with a central lake.
    Must go around or swim through."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.RandomState(seed + 5000)
    S = SIZE

    noise = rng.uniform(0, 1, (S, S)).astype(np.float32)
    hm = gaussian_filter(noise, sigma=12).astype(np.float32)
    hm = (hm - hm.min()) / (hm.max() - hm.min())

    Y, X = np.mgrid[:S, :S]
    cx, cy = S/2, S/2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    border_noise = gaussian_filter(rng.uniform(0, 1, (S, S)), sigma=8)
    radius = S * 0.42 + border_noise * S * 0.08
    island = dist < radius
    hm[~island] = 0.0

    land = hm > 0
    if land.any():
        hm[land] = 0.06 + hm[land] * 0.50

    # Central lake: elliptical, wavy border
    lake_rx = rng.randint(25, 40)   # wider lake
    lake_ry = rng.randint(35, 55)  # elongated N-S to block E-W travel
    lake_noise = gaussian_filter(rng.uniform(-1, 1, (S, S)), sigma=6) * 10
    lake_dist = ((X - cx + lake_noise)**2 / lake_rx**2 +
                 (Y - cy)**2 / lake_ry**2)

    for r in range(S):
        for c in range(S):
            if island[r, c] and lake_dist[r, c] < 1.0:
                depth = 1.0 - lake_dist[r, c]
                if depth > 0.6:
                    hm[r, c] = 0.005  # ocean
                elif depth > 0.3:
                    hm[r, c] = 0.03   # deep water
                else:
                    hm[r, c] = 0.04   # water

    np.clip(hm, 0, 0.99, out=hm)

    left_land = np.where(island & (X < S * 0.25) & (hm > 0.05) & (hm < 0.60))
    right_land = np.where(island & (X > S * 0.75) & (hm > 0.05) & (hm < 0.60))

    if len(left_land[0]) == 0 or len(right_land[0]) == 0:
        return hm, (S//2, S//4), (S//2, 3*S//4)

    li = rng.randint(len(left_land[0]))
    ri = rng.randint(len(right_land[0]))
    spawn = (int(left_land[0][li]), int(left_land[1][li]))
    target = (int(right_land[0][ri]), int(right_land[1][ri]))

    return hm, spawn, target


def generate_map(seed: int) -> tuple[np.ndarray, tuple, tuple, str]:
    """Generate a map with a barrier. Alternates ridge and lake."""
    if seed % 2 == 0:
        hm, spawn, target = _make_island_with_ridge(seed)
        return hm, spawn, target, "ridge"
    else:
        hm, spawn, target = _make_island_with_lake(seed)
        return hm, spawn, target, "lake"

# ── Terrain helpers ──────────────────────────────────────────────────────

def _terrain_idx(hm: np.ndarray) -> np.ndarray:
    return np.searchsorted(DEFAULT_THRESHOLDS, hm).clip(0, len(TERRAIN_NAMES) - 1)


def _generate_apple_map(idx_map: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed + 77777)
    forest_idx = TERRAIN_NAMES.index("forest")
    return (idx_map == forest_idx) & (rng.random(idx_map.shape) < APPLE_PROB)

# ── A* pathfinding ───────────────────────────────────────────────────────

def _astar(hm, start, goal, idx_map, strategy: str) -> list | None:
    H, W = hm.shape
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    if strategy == "risky":
        def edge_cost(r, c):
            return 1.0
    elif strategy == "conservative":
        def edge_cost(r, c):
            name = TERRAIN_NAMES[int(idx_map[r, c])]
            drain = abs(HP_DRAIN[name])
            if drain >= 1.0:
                return 5.0 + drain * 10   # strongly avoid dangerous terrain
            return 1.0 + drain * 2
    else:
        raise ValueError(strategy)

    counter = 0
    h0 = abs(start[0]-goal[0]) + abs(start[1]-goal[1])
    open_set = [(h0, 0.0, counter, start)]
    best_g = {start: 0.0}
    came_from = {}

    while open_set:
        _, g, _, pos = heapq.heappop(open_set)
        if pos == goal:
            path = [goal]
            p = goal
            while p != start:
                p = came_from[p]
                path.append(p)
            path.reverse()
            return path
        if g > best_g.get(pos, float("inf")):
            continue
        for dr, dc in dirs:
            nr, nc = pos[0]+dr, pos[1]+dc
            if 0 <= nr < H and 0 <= nc < W:
                ng = g + edge_cost(nr, nc)
                if ng < best_g.get((nr, nc), float("inf")):
                    best_g[(nr, nc)] = ng
                    came_from[(nr, nc)] = pos
                    h = abs(nr-goal[0]) + abs(nc-goal[1])
                    counter += 1
                    heapq.heappush(open_set, (ng+h, ng, counter, (nr, nc)))
    return None


def _simulate_walk(path, idx_map, apple_map, hp_drain) -> dict:
    hp = INIT_HP
    hp_trace = [hp]
    apples = 0
    terrain_steps = {n: 0 for n in TERRAIN_NAMES}

    for p in path[1:]:
        name = TERRAIN_NAMES[int(idx_map[p[0], p[1]])]
        terrain_steps[name] += 1
        hp += hp_drain[name]
        if apple_map[p[0], p[1]] and hp < MAX_HP:
            hp = min(MAX_HP, hp + APPLE_HEAL)
            apples += 1
        hp_trace.append(hp)

    alive = all(h > 0 for h in hp_trace)
    return {
        "length": len(path) - 1,
        "final_hp": hp_trace[-1],
        "min_hp": min(hp_trace),
        "alive": alive,
        "outcome": "reached" if alive else "died",
        "apples_eaten": apples,
        "terrain_steps": terrain_steps,
    }

# ── Reward functions ─────────────────────────────────────────────────────

def reward_safe(outcome: str, length: int) -> float:
    """Pure survival: +1 if reached. Maximizing E[r_safe] → conservative policy."""
    return 1.0 if outcome == "reached" else 0.0


def reward_fast(outcome: str, length: int) -> float:
    """Linear speed penalty with tight budget.
    r = max(0, 1 - λ·steps).  λ=0.006 → budget≈167 steps.
    Conservative paths (~150 steps) get small reward (~0.10).
    Risky paths (~125 steps) get large reward (~0.25).
    The 2.2× per-episode reward ratio overcomes the ~2× survival ratio."""
    if outcome != "reached":
        return 0.0
    lam = 0.007
    return max(0.0, 1.0 - lam * length)

# ── Simulation ───────────────────────────────────────────────────────────

def run_simulation(n_maps: int = 20, base_seed: int = 42,
                   hp_drain: dict | None = None, verbose: bool = True) -> dict:
    if hp_drain is None:
        hp_drain = HP_DRAIN

    results = {"conservative": [], "risky": []}

    for i in range(n_maps):
        seed = base_seed + i
        hm, spawn, target, barrier = generate_map(seed)
        idx_map = _terrain_idx(hm)
        apple_map = _generate_apple_map(idx_map, seed)

        for strategy in ["conservative", "risky"]:
            path = _astar(hm, spawn, target, idx_map, strategy)
            if path is None:
                results[strategy].append({
                    "seed": seed, "barrier": barrier, "outcome": "no_path",
                    "length": 0, "final_hp": 0, "min_hp": 0, "alive": False,
                    "apples_eaten": 0, "terrain_steps": {},
                })
                continue
            res = _simulate_walk(path, idx_map, apple_map, hp_drain)
            res["seed"] = seed
            res["barrier"] = barrier
            res["r_safe"] = reward_safe(res["outcome"], res["length"])
            res["r_fast"] = reward_fast(res["outcome"], res["length"])
            results[strategy].append(res)

    if verbose:
        _print_summary(results, n_maps, hp_drain)

    return results


def _print_summary(results, n_maps, hp_drain):
    print(f"\n{'='*72}")
    print(f"INIT_HP={INIT_HP}, apple_prob={APPLE_PROB:.0%}, apple_heal={APPLE_HEAL}")
    print(f"Drains: grass={hp_drain['grassland']}, forest={hp_drain['forest']}, "
          f"water={hp_drain['water']}, rocky={hp_drain['rocky']}, mtn={hp_drain['mountains']}")
    print(f"{'='*72}")

    for strategy in ["conservative", "risky"]:
        eps = results[strategy]
        alive = [e for e in eps if e["alive"]]
        n = len(eps)
        avg_len = np.mean([e["length"] for e in eps if e["length"] > 0])
        avg_len_alive = np.mean([e["length"] for e in alive]) if alive else 0
        avg_hp = np.mean([e["final_hp"] for e in alive]) if alive else 0
        avg_min = np.mean([e["min_hp"] for e in alive]) if alive else 0
        avg_apples = np.mean([e["apples_eaten"] for e in eps if e["length"] > 0])
        avg_r_safe = np.mean([e.get("r_safe", 0) for e in eps])
        avg_r_fast = np.mean([e.get("r_fast", 0) for e in eps])
        survival = len(alive) / n

        print(f"\n  {strategy.upper()}")
        print(f"  survival: {len(alive)}/{n} ({survival:.0%})")
        print(f"  avg_steps (all): {avg_len:.0f}   avg_steps (survivors): {avg_len_alive:.0f}")
        print(f"  survivors: avg_hp={avg_hp:.1f}  min_hp={avg_min:.1f}  apples={avg_apples:.1f}")
        print(f"  E[r_safe]={avg_r_safe:.3f}   E[r_fast]={avg_r_fast:.3f}")

        if alive:
            all_t = {}
            for e in alive:
                for t, cnt in e.get("terrain_steps", {}).items():
                    all_t[t] = all_t.get(t, 0) + cnt
            total = sum(all_t.values()) or 1
            fracs = sorted(((t, all_t[t]/total) for t in TERRAIN_NAMES if all_t.get(t, 0) > 0),
                           key=lambda x: -x[1])
            print(f"  terrain: {'  '.join(f'{t}={f:.0%}' for t,f in fracs[:6])}")

    cons = results["conservative"]
    risk = results["risky"]
    c_alive = sum(1 for e in cons if e["alive"])
    r_alive = sum(1 for e in risk if e["alive"])
    c_steps = np.mean([e["length"] for e in cons if e["alive"]]) if c_alive else 999
    r_steps = np.mean([e["length"] for e in risk if e["alive"]]) if r_alive else 999
    c_r_fast = np.mean([e.get("r_fast", 0) for e in cons])
    r_r_fast = np.mean([e.get("r_fast", 0) for e in risk])

    print(f"\n  {'─'*50}")
    print(f"  CONSERVATIVE: {c_alive}/{n_maps} survive, avg {c_steps:.0f} steps")
    print(f"  RISKY:        {r_alive}/{n_maps} survive, avg {r_steps:.0f} steps")
    if c_alive > 0 and r_alive > 0:
        speedup = (c_steps - r_steps) / c_steps * 100
        print(f"  Speed diff: risky is {speedup:.0f}% faster")
    print(f"  E[r_safe]: cons={np.mean([e.get('r_safe',0) for e in cons]):.3f}  risky={np.mean([e.get('r_safe',0) for e in risk]):.3f}")
    print(f"  E[r_fast]: cons={c_r_fast:.3f}  risky={r_r_fast:.3f}")

    # Verdict
    surv_gap = c_alive/n_maps - r_alive/n_maps
    speed_pct = (c_steps - r_steps) / max(c_steps, 1) * 100
    fast_gap = r_r_fast - c_r_fast

    print(f"\n  VERDICT:")
    if surv_gap > 0.15 and speed_pct > 15 and fast_gap > 0:
        print(f"  ✓ GOOD TRADEOFF: cons +{surv_gap:.0%} survival, risky +{speed_pct:.0f}% speed")
        print(f"  ✓ r_safe favors conservative, r_fast favors risky")
    else:
        issues = []
        if surv_gap < 0.15:
            issues.append(f"survival gap too small ({surv_gap:.0%})")
        if speed_pct < 15:
            issues.append(f"speed diff too small ({speed_pct:.0f}%)")
        if fast_gap <= 0:
            issues.append("r_fast doesn't favor risky")
        print(f"  ✗ Issues: {', '.join(issues)}")

# ── Parameter sweep ──────────────────────────────────────────────────────

def sweep():
    print("PARAMETER SWEEP")
    print("="*72)

    configs = []
    for grass in [-0.15, -0.20, -0.25, -0.30, -0.40]:
        for forest_delta in [0.0, -0.10, -0.20]:
            for mtn in [-2.0, -3.0, -5.0, -8.0]:
                for water in [-1.0, -1.5, -2.5]:
                    configs.append({
                        "ocean": water * 2,
                        "deep_water": water * 1.3,
                        "water": water,
                        "beach": grass,
                        "sandy": grass,
                        "grassland": grass,
                        "forest": grass + forest_delta,
                        "rocky": mtn * 0.5,
                        "mountains": mtn,
                    })

    print(f"Testing {len(configs)} configs × 10 maps...")
    best_score, best_cfg = -1, None

    for cfg in configs:
        results = run_simulation(n_maps=10, hp_drain=cfg, verbose=False)
        cons, risk = results["conservative"], results["risky"]
        n = len(cons)
        c_alive = sum(1 for e in cons if e["alive"])
        r_alive = sum(1 for e in risk if e["alive"])
        c_surv, r_surv = c_alive/n, r_alive/n
        c_steps = np.mean([e["length"] for e in cons if e["alive"]]) if c_alive else 999
        r_steps = np.mean([e["length"] for e in risk if e["alive"]]) if r_alive else 999
        c_rf = np.mean([e.get("r_fast", 0) for e in cons])
        r_rf = np.mean([e.get("r_fast", 0) for e in risk])

        surv_gap = c_surv - r_surv
        speed_gap = (c_steps - r_steps) / max(c_steps, 1)

        score = 0
        if 0.80 <= c_surv: score += 2
        if 0.30 <= r_surv <= 0.70: score += 3
        elif 0.20 <= r_surv <= 0.80: score += 1
        if 0.15 <= surv_gap <= 0.55: score += 2
        if speed_gap >= 0.15: score += 2
        if r_rf > c_rf: score += 2  # key: r_fast must favor risky

        tag = f"g={cfg['grassland']:.2f} f={cfg['forest']:.2f} w={cfg['water']:.1f} m={cfg['mountains']:.1f}"
        line = (f"{tag} | surv: c={c_surv:.0%} r={r_surv:.0%} gap={surv_gap:.0%} | "
                f"steps: c={c_steps:.0f} r={r_steps:.0f} Δ={speed_gap:.0%} | "
                f"rf: c={c_rf:.3f} r={r_rf:.3f} | score={score}")

        if score >= 9:
            print(f"  ★★ {line}")
        elif score >= 7:
            print(f"  ★  {line}")

        if score > best_score:
            best_score, best_cfg = score, cfg

    print(f"\n{'='*72}")
    print(f"BEST (score={best_score}):")
    print(f"  grass={best_cfg['grassland']}, forest={best_cfg['forest']}, "
          f"water={best_cfg['water']}, rocky={best_cfg['rocky']}, mtn={best_cfg['mountains']}")
    print(f"\nDetailed run with best config:")
    run_simulation(n_maps=20, hp_drain=best_cfg, verbose=True)
    return best_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--maps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sweep:
        sweep()
    else:
        run_simulation(n_maps=args.maps, base_seed=args.seed)


if __name__ == "__main__":
    main()
