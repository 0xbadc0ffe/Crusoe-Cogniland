"""Tune tile-effect parameters for the Survival Kit game.

Runs a small simulation over generated maps to verify whether a candidate
parameter table satisfies the design goals:
  - Naive greedy-compass paths (no tools) usually fail.
  - Grassland-only paths are infeasible without healing.
  - Rough-terrain HP drain halves HP in 6-7 steps.
  - Each tool unlocks one terrain family.

Usage:
    python scripts/tune_tile_effects.py --seeds 20
"""

from __future__ import annotations

import argparse
import heapq
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import generate_maps as gt


# ── Parameter table ────────────────────────────────────────────────────────

@dataclass
class TileEffects:
    hp_drain: dict[str, int] = field(default_factory=lambda: {
        "ocean":       16,
        "deep_water":  10,
        "water":       6,
        "beach":       1,
        "sandy":       1,
        "grassland":   1,
        "forest":      3,
        "rocky":       6,
        "mountains":   12,
    })
    raft_drain: dict[str, int] = field(default_factory=lambda: {
        "water": 1, "deep_water": 3, "ocean": 8,
    })
    rope_drain: dict[str, int] = field(default_factory=lambda: {
        "rocky": 1, "mountains": 3,
    })
    shoes_drain_grassland: int = 0.5
    shoes_k: int = 10

    berry_heal: int = 10
    forest_wood: int = 10
    wood_max: int = 100
    craft_cost: int = 100
    hp_max: int = 100
    init_hp: int = 100


def drain_for(terrain: str, tools: frozenset[str], consec_grass: int,
              fx: TileEffects) -> int:
    if "raft" in tools and terrain in fx.raft_drain:
        return fx.raft_drain[terrain]
    if "rope" in tools and terrain in fx.rope_drain:
        return fx.rope_drain[terrain]
    if "shoes" in tools and terrain == "grassland" and consec_grass >= fx.shoes_k:
        return fx.shoes_drain_grassland
    return fx.hp_drain.get(terrain, 1)


# ── Map prep ───────────────────────────────────────────────────────────────

def _terrain_idx(hm: np.ndarray, biome: str) -> np.ndarray:
    return gt._terrain_idx(hm, biome).astype(np.int32)


def _is_deadly(hm_val: float) -> bool:
    return hm_val <= gt.DEADLY_VALUE / 2


def sample_spawn_target(
    hm: np.ndarray, biome: str, seed: int, min_manhattan: int = 70,
) -> tuple[tuple[int, int], tuple[int, int]]:
    rng = np.random.RandomState(seed + 77777)
    thresholds = gt.BIOME_THRESHOLDS[biome]
    water_upper = thresholds[gt.TERRAIN_NAMES.index("water")]
    land_r, land_c = np.where(hm > water_upper)
    if len(land_r) < 2:
        mid = gt.CROP_SIZE // 2
        return (mid, mid), (mid, mid)

    for _ in range(500):
        i = rng.randint(len(land_r))
        j = rng.randint(len(land_r))
        s = (int(land_r[i]), int(land_c[i]))
        t = (int(land_r[j]), int(land_c[j]))
        if abs(s[0] - t[0]) + abs(s[1] - t[1]) >= min_manhattan:
            return s, t

    i0 = 0
    idx_far = int(np.argmax((land_r - land_r[i0]) ** 2 + (land_c - land_c[i0]) ** 2))
    return (int(land_r[i0]), int(land_c[i0])), (int(land_r[idx_far]), int(land_c[idx_far]))


# ── Simulator core ─────────────────────────────────────────────────────────

@dataclass
class SimResult:
    alive: bool
    reached: bool
    steps: int
    final_hp: int
    min_hp: int
    wood: int
    path: list[tuple[int, int]] = field(default_factory=list)


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def simulate_greedy_compass(
    hm: np.ndarray, berry_mask: np.ndarray, biome: str,
    spawn: tuple[int, int], target: tuple[int, int],
    fx: TileEffects, tools: frozenset[str] = frozenset(),
    max_steps: int = 4000,
) -> SimResult:
    """Always step toward target minimizing Euclidean distance."""
    H, W = hm.shape
    t_idx = _terrain_idx(hm, biome)
    berries = berry_mask.copy()
    pos = spawn
    hp = fx.init_hp
    wood = 0
    consec = 0
    path = [pos]
    min_hp = hp
    visit_counts: dict[tuple[int, int], int] = {}

    while True:
        if pos == target:
            return SimResult(True, True, len(path) - 1, hp, min_hp, wood, path)
        if len(path) - 1 >= max_steps:
            return SimResult(hp > 0, False, len(path) - 1, hp, min_hp, wood, path)

        best = None
        best_key = None
        for dr, dc in DIRS:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if _is_deadly(hm[nr, nc]):
                continue
            tname = gt.TERRAIN_NAMES[int(t_idx[nr, nc])]
            new_consec = consec + 1 if tname == "grassland" else 0
            d = drain_for(tname, tools, new_consec, fx)
            euc = (nr - target[0]) ** 2 + (nc - target[1]) ** 2
            key = (euc, d, nr, nc)
            if best_key is None or key < best_key:
                best_key = key
                best = (nr, nc, tname, d)

        if best is None:
            return SimResult(False, False, len(path) - 1, hp, min_hp, wood, path)

        nr, nc, tname, drain = best
        visit_counts[(nr, nc)] = visit_counts.get((nr, nc), 0) + 1
        if visit_counts[(nr, nc)] > 6:
            return SimResult(hp > 0, False, len(path) - 1, hp, min_hp, wood, path)

        if berries[nr, nc]:
            hp = min(fx.hp_max, hp + fx.berry_heal)
            berries[nr, nc] = False
        if tname == "forest":
            wood += fx.forest_wood
        hp -= drain
        min_hp = min(min_hp, hp)
        consec = consec + 1 if tname == "grassland" else 0
        pos = (nr, nc)
        path.append(pos)
        if hp <= 0:
            return SimResult(False, False, len(path) - 1, hp, min_hp, wood, path)


def simulate_astar(
    hm: np.ndarray, berry_mask: np.ndarray, biome: str,
    spawn: tuple[int, int], target: tuple[int, int],
    fx: TileEffects, tools: frozenset[str] = frozenset(),
    grassland_only: bool = False,
    max_iters: int = 400_000,
) -> SimResult:
    """A* minimizing step count subject to HP > 0.

    State = (r, c, hp, consec_grass_capped). Berries are treated as
    repeatable heals (slightly optimistic, but fine for tuning bounds).
    """
    H, W = hm.shape
    t_idx = _terrain_idx(hm, biome)
    hp_max = fx.hp_max
    tr, tc = target

    def heuristic(r, c):
        return abs(r - tr) + abs(c - tc)

    start = (spawn[0], spawn[1], fx.init_hp, 0)
    visited_best: dict[tuple[int, int, int, int], int] = {start: 0}
    came_from: dict[tuple, tuple | None] = {start: None}
    counter = 0
    open_heap = [(heuristic(spawn[0], spawn[1]), 0, counter, start)]

    it = 0
    while open_heap and it < max_iters:
        it += 1
        f, g, _, state = heapq.heappop(open_heap)
        if visited_best.get(state, 1 << 30) < g:
            continue
        r, c, hp, consec = state
        if (r, c) == target:
            # reconstruct
            path = []
            s: tuple | None = state
            while s is not None:
                path.append((s[0], s[1]))
                s = came_from.get(s)
            path.reverse()
            return SimResult(True, True, len(path) - 1, hp, hp, 0, path)

        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if _is_deadly(hm[nr, nc]):
                continue
            tname = gt.TERRAIN_NAMES[int(t_idx[nr, nc])]
            if grassland_only and tname != "grassland":
                continue
            new_consec = (consec + 1) if tname == "grassland" else 0
            drain = drain_for(tname, tools, new_consec, fx)
            nhp = hp
            if berry_mask[nr, nc]:
                nhp = min(hp_max, nhp + fx.berry_heal)
            nhp -= drain
            if nhp <= 0:
                continue
            nstate = (nr, nc, int(nhp), min(new_consec, fx.shoes_k))
            ng = g + 1
            if ng < visited_best.get(nstate, 1 << 30):
                visited_best[nstate] = ng
                came_from[nstate] = state
                counter += 1
                heapq.heappush(open_heap, (ng + heuristic(nr, nc), ng, counter, nstate))

    return SimResult(False, False, 0, 0, 0, 0, [])


# ── Synthetic checks ───────────────────────────────────────────────────────

def check_rocky_7_steps(fx: TileEffects) -> bool:
    hp = fx.init_hp
    for _ in range(7):
        hp -= fx.hp_drain["rocky"]
    return hp <= fx.init_hp / 2


def check_grassland_only_infeasible(fx: TileEffects, length: int = 100) -> bool:
    hp = fx.init_hp
    for _ in range(length):
        hp -= fx.hp_drain["grassland"]
    return hp <= 0


def check_shoes_grassland_feasible(fx: TileEffects, length: int = 100) -> bool:
    hp = fx.init_hp
    consec = 0
    for _ in range(length):
        consec += 1
        d = fx.shoes_drain_grassland if consec >= fx.shoes_k else fx.hp_drain["grassland"]
        hp -= d
    return hp > 0


# ── Tuning runner ──────────────────────────────────────────────────────────

def run_tuning(args: argparse.Namespace) -> None:
    fx = TileEffects()
    biomes = list(gt.ALL_BIOMES)
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    tool_configs: list[tuple[str, frozenset[str]]] = [
        ("none",  frozenset()),
        ("raft",  frozenset({"raft"})),
        ("rope",  frozenset({"rope"})),
        ("shoes", frozenset({"shoes"})),
    ]

    results: dict[tuple[str, str, str], list[SimResult]] = {}

    print(f"Simulating {len(biomes)} biomes × {len(seeds)} seeds × {len(tool_configs) + 1} policies")
    for biome in biomes:
        for seed in seeds:
            smap = gt.generate_map(seed, biome)
            hm = smap.heightmap.numpy()
            bm = smap.berry_mask.numpy()
            spawn, target = sample_spawn_target(hm, biome, seed)

            r_g = simulate_greedy_compass(hm, bm, biome, spawn, target, fx)
            results.setdefault((biome, "greedy", "none"), []).append(r_g)

            for tname, toolset in tool_configs:
                r_a = simulate_astar(hm, bm, biome, spawn, target, fx, toolset)
                results.setdefault((biome, "astar", tname), []).append(r_a)

    _print_report(biomes, tool_configs, results)
    _check_acceptance(results, biomes, fx)
    _print_final_dict(fx)


def _print_report(
    biomes: list[str],
    tool_configs: list[tuple[str, frozenset[str]]],
    results: dict[tuple[str, str, str], list[SimResult]],
) -> None:
    print("\n" + "=" * 72)
    print("Per-biome results")
    print("=" * 72)
    for biome in biomes:
        print(f"\n{biome.upper()}")
        print(f"  {'policy/tool':<16}{'survival':>11}{'avg_steps':>12}{'avg_hp':>10}")
        print("  " + "-" * 50)
        rows: list[tuple[str, list[SimResult]]] = [
            ("greedy/none", results.get((biome, "greedy", "none"), [])),
        ]
        for tname, _ in tool_configs:
            rows.append((f"astar/{tname}", results.get((biome, "astar", tname), [])))
        for label, rs in rows:
            if not rs:
                continue
            alive = sum(1 for r in rs if r.reached)
            surv = alive / len(rs) * 100.0
            succ = [r for r in rs if r.reached]
            avg_steps = float(np.mean([r.steps for r in succ])) if succ else float("nan")
            avg_hp = float(np.mean([r.final_hp for r in succ])) if succ else float("nan")
            print(f"  {label:<16}{surv:>10.0f}%{avg_steps:>12.1f}{avg_hp:>10.1f}")


def _check_acceptance(
    results: dict[tuple[str, str, str], list[SimResult]],
    biomes: list[str],
    fx: TileEffects,
) -> None:
    print("\n" + "=" * 72)
    print("Acceptance checks")
    print("=" * 72)

    def surv(biome: str, policy: str, tool: str) -> float:
        rs = results.get((biome, policy, tool), [])
        if not rs:
            return 0.0
        return sum(1 for r in rs if r.reached) / len(rs)

    def avg_across(policy: str, tool: str) -> float:
        return float(np.mean([surv(b, policy, tool) for b in biomes]))

    checks: list[tuple[str, bool]] = [
        ("greedy no-tools avg survival < 20%",
         avg_across("greedy", "none") < 0.20),
        ("astar no-tools avg survival < 50%",
         avg_across("astar", "none") < 0.50),
        ("astar raft on archipelago >= 80%",
         surv("archipelago", "astar", "raft") >= 0.80),
        ("astar rope on highland >= 80%",
         surv("highland", "astar", "rope") >= 0.80),
        ("astar shoes on grassland >= 80%",
         surv("grassland", "astar", "shoes") >= 0.80),
        ("astar shoes doesn't dominate archipelago (±35pt)",
         abs(surv("archipelago", "astar", "shoes") - surv("archipelago", "astar", "none")) <= 0.35),
        ("astar shoes doesn't dominate highland (±35pt)",
         abs(surv("highland", "astar", "shoes") - surv("highland", "astar", "none")) <= 0.35),
        ("7 rocky steps halve HP",
         check_rocky_7_steps(fx)),
        ("100 grassland steps kill without shoes",
         check_grassland_only_infeasible(fx, 100)),
        ("100 grassland steps survive with shoes",
         check_shoes_grassland_feasible(fx, 100)),
    ]
    for label, ok in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}")


def _print_final_dict(fx: TileEffects) -> None:
    print("\n" + "=" * 72)
    print("Final TileEffects")
    print("=" * 72)
    print(f"  hp_drain              = {fx.hp_drain}")
    print(f"  raft_drain            = {fx.raft_drain}")
    print(f"  rope_drain            = {fx.rope_drain}")
    print(f"  shoes_k               = {fx.shoes_k}")
    print(f"  shoes_drain_grassland = {fx.shoes_drain_grassland}")
    print(f"  berry_heal            = {fx.berry_heal}")
    print(f"  forest_wood           = {fx.forest_wood}")
    print(f"  init_hp               = {fx.init_hp}")
    print(f"  hp_max                = {fx.hp_max}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()
    run_tuning(args)


if __name__ == "__main__":
    main()
