#!/usr/bin/env python3
"""Design preview: 3-biome / 3-skill environment with stochastic rollouts.

Each biome has TWO barrier features (placed on opposite diagonals) and a
unique scattered cue tile. With three skills (raft, harness, machete), each
biome ends up with two viable skills and one wrong skill — the skill whose
specialty tile is absent.

    biome    barriers (diagonals)  cue      viable             wrong
    aquatic  water + rock          sand     raft + harness     machete
    highland rock  + trees         dirt     harness + machete  raft
    wetland  water + trees         reeds    raft + machete     harness

Rollouts use a noisy-greedy policy on a slip-aware Dijkstra cost-to-go
surface (1/(1-p_slip) per walkable cell) under each skill, with stochastic
slip simulated at every step. Viable-skill rollouts thread the matching
barrier; wrong-skill rollouts pay the weight tax on every land cell while
gaining nothing, so they wander and often time out.

Out: outputs/previews/proposed_design.png
"""
from __future__ import annotations

import heapq
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import opensimplex
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]

# ── Tile IDs (preview-local; NOT the env's) ────────────────────────────────
GRASS, SAND, DIRT, REEDS, WATER, ROCK, TREE, TARGET = range(8)

TILE_COLORS = np.array([
    [108, 168, 100],   # grass
    [231, 195, 122],   # sand
    [134,  92,  60],   # dirt
    [ 82, 158, 152],   # reeds (teal)
    [ 55,  95, 155],   # water
    [110, 110, 120],   # rock
    [ 38,  80,  44],   # tree
    [255, 250, 215],   # target
], dtype=np.uint8)

# ── Skills ─────────────────────────────────────────────────────────────────
NONE, RAFT, HARNESS, MACHETE = range(4)
SKILL_RGB = {
    RAFT:    (0.18, 0.46, 0.85),   # blue
    HARNESS: (0.92, 0.42, 0.10),   # orange
    MACHETE: (0.22, 0.70, 0.30),   # green
}
SKILL_NAME = {RAFT: "raft", HARNESS: "harness", MACHETE: "machete"}

WRONG_SKILL = {"aquatic": MACHETE, "highland": RAFT, "wetland": HARNESS}
BIOMES = ["aquatic", "highland", "wetland"]
SEEDS = [7, 13, 21, 42, 77]
N_ROLLOUTS = 30
TRAJ_ALPHA = 0.12


# ── Slip table + Dijkstra ──────────────────────────────────────────────────

def slip(skill: int, tile: int, grass_noskill: float = 0.0) -> float:
    """Proposed slip table (mirrors current env semantics; trees specialised
    by machete; weight tax 0.50 on grass/sand/dirt/reeds when any skill held).
    """
    if tile == TARGET:
        return 0.0
    if tile == GRASS:
        return grass_noskill if skill == NONE else 0.50
    if tile in (SAND, DIRT, REEDS):
        return 0.30 if skill == NONE else 0.50
    if tile == WATER:
        return 0.0 if skill == RAFT else 0.75
    if tile == ROCK:
        return 0.0 if skill == HARNESS else 0.75
    if tile == TREE:
        return 0.0 if skill == MACHETE else 0.75
    return 0.75


def ctg_expected_attempts(terrain: np.ndarray, target, skill: int) -> np.ndarray:
    """Dijkstra on expected-attempts cost surface (1/(1-p_slip) per cell)."""
    H, W = terrain.shape
    cost = np.full((H, W), np.inf, dtype=np.float64)
    cost[target] = 0.0
    pq = [(0.0, int(target[0]), int(target[1]))]
    while pq:
        c, r, cc = heapq.heappop(pq)
        if c > cost[r, cc]:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, ncc = r + dr, cc + dc
            if not (0 <= nr < H and 0 <= ncc < W):
                continue
            p = slip(skill, int(terrain[nr, ncc]))
            if p >= 1.0:
                continue
            step = 1.0 / (1.0 - p)
            nc_v = c + step
            if nc_v < cost[nr, ncc]:
                cost[nr, ncc] = nc_v
                heapq.heappush(pq, (nc_v, nr, ncc))
    return cost


# ── Rollouts ───────────────────────────────────────────────────────────────

def rollout(terrain, spawn, target, skill, ctg, rng,
            max_steps: int = 300, eps: float = 0.08):
    pos = (int(spawn[0]), int(spawn[1]))
    target_t = (int(target[0]), int(target[1]))
    path = [pos]
    H, W = terrain.shape
    for _ in range(max_steps):
        if pos == target_t:
            break
        neighbors = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < H and 0 <= nc < W:
                v = ctg[nr, nc]
                if np.isfinite(v):
                    neighbors.append(((nr, nc), v))
        if not neighbors:
            break
        if rng.random() < eps:
            (nr, nc), _ = neighbors[int(rng.integers(len(neighbors)))]
        else:
            min_v = min(v for _, v in neighbors)
            tied = [pn for pn, v in neighbors if v == min_v]
            nr, nc = tied[int(rng.integers(len(tied)))]
        tile = int(terrain[nr, nc])
        if rng.random() < slip(skill, tile):
            path.append(pos)            # slipped — stay
        else:
            pos = (nr, nc)
            path.append(pos)
    reached = pos == target_t
    return np.asarray(path, dtype=np.int32), reached


# ── Map generation (procedural, deterministic by seed) ─────────────────────

def _warp_fields(size: int, seed: int, scale: float = 12.0, amp: float = 4.0):
    """Two independent low-freq simplex warp fields (Y and X displacement)."""
    sim_y = opensimplex.OpenSimplex(seed=seed * 31 + 1)
    sim_x = opensimplex.OpenSimplex(seed=seed * 31 + 2)
    xs = np.arange(size, dtype=np.float64) / scale
    ys = np.arange(size, dtype=np.float64) / scale
    wy = sim_y.noise2array(xs, ys) * amp
    wx = sim_x.noise2array(xs, ys) * amp
    return wy, wx


def _carve_capsule(terrain, p0, p1, width, warp_y, warp_x,
                   fill_tile, apron_tile=None, apron_width=2):
    H, W = terrain.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    yy_w = yy + warp_y
    xx_w = xx + warp_x
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v = p1 - p0
    vv = float((v * v).sum())
    if vv == 0:
        return
    pts = np.stack([yy_w, xx_w], axis=-1)
    w = pts - p0[None, None, :]
    t = np.clip((w * v[None, None, :]).sum(-1) / vv, 0, 1)
    proj = p0[None, None, :] + t[..., None] * v[None, None, :]
    dist = np.linalg.norm(pts - proj, axis=-1)
    inside = dist < width
    if apron_tile is not None and apron_width > 0:
        apron = (dist < width + apron_width) & ~inside
        terrain[apron] = apron_tile
    terrain[inside] = fill_tile


def _scatter_on_grass(terrain, cue_tile, rate, rng):
    H, W = terrain.shape
    mask = (terrain == GRASS) & (rng.random((H, W)) < rate)
    terrain[mask] = cue_tile


def _clear_corner(terrain, r, c, rad=2, value=GRASS):
    H, W = terrain.shape
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                terrain[rr, cc] = value


def gen_biome(biome: str, size: int, seed: int):
    rng = np.random.default_rng(seed)
    wy, wx = _warp_fields(size, seed)
    terrain = np.full((size, size), GRASS, dtype=np.int8)

    margin = 6
    # diagonal 1: NW → SE (TL to BR)
    d1_p0, d1_p1 = (margin, margin), (size - margin, size - margin)
    # diagonal 2: NE → SW (TR to BL)
    d2_p0, d2_p1 = (margin, size - margin), (size - margin, margin)

    if biome == "aquatic":
        # water + rock barriers; sand cue; NO trees
        _carve_capsule(terrain, d1_p0, d1_p1, width=2.6, warp_y=wy, warp_x=wx,
                       fill_tile=WATER, apron_tile=SAND, apron_width=2)
        _carve_capsule(terrain, d2_p0, d2_p1, width=2.6, warp_y=wy, warp_x=wx,
                       fill_tile=ROCK,  apron_tile=DIRT, apron_width=2)
        _scatter_on_grass(terrain, SAND, 0.07, rng)
    elif biome == "highland":
        # rock + trees barriers; dirt cue; NO water
        _carve_capsule(terrain, d2_p0, d2_p1, width=2.6, warp_y=wy, warp_x=wx,
                       fill_tile=ROCK, apron_tile=DIRT, apron_width=2)
        _carve_capsule(terrain, d1_p0, d1_p1, width=3.2, warp_y=wy, warp_x=wx,
                       fill_tile=TREE)
        _scatter_on_grass(terrain, DIRT, 0.07, rng)
    elif biome == "wetland":
        # water + trees barriers; reeds cue; NO rock
        _carve_capsule(terrain, d1_p0, d1_p1, width=2.6, warp_y=wy, warp_x=wx,
                       fill_tile=WATER, apron_tile=SAND, apron_width=1)
        _carve_capsule(terrain, d2_p0, d2_p1, width=3.2, warp_y=wy, warp_x=wx,
                       fill_tile=TREE)
        _scatter_on_grass(terrain, REEDS, 0.07, rng)
    else:
        raise ValueError(biome)

    spawn = (size - 3, 2)
    target = (2, size - 3)
    _clear_corner(terrain, *spawn, rad=2, value=GRASS)
    _clear_corner(terrain, *target, rad=2, value=GRASS)
    terrain[target] = TARGET
    return terrain, spawn, target


# ── Plotting ───────────────────────────────────────────────────────────────

def draw_cell(ax, terrain, spawn, target, biome, plot_rng):
    # darken the map slightly so trajectories pop
    rgb = (TILE_COLORS[terrain].astype(np.float32) * 0.78).clip(0, 255).astype(np.uint8)
    ax.imshow(rgb, interpolation="nearest")
    succ = {RAFT: 0, HARNESS: 0, MACHETE: 0}
    for skill in (RAFT, HARNESS, MACHETE):
        ctg = ctg_expected_attempts(terrain, target, skill)
        if not np.isfinite(ctg[spawn]):
            continue
        rng = np.random.default_rng(plot_rng.integers(0, 2**31))
        for _ in range(N_ROLLOUTS):
            path, reached = rollout(terrain, spawn, target, skill, ctg, rng,
                                    max_steps=4 * terrain.shape[0])
            succ[skill] += int(reached)
            if len(path) < 2:
                continue
            jit = rng.uniform(-0.18, 0.18, size=path.shape)
            xy = (path + jit)[:, ::-1]  # → (col, row)
            segs = np.concatenate([xy[:-1, None, :], xy[1:, None, :]], axis=1)
            rgba = (*SKILL_RGB[skill], TRAJ_ALPHA)
            ax.add_collection(LineCollection(segs, colors=[rgba] * len(segs),
                                             linewidths=0.7, capstyle="round"))
    sr, sc = spawn
    tr, tc = target
    ax.scatter([sc], [sr], marker="o", s=42, facecolor="#39ff14",
               edgecolor="black", lw=0.6, zorder=6)
    ax.scatter([tc], [tr], marker="*", s=110, facecolor="white",
               edgecolor="black", lw=0.6, zorder=6)
    ax.set_xticks([])
    ax.set_yticks([])
    H, W = terrain.shape
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    # success counts in top-left
    txt = "  ".join(
        f"{SKILL_NAME[s][0].upper()}{succ[s]:>2}"
        for s in (RAFT, HARNESS, MACHETE)
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=7,
            va="top", ha="left", color="white", family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55))


def main():
    size = 64
    n = len(SEEDS)
    fig, axes = plt.subplots(len(BIOMES), n, figsize=(2.6 * n, 2.6 * len(BIOMES)))
    axes = np.atleast_2d(axes)

    plot_rng = np.random.default_rng(0)
    for i, biome in enumerate(BIOMES):
        for j, seed in enumerate(SEEDS):
            terrain, spawn, target = gen_biome(biome, size, seed)
            draw_cell(axes[i, j], terrain, spawn, target, biome,
                      plot_rng=np.random.default_rng(seed * 1000 + i))
            if i == 0:
                axes[i, j].set_title(f"seed {seed}", fontsize=10)
            if j == 0:
                wrong = SKILL_NAME[WRONG_SKILL[biome]]
                axes[i, j].set_ylabel(f"{biome}\n(wrong: {wrong})", fontsize=11)

    handles = [
        Line2D([0], [0], color=SKILL_RGB[s], lw=3.5, label=SKILL_NAME[s])
        for s in (RAFT, HARNESS, MACHETE)
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=10,
               framealpha=0.92)
    fig.suptitle(
        f"Proposed 3-biome / 3-skill design  ·  {N_ROLLOUTS} stochastic "
        f"rollouts/skill  ·  inset counts = reach/{N_ROLLOUTS}",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = ROOT / "mapgen_preview" / "proposed_design.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
