"""Generate island maps for the Survival Kit strategy game.

Four biome types, all from generate_island() with different sink_mode + thresholds:
  - balanced:    default config (sink_mode=1, default thresholds)
  - archipelago: sink_mode=1, lower thresholds → more water
  - highland:    sink_mode=0, lower mountain thresholds + ridge overlay
  - grassland:   sink_mode=0, raised thresholds → mostly grassland

Usage:
    python scripts/generate_strategy_maps.py --preview
    python scripts/generate_strategy_maps.py --preview --simulate
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image as PILImage
from cogniland.env.islands import colorize
from cogniland.env.types import EnvConfig, TerrainDef, _DEFAULT_TERRAINS

SPRITE_DIR = Path(__file__).resolve().parent.parent / "data" / "strategy_maps" / "sprites"

# ── Constants ───────────────────────────────────────────────────────────────

SIZE = 250
TERRAIN_NAMES = [t.name for t in _DEFAULT_TERRAINS]
DEFAULT_THRESHOLDS = np.array([t.threshold for t in _DEFAULT_TERRAINS])

#                                   ocean   dw    water  beach  sandy  grass  forest rocky  mtn
THRESHOLDS_ARCHIPELAGO = np.array([0.015, 0.05,  0.15,  0.18,  0.22,  0.45,  0.75,  0.85,  1.0])
THRESHOLDS_GRASSLAND   = np.array([0.20,  0.25,  0.28,  0.30,  0.35,  0.75,  0.95,  0.98,  1.0])
THRESHOLDS_HIGHLAND    = np.array([0.20,  0.25,  0.28,  0.32,  0.36,  0.45,  0.65,  0.80,  1.0])

BIOME_THRESHOLDS = {
    "balanced":    DEFAULT_THRESHOLDS,
    "archipelago": THRESHOLDS_ARCHIPELAGO,
    "grassland":   THRESHOLDS_GRASSLAND,
    "highland":    THRESHOLDS_HIGHLAND,
}

BIOME_SINK_MODE = {
    "balanced": 0,
    "archipelago": 1,
    "grassland": 0,
    "highland": 0,
}

HP_DRAIN = {
    "ocean": -20.0, "deep_water": -8.0, "water": -3.0,
    "beach": -0.50, "sandy": -0.50, "grassland": -0.50,
    "forest": -1.0, "rocky": -3.0, "mountains": -8.0,
}
TOOL_EFFECTS = {
    "raft":       {"ocean": -1.5, "deep_water": -0.3, "water": -0.05,
                   "beach": -2.0, "sandy": -2.0, "grassland": -1.5,
                   "forest": -2.0, "rocky": -5.0, "mountains": -12.0},
    "rope":       {"rocky": -0.05, "mountains": -0.2, "forest": -0.10},
    "provisions": {"grassland": 0.10, "sandy": -0.05, "beach": -0.05,
                   "forest": -2.5, "rocky": -4.0, "mountains": -10.0},
}
INIT_HP = 100.0

ALL_BIOMES = ["balanced", "archipelago", "highland", "grassland"]


# ── Per-biome terrain helpers ─────────────────────────────────────────────

def _biome_compiled(biome: str):
    """Compiled terrain data with biome-specific thresholds (for colorize)."""
    thresholds = BIOME_THRESHOLDS[biome]
    terrains = tuple(
        TerrainDef(t.name, thresholds[i], t.move_cost, t.res_rate, t.hp_rate,
                   t.visibility, t.color, t.tags)
        for i, t in enumerate(_DEFAULT_TERRAINS)
    )
    return EnvConfig(terrains=terrains).compile_terrain("cpu")


def _terrain_idx(height: np.ndarray, biome: str) -> np.ndarray:
    return np.searchsorted(BIOME_THRESHOLDS[biome], height).clip(0, len(TERRAIN_NAMES) - 1)


def _terrain_fractions(hm: np.ndarray, biome: str) -> dict[str, float]:
    idx = _terrain_idx(hm, biome)
    island = idx > 0
    n = island.sum()
    if n == 0:
        return {name: 0.0 for name in TERRAIN_NAMES}
    return {name: float(((idx == i) & island).sum()) / n
            for i, name in enumerate(TERRAIN_NAMES)}


# ── Ridge overlay (highland only) ────────────────────────────────────────

def _fbm_noise(size: int, scale: float, octaves: int = 6,
               persistence: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    result = np.zeros((size, size), dtype=np.float64)
    amplitude, freq = 1.0, 1.0
    for _ in range(octaves):
        noise = rng.uniform(-1, 1, (size, size))
        sigma = scale / freq
        if sigma > 0.5:
            noise = gaussian_filter(noise, sigma=sigma, mode="wrap")
        result += noise * amplitude
        amplitude *= persistence
        freq *= 2.0
    lo, hi = result.min(), result.max()
    if hi - lo > 1e-10:
        result = (result - lo) / (hi - lo)
    return result.astype(np.float32)


def _add_ridge(hm: np.ndarray, size: int, seed: int,
               rng: np.random.RandomState) -> None:
    angle = rng.uniform(0, math.pi)
    cx, cy = size / 2, size / 2
    Y, X = np.mgrid[:size, :size]
    dist = (X - cx) * math.sin(angle) - (Y - cy) * math.cos(angle)
    warp = _fbm_noise(size, scale=30, octaves=4, seed=seed + 8000)
    dist = np.abs(dist + (warp - 0.5) * 60)
    ridge = np.clip(1.0 - dist / 15, 0, 1) ** 1.3
    land = hm > 0.05
    hm[land] = hm[land] + ridge[land] * (0.98 - hm[land]) * 0.8
    np.clip(hm, 0, 0.99, out=hm)


# ── Heightmap generation ─────────────────────────────────────────────────

def _generate_heightmap(size: int, biome: str, seed: int) -> np.ndarray:
    from cogniland.env.islands import generate_island
    from cogniland.env.types import MapGenConfig

    rng = np.random.RandomState(seed)
    sink_mode = BIOME_SINK_MODE[biome]

    config = EnvConfig(map_generation=MapGenConfig(seed=seed, sink_mode=sink_mode))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    hm = generate_island(config).numpy()

    if biome == "highland":
        _add_ridge(hm, size, seed, rng)

    return hm.astype(np.float32)


# ── Spawn / target placement ─────────────────────────────────────────────

def _sample_spawn_target(hm: np.ndarray, biome: str, seed: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Sample spawn and target: both on land (not water), maximizing distance."""
    rng = np.random.RandomState(seed + 99999)
    thresholds = BIOME_THRESHOLDS[biome]
    water_thresh = thresholds[2]  # max water threshold

    # Collect all land positions
    land_rows, land_cols = np.where(hm > water_thresh)
    if len(land_rows) < 2:
        center = SIZE // 2
        return (center, center), (center, center)

    best_spawn, best_target, best_dist = None, None, 0
    for _ in range(500):
        i = rng.randint(len(land_rows))
        j = rng.randint(len(land_rows))
        s = (int(land_rows[i]), int(land_cols[i]))
        t = (int(land_rows[j]), int(land_cols[j]))
        d = math.hypot(s[0] - t[0], s[1] - t[1])
        if d > best_dist:
            best_spawn, best_target, best_dist = s, t, d
            if d > 180:
                break
    return best_spawn, best_target


# ── Tool placement ────────────────────────────────────────────────────────

@dataclass
class ToolSite:
    tool: str
    row: int
    col: int


def _place_tool_sites(spawn: tuple[int, int], rng: np.random.RandomState,
                      dist: int = 3) -> list[ToolSite]:
    sr, sc = spawn
    base_angle = rng.uniform(0, 2 * math.pi)
    sites = []
    for i, tool in enumerate(["raft", "rope", "provisions"]):
        angle = base_angle + i * 2 * math.pi / 3
        r = max(1, min(SIZE - 2, int(round(sr + dist * math.sin(angle)))))
        c = max(1, min(SIZE - 2, int(round(sc + dist * math.cos(angle)))))
        sites.append(ToolSite(tool=tool, row=r, col=c))
    return sites


# ── Map dataclass & generation ────────────────────────────────────────────

@dataclass
class StrategyMap:
    heightmap: torch.Tensor
    spawn: tuple[int, int]
    target: tuple[int, int]
    tool_sites: list[ToolSite]
    terrain_fractions: dict[str, float]
    biome: str
    seed: int


def generate_strategy_map(seed: int, biome: str) -> StrategyMap:
    rng = np.random.RandomState(seed)
    hm = _generate_heightmap(SIZE, biome, seed)
    spawn, target = _sample_spawn_target(hm, biome, seed)
    sites = _place_tool_sites(spawn, rng)

    return StrategyMap(
        heightmap=torch.from_numpy(hm),
        spawn=spawn, target=target, tool_sites=sites,
        terrain_fractions=_terrain_fractions(hm, biome),
        biome=biome, seed=seed,
    )


def generate_strategy_dataset(base_seed: int, count_per_biome: int = 3) -> list[StrategyMap]:
    maps = []
    for biome in ALL_BIOMES:
        for i in range(count_per_biome):
            m = generate_strategy_map(base_seed + i, biome=biome)
            maps.append(m)
            f = m.terrain_fractions
            print(f"  {m.biome:>12} seed={m.seed} "
                  f"forest={f.get('forest',0):.0%} rocky={f.get('rocky',0):.0%} "
                  f"mtn={f.get('mountains',0):.0%}")
    return maps


# ── A* simulation ────────────────────────────────────────────────────────

def _hp_cost(terrain_name: str, tool: str | None) -> float:
    base = HP_DRAIN.get(terrain_name, -0.3)
    if tool and terrain_name in TOOL_EFFECTS.get(tool, {}):
        return TOOL_EFFECTS[tool][terrain_name]
    return base


def _simulate_astar(hm: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
                    tool: str | None, biome: str) -> dict:
    idx_map = _terrain_idx(hm, biome)
    H, W = hm.shape
    min_cost = 0.05

    counter = 0
    h0 = (abs(start[0] - goal[0]) + abs(start[1] - goal[1])) * min_cost
    open_set = [(h0, 0.0, counter, start)]
    best_g = {start: 0.0}
    came_from = {}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    found = False

    while open_set:
        f, g, _, pos = heapq.heappop(open_set)
        if pos == goal:
            found = True
            break
        if g > best_g.get(pos, float("inf")):
            continue
        for dr, dc in dirs:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < H and 0 <= nc < W:
                tname = TERRAIN_NAMES[int(idx_map[nr, nc])]
                drain = _hp_cost(tname, tool)
                edge_cost = max(abs(drain), 0.05)
                if drain <= -10.0:
                    edge_cost = 200.0
                elif drain <= -5.0:
                    edge_cost = 50.0
                ng = g + edge_cost
                if ng < best_g.get((nr, nc), float("inf")):
                    best_g[(nr, nc)] = ng
                    came_from[(nr, nc)] = pos
                    h = (abs(nr - goal[0]) + abs(nc - goal[1])) * min_cost
                    counter += 1
                    heapq.heappush(open_set, (ng + h, ng, counter, (nr, nc)))

    if not found:
        return {"path": [], "hp_trace": [], "length": 0, "final_hp": 0,
                "min_hp": 0, "alive": False, "outcome": "no_path"}

    path = [goal]
    pos = goal
    while pos != start:
        pos = came_from[pos]
        path.append(pos)
    path.reverse()

    hp = INIT_HP
    hp_trace = [hp]
    for p in path[1:]:
        hp += _hp_cost(TERRAIN_NAMES[int(idx_map[p[0], p[1]])], tool)
        hp_trace.append(hp)

    alive = all(h > 0 for h in hp_trace)
    return {"path": path, "hp_trace": hp_trace, "length": len(path) - 1,
            "final_hp": hp_trace[-1], "min_hp": min(hp_trace),
            "alive": alive, "outcome": "reached" if alive else "died"}


def simulate_routes(smap: StrategyMap) -> dict[str, dict]:
    hm = smap.heightmap.numpy()
    biome = smap.biome
    spawn, target = smap.spawn, smap.target
    results = {}

    for tool_name in [None, "raft", "rope", "provisions"]:
        label = tool_name or "none"
        if not tool_name:
            results[label] = _simulate_astar(hm, spawn, target, None, biome)
            continue

        site = next((s for s in smap.tool_sites if s.tool == tool_name), None)
        if not site:
            results[label] = {"outcome": "no_site"}
            continue
        leg1 = _simulate_astar(hm, spawn, (site.row, site.col), None, biome)
        if leg1["outcome"] == "no_path":
            results[label] = {"outcome": "no_path_to_tool"}
            continue
        leg2 = _simulate_astar(hm, (site.row, site.col), target, tool_name, biome)
        if leg2["outcome"] == "no_path":
            results[label] = {"outcome": "no_path_to_target"}
            continue

        hp = leg1["hp_trace"][-1] if leg1["hp_trace"] else INIT_HP
        leg2_hp = []
        for p in leg2["path"][1:]:
            hp += _hp_cost(TERRAIN_NAMES[int(_terrain_idx(hm[p[0], p[1]], biome))], tool_name)
            leg2_hp.append(hp)
        full_path = leg1["path"] + leg2["path"][1:]
        full_hp = leg1["hp_trace"] + leg2_hp
        results[label] = {
            "path": full_path, "hp_trace": full_hp,
            "length": len(full_path) - 1,
            "final_hp": full_hp[-1] if full_hp else 0,
            "min_hp": min(full_hp) if full_hp else 0,
            "alive": all(h > 0 for h in full_hp),
            "outcome": "reached" if all(h > 0 for h in full_hp) else "died",
        }
    return results


# ── Visualization ────────────────────────────────────────────────────────

def _preview_grid(maps: list[StrategyMap], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compiled_cache = {b: _biome_compiled(b) for b in ALL_BIOMES}

    ncols = len(ALL_BIOMES)
    nrows = max(sum(1 for m in maps if m.biome == b) for b in ALL_BIOMES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    # Load sprites
    sprites = {}
    for name in ["agent", "raft", "rope", "provisions"]:
        img = PILImage.open(SPRITE_DIR / f"{name}.png").convert("RGBA")
        sprites[name] = np.array(img)

    def _place_sprite(ax, sprite, col, row, zoom=3.0):
        im = OffsetImage(sprite, zoom=zoom)
        ab = AnnotationBbox(im, (col, row), frameon=False, zorder=6)
        ax.add_artist(ab)

    # Group maps by biome
    biome_maps = {b: [m for m in maps if m.biome == b] for b in ALL_BIOMES}

    for col, biome in enumerate(ALL_BIOMES):
        for row, m in enumerate(biome_maps[biome]):
            ax = axes[row, col]
            rgb = colorize(m.heightmap, compiled_cache[m.biome]).numpy().astype("uint8")
            ax.imshow(rgb)

            _place_sprite(ax, sprites["agent"], m.spawn[1], m.spawn[0])
            ax.scatter(m.target[1], m.target[0], c="red", s=120, marker="*",
                       edgecolors="k", linewidth=1.2, zorder=6)
            for site in m.tool_sites:
                _place_sprite(ax, sprites[site.tool], site.col, site.row, zoom=2.5)

            # Crop to land bounding box
            land = m.heightmap.numpy() > 0
            rows_any = np.where(land.any(axis=1))[0]
            cols_any = np.where(land.any(axis=0))[0]
            if len(rows_any) > 0 and len(cols_any) > 0:
                margin = 5
                ax.set_xlim(max(0, cols_any[0] - margin), min(SIZE - 1, cols_any[-1] + margin))
                ax.set_ylim(min(SIZE - 1, rows_any[-1] + margin), max(0, rows_any[0] - margin))

            f = m.terrain_fractions
            title = f"seed={m.seed}\n" \
                    f"forest={f.get('forest',0):.0%} rocky={f.get('rocky',0):.0%} " \
                    f"mtn={f.get('mountains',0):.0%}"
            if row == 0:
                title = f"{biome.upper()}\n{title}"
            ax.set_title(title, fontsize=8)
            ax.set_axis_off()

        # Hide unused rows
        for row in range(len(biome_maps[biome]), nrows):
            axes[row, col].set_visible(False)

    from matplotlib.lines import Line2D
    tool_colors = {"raft": "cyan", "rope": "gray", "provisions": "red"}
    legend = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="royalblue",
               markersize=10, markeredgecolor="k", label="Agent"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markersize=12, markeredgecolor="k", label="Target"),
    ] + [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=tool_colors[t], markersize=10,
               markeredgecolor="k", label=t.capitalize())
        for t in ["raft", "rope", "provisions"]
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("Strategy Maps", fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Preview saved: {output_path}")


def _plot_simulation(smap, routes, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compiled = _biome_compiled(smap.biome)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax = axes[0]
    rgb = colorize(smap.heightmap, compiled).numpy().astype("uint8")
    ax.imshow(rgb)
    colors = {"none": "white", "raft": "cyan", "rope": "orange", "provisions": "lime"}
    for label, res in routes.items():
        if not res.get("path"):
            continue
        path = np.array(res["path"])
        style = "-" if res.get("alive") else "--"
        ax.plot(path[:, 1], path[:, 0], style, color=colors[label],
                linewidth=1.5, alpha=0.8, label=f"{label} ({res['length']})")
    for site in smap.tool_sites:
        ax.scatter(site.col, site.row, c=colors[site.tool], marker="o", s=100,
                   edgecolors="k", linewidth=1, zorder=5)
    ax.scatter(smap.spawn[1], smap.spawn[0], c="lime", s=120, marker="o",
               edgecolors="k", linewidth=1.5, zorder=6)
    ax.scatter(smap.target[1], smap.target[0], c="gold", s=160, marker="*",
               edgecolors="k", linewidth=1.5, zorder=6)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"seed={smap.seed} [{smap.biome}]")
    ax.set_axis_off()

    ax = axes[1]
    for label, res in routes.items():
        if not res.get("hp_trace"):
            continue
        hp = res["hp_trace"]
        style = "-" if res.get("alive") else "--"
        ax.plot(hp, style, color=colors[label], linewidth=1.5,
                label=f"{label}: final={hp[-1]:.0f} min={min(hp):.0f}")
    ax.axhline(0, color="red", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("HP")
    ax.set_title("HP over trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=3, help="Maps per biome")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--output-dir", type=str, default="data/strategy_maps")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    print(f"Generating {args.count} maps per biome (seed={args.seed})")
    maps = generate_strategy_dataset(args.seed, count_per_biome=args.count)

    out_dir.mkdir(parents=True, exist_ok=True)
    heightmaps = torch.stack([m.heightmap for m in maps])
    save_path = out_dir / f"strategy_seed{args.seed}_n{len(maps)}.pt"
    torch.save({"maps": heightmaps, "metadata": {
        "seed": args.seed, "count_per_biome": args.count,
        "tool_sites": [[{"tool": s.tool, "row": s.row, "col": s.col}
                        for s in m.tool_sites] for m in maps],
        "spawns": [m.spawn for m in maps],
        "targets": [m.target for m in maps],
        "biomes": [m.biome for m in maps],
    }}, save_path)
    print(f"\nSaved: {save_path} ({len(maps)} maps)")

    if args.preview:
        _preview_grid(maps, out_dir / "preview.png")

    if args.simulate:
        print("\nSimulating...")
        sim_dir = out_dir / "simulations"
        sim_dir.mkdir(exist_ok=True)
        summary = []
        for m in maps:
            print(f"\n  seed={m.seed} [{m.biome}]: "
                  f"spawn=({m.spawn[0]},{m.spawn[1]}) target=({m.target[0]},{m.target[1]})")
            routes = simulate_routes(m)
            for label, res in routes.items():
                s = res.get("outcome", "?")
                l = res.get("length", 0)
                fhp = res.get("final_hp", 0)
                mhp = res.get("min_hp", 0)
                print(f"    {label:>12}: {s:>8} len={l:>4} final_hp={fhp:>6.1f} min_hp={mhp:>6.1f}")
                summary.append({"seed": m.seed, "biome": m.biome, "tool": label,
                                "outcome": s, "length": l, "final_hp": fhp, "min_hp": mhp})
            _plot_simulation(m, routes, sim_dir / f"sim_{m.biome}_seed{m.seed}.png")

        print("\n" + "=" * 70)
        print("SUMMARY")
        for biome in ALL_BIOMES:
            print(f"\n  {biome.upper()}")
            print(f"  {'tool':>12} | {'alive':>5} | {'died':>4} | {'avg_hp':>8}")
            print(f"  {'-' * 45}")
            for tool in ["none", "raft", "rope", "provisions"]:
                entries = [s for s in summary
                           if s["biome"] == biome and s["tool"] == tool and s["length"] > 0]
                if not entries:
                    continue
                alive = sum(1 for e in entries if e["outcome"] == "reached")
                died = len(entries) - alive
                avg_hp = np.mean([e["final_hp"] for e in entries])
                print(f"  {tool:>12} | {alive:>5} | {died:>4} | {avg_hp:>8.1f}")


if __name__ == "__main__":
    main()
