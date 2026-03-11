"""Map parameter explorer — sweeps all simplex noise parameters.

Generates one labeled PNG grid per parameter axis.
Rows = seeds, columns = parameter values, all others held at defaults.

Usage:
  python scripts/sample_maps.py                        # all axes, defaults
  python scripts/sample_maps.py --axes scale sink_mode # only these axes
  python scripts/sample_maps.py --seeds 5 --output assets/survey
  python scripts/sample_maps.py --size 50              # larger maps

Notes on the noise pipeline (generate_island):
  - SimplexNoise.fractal() controls amplitude via `gain`, NOT `persistence`.
    persistence only affects noise(), which is not called here.
  - hgrid = size * scale  (larger scale → zoomed-out, smoother terrain)
  - lacunarity: frequency multiplier per octave  (>1 adds fine detail faster)
  - gain: amplitude multiplier per octave  (<1 fades each octave out)
  - sink_mode 0: raw heightmap / 1: x³ (pushes land inward) / 2: (2x)² (strong sink)
  - filtering shapes the island boundary: circle, square, diamond
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from cogniland.env.islands import colorize, generate_island
from cogniland.env.types import EnvConfig


# ---------------------------------------------------------------------------
# Default sweep ranges for every tunable noise parameter
# ---------------------------------------------------------------------------

SWEEP_AXES: dict[str, list] = {
    "scale":       [0.1, 0.2, 0.33, 0.5, 0.7],
    "octaves":     [2, 4, 6, 8, 10],
    "lacunarity":  [1.5, 1.8, 2.0, 2.5, 3.0],
    "gain":        [0.3, 0.45, 0.65, 0.8, 0.95],
    "sink_mode":   [0, 1, 2],
    "filtering":   ["circle", "square", "diamond"],
}

# Default config values used when a parameter is held fixed
DEFAULTS = dict(
    size=250,
    scale=0.33,
    octaves=6,
    lacunarity=2.0,
    # gain is not in EnvConfig — handled separately
    sink_mode=1,
    filtering="square",
)

DEFAULT_GAIN = 0.65  # SimplexNoise fractal() default


# ---------------------------------------------------------------------------
# Patched generate_island that accepts a `gain` override
# ---------------------------------------------------------------------------

def generate_island_with_gain(config: EnvConfig, gain: float) -> torch.Tensor:
    """Same as generate_island but passes `gain` to fractal()."""
    import math
    import random
    import numpy as np
    from cogniland.simplexnoise.noise import SimplexNoise, normalize

    size = config.size
    scale = size * config.scale
    sn = SimplexNoise(num_octaves=config.octaves, persistence=config.persistence, dimensions=2)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    world = torch.zeros(size, size)
    for i in range(size):
        for j in range(size):
            world[i, j] = normalize(sn.fractal(i, j, hgrid=scale, lacunarity=config.lacunarity, gain=gain))

    if config.sink_mode == 1:
        world = world ** 3
    elif config.sink_mode == 2:
        world = (2 * world) ** 2

    world = world / torch.max(world)

    if config.filtering:
        center = size // 2
        circle_grad = torch.zeros(size, size)
        for y in range(size):
            for x in range(size):
                dx = abs(x - center)
                dy = abs(y - center)
                if config.filtering == "circle":
                    dist = math.sqrt(dx * dx + dy * dy)
                elif config.filtering == "diamond":
                    dist = dx + dy
                else:  # square
                    dist = max(dx ** 2, dy ** 2)
                circle_grad[y, x] = dist
        circle_grad = circle_grad / torch.max(circle_grad)
        circle_grad = -(circle_grad - 0.5) * 2.0
        for y in range(size):
            for x in range(size):
                if circle_grad[y, x] > 0:
                    circle_grad[y, x] *= 20
        circle_grad = circle_grad / torch.max(circle_grad)
        world_noise = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                world_noise[i, j] = world[i, j] * circle_grad[i, j]
                if world_noise[i, j] > 0:
                    world_noise[i, j] *= 20
        world_noise = world_noise / torch.max(world_noise)
        world = world_noise

    return world


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def make_map(config: EnvConfig, gain: float) -> torch.Tensor:
    """Generate and colorize a map, returning an [H, W, 3] uint8 numpy array."""
    world = generate_island_with_gain(config, gain)
    return colorize(world, config).numpy().astype("uint8")


def render_axis(
    param: str,
    values: list,
    seeds: list[int],
    base: dict,
    output_dir: Path,
) -> None:
    """One PNG grid for a single parameter axis."""
    n_rows = len(seeds)
    n_cols = len(values)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(f"{param}", fontsize=13, y=1.01)

    for col, val in enumerate(values):
        for row, seed in enumerate(seeds):
            kwargs = {k: v for k, v in base.items() if k != "gain"}
            gain = base.get("gain", DEFAULT_GAIN)

            if param == "gain":
                gain = val
            else:
                kwargs[param] = val
            kwargs["seed"] = seed

            cfg = EnvConfig(**kwargs)
            img = make_map(cfg, gain)

            ax = axes[row][col]
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{val}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"seed {seed}", fontsize=8)

    plt.tight_layout()
    path = output_dir / f"{param}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {param:12s}  →  {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep simplex noise parameters and save PNG grids."
    )
    parser.add_argument(
        "--axes", nargs="+", default=list(SWEEP_AXES.keys()),
        choices=list(SWEEP_AXES.keys()),
        help="Which parameter axes to sweep (default: all)",
    )
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per combo")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--size", type=int, default=DEFAULTS["size"], help="Map size in tiles (default: 250)")
    parser.add_argument("--output", default="assets/maps_survey")

    # Per-axis value overrides
    for name, defaults in SWEEP_AXES.items():
        t = type(defaults[0])
        parser.add_argument(f"--{name}", nargs="+", type=t, default=None,
                            help=f"Override sweep values for {name} (default: {defaults})")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    base = {**DEFAULTS, "size": args.size, "gain": DEFAULT_GAIN}

    print(f"Map size    : {args.size}×{args.size}")
    print(f"Seeds       : {seeds}")
    print(f"Axes        : {args.axes}")
    print(f"Output      : {output_dir}/\n")

    for param in args.axes:
        values = getattr(args, param.replace("-", "_")) or SWEEP_AXES[param]
        render_axis(param, values, seeds, base, output_dir)

    print(f"\nDone — {len(args.axes)} grid(s) saved.")


if __name__ == "__main__":
    main()
