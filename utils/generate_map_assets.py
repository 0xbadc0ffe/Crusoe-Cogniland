"""Generate PNG images for all custom maps and save to assets/maps/."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent))

from cogniland.env.custom_maps import _REGISTRY
from cogniland.env.constants import TERRAIN_LEVELS, palette

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "maps"

# Map terrain index → color (float RGB)
_THRESHOLDS = np.array([TERRAIN_LEVELS[i]["threshold"] for i in range(9)])
_COLOR_LUT = np.array(
    [palette[TERRAIN_LEVELS[i]["color"]] for i in range(9)], dtype=np.float32
) / 255.0


def _terrain_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert a [H, W] heightmap to an [H, W, 3] RGB image."""
    terrain_idx = np.searchsorted(_THRESHOLDS, arr).clip(0, 8)
    return _COLOR_LUT[terrain_idx]


def render_map(name: str, arr: np.ndarray, spawn: tuple, target: tuple) -> plt.Figure:
    rgb = _terrain_rgb(arr)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(rgb, origin="upper", interpolation="nearest")

    # Grid lines to delineate cells
    h, w = arr.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3, alpha=0.25)
    ax.tick_params(which="minor", length=0)

    # Major ticks every 5 cells
    ax.set_xticks(np.arange(0, w, 5))
    ax.set_yticks(np.arange(0, h, 5))
    ax.tick_params(labelsize=7)

    # Spawn and target markers
    sr, sc = spawn
    tr, tc = target
    ax.scatter(sc, sr, c="lime", s=150, marker="o",
               edgecolors="black", linewidth=1.2, zorder=5)
    ax.scatter(tc, tr, c="gold", s=200, marker="*",
               edgecolors="black", linewidth=1.2, zorder=5)

    spawn_patch = mpatches.Patch(color="lime", label=f"Spawn ({sr},{sc})")
    target_patch = mpatches.Patch(color="gold", label=f"Target ({tr},{tc})")
    ax.legend(handles=[spawn_patch, target_patch], fontsize=7,
              loc="upper right", framealpha=0.8)

    ax.set_title(name.replace("_", " ").title(), fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for map_name, (fn, spawn, target) in _REGISTRY.items():
        arr = fn()
        fig = render_map(map_name, arr, spawn, target)
        out_path = OUT_DIR / f"{map_name}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path.relative_to(OUT_DIR.parent.parent)}")

    print(f"\nDone — {len(_REGISTRY)} map(s) written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
