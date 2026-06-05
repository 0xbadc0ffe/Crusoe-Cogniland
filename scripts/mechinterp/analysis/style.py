"""Plot styling — Goodfire-inspired clean light theme + shared label palettes.

Reused by every figure in the pipeline so BT and BTC outputs look consistent and
are close to paper-quality. The categorical colours echo the env's own semantics:
skills reuse the trajectory-grid colours (none=blue / build=yellow / mine=purple),
and the map categories sit on a diverging water<->rock axis (lakes=blue,
balanced=purple, rocky=warm), which matches the ordinal belief score.
"""
from __future__ import annotations

import matplotlib as mpl

# --- panel / canvas (the soft blue-grey panel + white grid in the reference fig) ---
PANEL = "#eef3f8"
GRIDC = "#ffffff"
INKC = "#2b3a4a"

# --- map category = belief (diverging: rocky <- balanced -> lakes, the water axis) ---
CATEGORY_COLORS = {
    "rocky": "#d1495b",      # warm
    "balanced": "#8d6cab",   # purple midpoint
    "lakes": "#2e86c1",      # blue (water)
}
CATEGORY_ORDER = ["rocky", "balanced", "lakes"]
# ordinal belief score in [-1, 1] along the same axis
CATEGORY_ORDINAL = {"rocky": -1.0, "balanced": 0.0, "lakes": 1.0}

# --- committed skill (matches decode_dataset.py / the trajectory grids) ---
SKILL_COLORS = {"none": "#1f5fd0", "build": "#ffd000", "mine": "#a800e6"}
SKILL_ORDER = ["none", "build", "mine"]

# --- accents for centroid paths / fitted curves (orange + yellow, as in the fig) ---
ACCENT_PATH = "#f0892b"      # centroid-to-centroid path
ACCENT_SPLINE = "#ffcf3f"    # manifold-fitting spline
# diverging colormap for continuous belief score / probabilities
BELIEF_CMAP = "RdBu"         # red(rocky) <-> blue(lakes); use with vmin=-1,vmax=1
PROB_CMAP = "magma"


def colors_for(label_kind: str) -> dict:
    return {"category": CATEGORY_COLORS, "skill": SKILL_COLORS}[label_kind]


def order_for(label_kind: str) -> list:
    return {"category": CATEGORY_ORDER, "skill": SKILL_ORDER}[label_kind]


def apply_theme() -> None:
    """Install the light, clean rcParams used across the pipeline."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 150,
        "figure.dpi": 120,
        "axes.facecolor": PANEL,
        "axes.edgecolor": "#c6d2de",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDC,
        "grid.linewidth": 1.1,
        "axes.labelcolor": INKC,
        "axes.titlecolor": INKC,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "text.color": INKC,
        "xtick.color": INKC,
        "ytick.color": INKC,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
