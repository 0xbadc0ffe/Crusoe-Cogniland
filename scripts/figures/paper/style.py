#!/usr/bin/env python3
"""Shared look and shared maths for every Cogniland paper figure.

Edit here to change something that should change *everywhere* — a colour, a
font size, the DPI. Anything that is words rather than style lives in text.py.
"""
from __future__ import annotations

import numpy as np

# ── palette ──────────────────────────────────────────────────────────────
# One colour per agent, used consistently across every figure and the report's
# CSS. LIGHT is the same hue lifted for translucent overlays (many rollouts
# stacked on one axes), so bundles read as the same agent.
AGENT = {"ppo": "#d97706", "dreamer": "#2563eb", "storm": "#16a34a"}
AGENT_LIGHT = {"ppo": "#f59e0b", "dreamer": "#60a5fa", "storm": "#34d399"}

# semantic colours, independent of the agent hues
OK = "#22c55e"          # correct door / success
BAD = "#ef4444"         # wrong door / decoy
WARN = "#f59e0b"        # timeout
GREY = "#6b7280"        # reference lines, de-emphasised annotation
INK = "#111827"         # strong foreground marks
TRAJ = "#fde68a"        # a single trajectory drawn over a map
ZOOM = "#facc15"        # zoom / region-of-interest outlines
PASSAGE = "#38bdf8"     # the gap through the tree wall

# per-map-type colours, for panels split by category
CAT = {"balanced": "#94a3b8", "lakes": "#3b82f6", "rocky": "#a16207"}

# ── canonical orderings ──────────────────────────────────────────────────
AGENTS = ("ppo", "dreamer", "storm")
# rocky -> balanced -> lakes reads as a terrain gradient; keep it everywhere
CATS = ("rocky", "balanced", "lakes")
# ...except where the story is navigation-then-memory, which wants balanced first
CATS_EVAL = ("balanced", "lakes", "rocky")

# ── matplotlib defaults ──────────────────────────────────────────────────
RC = {"figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
      "axes.titlesize": 9.5, "axes.labelsize": 9,
      "axes.spines.top": False, "axes.spines.right": False,
      "legend.frameon": False}

# the map-generation chapter packs more panels per plate, so it runs smaller
RC_DENSE = {**RC, "figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
            "axes.titlesize": 9}


def rc(dense: bool = False) -> dict:
    return RC_DENSE if dense else RC


# ── shared maths ─────────────────────────────────────────────────────────

def smooth(y, k=9):
    """Moving average that shrinks its window at the edges.

    A plain `np.convolve(..., "same")` pads with zeros and so invents a dive at
    the right-hand end of every curve — which looked exactly like a training
    collapse. This keeps the endpoints honest.
    """
    y = np.asarray(y, float)
    if k <= 1 or y.size < 3:
        return y
    out = np.empty_like(y)
    half = k // 2
    for i in range(y.size):
        a, b = max(0, i - half), min(y.size, i + half + 1)
        out[i] = y[a:b].mean()
    return out


def wilson(k, n, z=1.96):
    """Wilson score interval — behaves near 0 and 1 where the normal one doesn't."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def xy(series, key, xscale=1.0):
    """Pull one logged series out of training_data.json as (x, y) arrays."""
    pts = series.get(key) or []
    if not pts:
        return np.array([]), np.array([])
    a = np.asarray(pts, float)
    return a[:, 0] * xscale, a[:, 1]


def savefig(fig, out_dir, name, **kw):
    """Write a figure and say so, so make_figures.py output is legible."""
    from pathlib import Path
    p = Path(out_dir) / name
    fig.savefig(p, bbox_inches="tight", **kw)
    print(f"  wrote {p.name}")
    return p
