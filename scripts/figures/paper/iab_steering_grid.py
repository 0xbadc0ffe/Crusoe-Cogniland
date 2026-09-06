#!/usr/bin/env python3
"""Appendix figure: the paper's PPO steering (act-11 gated clamp at the act-5
operating points) on held-out maps of the three categories, drawn from the
stored rollouts of the all-eligible run (outputs/behavior_steering/act11/
rows_all_{lakes,balanced,rocky}.json). One row per map, one panel per arm
(unsteered / suppress bridge / suppress tunnel), six rollouts overlaid,
identical seeds across arms. Panel labels carry the map seed, the share of
rollouts that took the top flag and the mean tool counts.

  PYTHONPATH=src:scripts/mechinterp/behavior_steering python scripts/figures/paper/iab_steering_grid.py --per-cat 3
  --layout wide  puts the three categories side by side (nine columns) instead of stacked.
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ACT = REPO / "outputs/behavior_steering/act11"
POOL = REPO / "data/bridge_tunnel/forkwall6k/test.pkl"
OUTS = [REPO / "paper/iab2026/paper/figures", REPO / "paper/figures/iab2026"]
ARMS = [("unsteered", "#6b7280"), ("suppress bridge", "#0e7490"), ("suppress tunnel", "#b91c1c")]
CATS = [("lakes", "bottom"), ("balanced", "either"), ("rocky", "top")]
INK = "#374151"
# rollout colours: warm tones read against green grass, blue water and grey rock
PALETTE = ["#dc2626", "#f97316", "#facc15", "#ec4899", "#7c3aed", "#f8fafc"]


def label(rs, mn, bd, seed):
    n = len(rs); top = sum(r["door"] == "top" for r in rs); to = sum(r["timeout"] for r in rs)
    l1 = f"seed {seed}   top flag {100 * top / n:.0f}%" + (f"   TO {to}/{n}" if to else "")
    l2 = f"{mn:.1f}$\\times$tunnel   {bd:.1f}$\\times$bridge"
    return f"{l1}\n{l2}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=3)
    ap.add_argument("--layout", choices=["stack", "wide"], default="stack")
    ap.add_argument("--seed", type=int, default=0, help="rng seed for the map sample")
    ap.add_argument("--markers", action="store_true", help="draw tool-event glyphs")
    ap.add_argument("--palette", choices=["warm", "turbo"], default="warm", help="rollout colours")
    a = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import grid_fig
    from grid_fig import _draw_panel, RC
    grid_fig.PALETTE = PALETTE if a.palette == "warm" else [matplotlib.colors.to_hex(matplotlib.colormaps["turbo"](x)) for x in np.linspace(0.05, 0.95, 6)]
    from cogniland.bridge_tunnel import tiles as T

    pool = pickle.load(open(POOL, "rb"))
    rng = np.random.default_rng(a.seed)
    blocks = []
    for cat, flag in CATS:
        rows = json.load(open(ACT / f"rows_all_{cat}.json"))
        mids = sorted({r["map_id"] for r in rows})
        pick = sorted(rng.choice(mids, size=min(a.per_cat, len(mids)), replace=False).tolist())
        blocks.append((cat, flag, pick, rows))
    cmap = matplotlib.colormaps["turbo"]

    with plt.rc_context(RC | {"font.size": 8}):
        if a.layout == "stack":
            pw, ph = 2.25, 1.12
            fig = plt.figure(figsize=(3 * pw + 0.3, a.per_cat * 3 * (ph + 0.42) + 3 * 0.7))
            outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.12)
            for bi, (cat, flag, pick, rows) in enumerate(blocks):
                inner = gridspec.GridSpecFromSubplotSpec(len(pick) + 1, 3, subplot_spec=outer[bi], wspace=0.06,
                                                         hspace=0.62, height_ratios=[0.6] + [1] * len(pick))
                for j, (tag, col) in enumerate(ARMS):          # header row: block title + arm names
                    hx = fig.add_subplot(inner[0, j]); hx.axis("off")
                    hx.text(.5, .0, tag, transform=hx.transAxes, ha="center", va="bottom", fontsize=9.5, weight="bold", color=col)
                    if j == 0:
                        hx.text(0, 1.0, f"{cat} maps  (rewarding flag: {flag})", transform=hx.transAxes,
                                ha="left", va="top", fontsize=10.5, weight="bold", color=INK)
                for i, mid in enumerate(pick):
                    for j, (tag, col) in enumerate(ARMS):
                        ax = fig.add_subplot(inner[i + 1, j])
                        rs = [r for r in rows if r["map_id"] == mid and r["arm"] == tag]
                        mn, bd = _draw_panel(ax, pool[mid], rs, cmap, T, plt, None, markers=a.markers)
                        ax.set_title(label(rs, mn, bd, pool[mid].seed), fontsize=6.6, loc="left", pad=2.5, color=INK, linespacing=1.15)
        else:
            pw, ph = 2.2, 1.1
            fig = plt.figure(figsize=(9 * pw + 2 * 0.6, a.per_cat * (ph + 0.34) + 0.6))
            outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.10, top=0.965, bottom=0.005, left=0.01, right=0.99)
            fig.suptitle("PPO+GRU on held-out maps, six stochastic rollouts each, identical seeds across the three arms.  "
                         "Gated gradient clamp at the frozen operating points."
                         + ("  X = tunnelled block, open square = placed bridge." if a.markers else ""),
                         y=0.995, fontsize=11)
            for bi, (cat, flag, pick, rows) in enumerate(blocks):
                inner = gridspec.GridSpecFromSubplotSpec(len(pick) + 1, 3, subplot_spec=outer[bi], wspace=0.06,
                                                         hspace=0.62, height_ratios=[0.6] + [1] * len(pick))
                for j, (tag, col) in enumerate(ARMS):
                    hx = fig.add_subplot(inner[0, j]); hx.axis("off")
                    hx.text(.5, .0, tag, transform=hx.transAxes, ha="center", va="bottom", fontsize=9.5, weight="bold", color=col)
                    if j == 1:
                        hx.text(.5, 1.0, f"{cat} maps  (rewarding flag: {flag})", transform=hx.transAxes,
                                ha="center", va="top", fontsize=11, weight="bold", color=INK)
                for i, mid in enumerate(pick):
                    for j, (tag, col) in enumerate(ARMS):
                        ax = fig.add_subplot(inner[i + 1, j])
                        rs = [r for r in rows if r["map_id"] == mid and r["arm"] == tag]
                        mn, bd = _draw_panel(ax, pool[mid], rs, cmap, T, plt, None, markers=a.markers)
                        ax.set_title(label(rs, mn, bd, pool[mid].seed), fontsize=6.4, loc="left", pad=2.5, color=INK, linespacing=1.15)
        name = "fig_steering_grid.png" if a.layout == "stack" else "fig_steering_grid_wide.png"
        for o in OUTS:
            o.mkdir(parents=True, exist_ok=True); fig.savefig(o / name, bbox_inches="tight")
        plt.close(fig)
    print("wrote", name, "maps:", {c: p for c, _, p, _ in blocks})


if __name__ == "__main__":
    main()
