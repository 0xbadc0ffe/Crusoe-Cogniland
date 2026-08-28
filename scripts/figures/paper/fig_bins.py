#!/usr/bin/env python3
"""The eight position bins the belief analyses use, drawn on one real map.

Every probe, transfer matrix and steering axis in the interpretability chapters
is fitted per bin, so the bins deserve to be shown rather than described. The
time axis is `col_rel_wall` -- the agent's column minus the wall column -- and
not raw t, because episodes differ in speed and only the spatial axis lines
them up. On this map the wall sits at column 62, so the two corridor bins land
exactly on the 16-column evidence-free memory corridor.

  PYTHONPATH=src:scripts/mechinterp/belief_report python scripts/figures/paper/fig_bins.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "mechinterp" / "belief_report"))

import data as D  # noqa: E402
from cogniland.bridge_tunnel import tiles as T  # noqa: E402

OUT = REPO / "paper/figures/forkwall_paper"
MAP_ID = 99
PHASE_COL = {"evidence": "#16a34a", "corridor": "#d97706", "past_wall": "#7c3aed"}
PHASE_LABEL = {"evidence": "evidence phase", "corridor": "memory corridor",
               "past_wall": "wall"}
RC = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
      "axes.spines.top": False, "axes.spines.right": False}


def main():
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    rec = pool[MAP_ID]
    H, W = rec.terrain.shape
    wall = int(rec.wall_col)

    with plt.rc_context(RC):
        # equal aspect: height must track the padded data span (64 x 43)
        fig, ax = plt.subplots(figsize=(13.0, 13.0 * 40 / 64))
        ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
        ax.set_xlim(-.5, W - .5)
        ax.set_ylim(H + 4.2, -4.0)          # headroom for labels, footroom for brackets

        # ── the eight bins as alternating bands ──────────────────────────
        for b in range(len(D.BIN_EDGES) - 1):
            lo = max(D.BIN_EDGES[b] + wall, 0)
            hi = min(D.BIN_EDGES[b + 1] + wall, W)
            phase = D.PHASE_OF_BIN[b]
            col = PHASE_COL[phase]
            ax.add_patch(Rectangle((lo - .5, -.5), hi - lo, H,
                                   facecolor=col, alpha=.13 + .07 * (b % 2),
                                   edgecolor="none", zorder=3))
            ax.plot([lo - .5, lo - .5], [-.5, H - .5], color="black", lw=2.0,
                    zorder=5)
            if b == len(D.BIN_EDGES) - 2:          # close the last bin
                ax.plot([hi - .5, hi - .5], [-.5, H - .5], color="black",
                        lw=2.0, zorder=5)
            mid = (lo + hi) / 2 - .5
            name = D.BIN_LABELS[b].split("\n")[0]
            ax.text(mid, -1.3, name, ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=col, zorder=6)
            ax.text(mid, H + 1.4, f"{hi - lo} col", ha="center", va="top",
                    fontsize=7, color="#6b7280", zorder=6)

        ax.text(W / 2 - .5, -3.6, "position bin, by columns to the wall "
                "(col$_{\\mathrm{rel\\,wall}}$)", ha="center", va="bottom",
                fontsize=9.5, color="#374151")

        # ── phase brackets under the map ─────────────────────────────────
        runs, start = [], 0
        for b in range(1, len(D.PHASE_OF_BIN) + 1):
            if b == len(D.PHASE_OF_BIN) or D.PHASE_OF_BIN[b] != D.PHASE_OF_BIN[start]:
                runs.append((start, b - 1, D.PHASE_OF_BIN[start]))
                start = b
        for b0, b1, phase in runs:
            lo = max(D.BIN_EDGES[b0] + wall, 0) - .5
            hi = min(D.BIN_EDGES[b1 + 1] + wall, W) - .5
            y = H + 2.6
            ax.plot([lo, hi], [y, y], color=PHASE_COL[phase], lw=2.4,
                    solid_capstyle="butt", zorder=6)
            ax.text((lo + hi) / 2, y + 0.7, PHASE_LABEL[phase], ha="center",
                    va="top", fontsize=8, color=PHASE_COL[phase], zorder=6)

        # ── landmarks ────────────────────────────────────────────────────
        ax.plot(rec.spawn[1], rec.spawn[0], "o", color="white", mec="black",
                ms=7, zorder=8)
        ax.annotate("spawn", (rec.spawn[1], rec.spawn[0]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color="white", zorder=8)
        for cells, name in ((rec.top_goal_cells, "top"),
                            (rec.bottom_goal_cells, "bottom")):
            good = rec.correct_target in ("either", name)
            for (r, c) in cells:
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                       edgecolor="#22c55e" if good else "#ef4444",
                                       lw=1.8, zorder=8))

        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"The eight position bins, on held-out map {MAP_ID} "
                     f"({rec.category}). Episodes are aligned by column, not by "
                     f"time, so agents of different speeds are compared at the "
                     f"same place.", loc="left", fontsize=10.5, pad=16)
        fig.tight_layout()
        fig.savefig(OUT / "fig_res_bins.png", bbox_inches="tight")
        print("wrote fig_res_bins.png")


if __name__ == "__main__":
    main()
