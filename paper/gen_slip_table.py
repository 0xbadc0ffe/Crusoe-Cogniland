#!/usr/bin/env python3
"""Slip-probability table: rows = land/rock/water, cols = noskill/harness/raft.

Values are read straight from ``skills.slip_chance`` (current SLIP_PROB_DEFAULT),
rendered as a viridis heatmap on a 0..1 scale with the value written in each cell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cogniland.nav.skills as sk  # noqa: E402
from cogniland.nav.tiles import DIRT, GRASS, ROCK, SAND, TREE, WATER  # noqa: E402

FIGDIR = Path(__file__).resolve().parent / "figures"

ROWS = [("grass", GRASS), ("dirt", DIRT), ("sand", SAND),
        ("water", WATER), ("rock", ROCK), ("tree", TREE)]
COLS = [("noskill", sk.NONE), ("harness", sk.HARNESS), ("raft", sk.RAFT)]

M = np.array([[sk.slip_chance(obj, tile) for _, obj in COLS]
              for _, tile in ROWS])

fig, ax = plt.subplots(figsize=(4.2, 4.0))
im = ax.imshow(M, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
ax.set_xticks(range(len(COLS)), [c for c, _ in COLS], fontsize=9)
ax.set_yticks(range(len(ROWS)), [r for r, _ in ROWS], fontsize=9)
ax.set_xlabel("carried item", fontsize=9)
ax.set_ylabel("terrain", fontsize=9)
for i in range(len(ROWS)):
    for j in range(len(COLS)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if M[i, j] < 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("slip probability", fontsize=8)
cb.ax.tick_params(labelsize=8)
ax.set_title(f"per-step slip probability  (SLIP={sk.SLIP_PROB_DEFAULT:g})",
             fontsize=9.5)
fig.tight_layout()
p = FIGDIR / "slip_table.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {p}\n{M}")
