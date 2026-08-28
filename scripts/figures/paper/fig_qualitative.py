#!/usr/bin/env python3
"""Qualitative figure: diverse stochastic routes on one map per biome.

Rows are biomes (lakes, rocky, balanced), columns are the three agents. Each
panel overlays 20 sampled rollouts on the map, coloured by route: crossing the
obstacle with a tool (through) against detouring around it (around). These are
the maps used for the steering study, chosen because all three agents split
between the two routes here.

  PYTHONPATH=src python scripts/figures/paper/fig_qualitative.py
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from cogniland.bridge_tunnel import tiles as T   # noqa: E402
OUT = REPO / "paper/figures/forkwall_paper"
SM = REPO / "scripts/mechinterp/steering_maps"
pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))

AGENTS = [("ppo", "PPO + GRU"), ("dreamer", "DreamerV3"), ("storm", "STORM")]
BIOMES = ["lakes", "rocky", "balanced"]
# One colour per rollout, evenly spaced through a perceptually uniform map, so
# individual episodes stay separable where many paths overlap.
CMAP = matplotlib.colormaps["turbo"]
data = {a: json.loads((SM / f"rollouts_{a}.json").read_text()) for a, _ in AGENTS}

rc = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 8.5}
with plt.rc_context(rc):
    fig, axes = plt.subplots(len(BIOMES), len(AGENTS), figsize=(12.6, 6.4))
    for ri, biome in enumerate(BIOMES):
        for ci, (ag, lab) in enumerate(AGENTS):
            ax = axes[ri, ci]
            d = data[ag][biome]; rec = pool[d["map_id"]]
            H, W = rec.terrain.shape
            ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
            ax.set_xlim(-.5, W-.5); ax.set_ylim(H-.5, -.5)
            nt = na = 0
            n_roll = len(d["rollouts"])
            for j, roll in enumerate(d["rollouts"]):
                p = np.array(roll["path"], float)
                col = CMAP(0.06 + 0.88 * j / max(n_roll - 1, 1))
                nt += roll["route"] == "through"; na += roll["route"] == "around"
                ax.plot(p[:, 1], p[:, 0], color=col, lw=1.0, alpha=.75, zorder=5)
            # doors
            for cells, name in ((rec.top_goal_cells, "top"), (rec.bottom_goal_cells, "bottom")):
                good = rec.correct_target in ("either", name)
                for (r, c) in cells:
                    ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, fill=False,
                                 edgecolor="#22c55e" if good else "#ef4444", lw=1.6, zorder=7))
            ax.plot(rec.spawn[1], rec.spawn[0], "o", color="white", mec="black", ms=5, zorder=8)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(lab, fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{biome}\nmap {d['map_id']}", fontsize=9)
    fig.suptitle("Twenty sampled rollouts per panel, one colour per episode, "
                 "on the same three held-out maps", y=1.0, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .98])
    fig.savefig(OUT / "fig_res_routes.png", bbox_inches="tight")
    print("wrote fig_res_routes.png")
