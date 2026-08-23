#!/usr/bin/env python3
"""Side-by-side trajectories of the three agents on identical held-out maps.

Reads the rollouts_{ppo,dreamer,storm}.json written by paper_rollouts.py.
Usage: PYTHONPATH=src python scripts/figures/paper_traj_fig.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from cogniland.bridge_tunnel.tiles import TILE_COLORS  # noqa: E402

AGENTS = ["ppo", "dreamer", "storm"]
LABEL = {"ppo": "PPO+GRU", "dreamer": "DreamerV3", "storm": "STORM"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=str(REPO / "paper/figures/forkwall_paper"))
    p.add_argument("--maps", default=str(REPO / "data/bridge_tunnel/forkwall6k/test.pkl"))
    p.add_argument("--map-ids", default="0,5,7")
    args = p.parse_args()
    d = Path(args.dir)
    ids = [int(x) for x in args.map_ids.split(",")]

    with open(args.maps, "rb") as f:
        pool = pickle.load(f)
    rows = {a: {r["map_id"]: r for r in json.loads((d / f"rollouts_{a}.json").read_text())}
            for a in AGENTS if (d / f"rollouts_{a}.json").exists()}

    rc = {"figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
          "axes.titlesize": 9}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(len(ids), len(rows), figsize=(4.1 * len(rows),
                                                               2.05 * len(ids)))
        axes = np.atleast_2d(axes)
        for r, mid in enumerate(ids):
            rec = pool[mid]
            for c, agent in enumerate([a for a in AGENTS if a in rows]):
                ax = axes[r, c]
                ax.imshow(TILE_COLORS[rec.terrain], interpolation="nearest")
                for cells, name in ((rec.top_goal_cells, "top"),
                                    (rec.bottom_goal_cells, "bottom")):
                    good = rec.correct_target in ("either", name)
                    for (rr, cc) in cells:
                        ax.add_patch(Rectangle((cc - .5, rr - .5), 1, 1, fill=False,
                                               edgecolor="#22c55e" if good else "#ef4444",
                                               lw=1.8, zorder=6))
                row = rows[agent].get(mid)
                if row:
                    t = np.asarray(row["traj"], dtype=float)
                    pts = t[:, [1, 0]].reshape(-1, 1, 2)
                    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    lc = LineCollection(segs, cmap="plasma",
                                        array=np.linspace(0, 1, len(segs)),
                                        linewidths=1.5, zorder=7)
                    ax.add_collection(lc)
                    ax.plot(t[0, 1], t[0, 0], "o", color="white", mec="black", ms=4,
                            zorder=8)
                    ok = "✓" if row["success"] else "✗"
                    ax.set_title(f"{LABEL[agent]} — {ok} {row['steps']} steps, "
                                 f"R={row['ret']:+.2f}", loc="left", fontsize=8.5)
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"map {mid}\n({rec.category})", fontsize=8)
        fig.suptitle("Same held-out maps, three agents — trajectory colour runs "
                     "dark→bright with time", y=1.005)
        fig.tight_layout()
        fig.savefig(d / "fig_trajectories.png", bbox_inches="tight")
        plt.close(fig)
    print("wrote", d / "fig_trajectories.png")


if __name__ == "__main__":
    main()
