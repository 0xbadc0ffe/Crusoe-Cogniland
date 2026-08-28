#!/usr/bin/env python3
"""Baseline-vs-steered trajectory panels on the thesis maps (626/77/99).

Same visual language as figure 7.5 (one colour per episode, evenly spaced from
`turbo`, terrain underneath, doors outlined), extended to one ROW PER CONDITION
so the consolidation is visible: baseline split on top, each steered condition
under it. Tool events are drawn as markers so "still mines the minimal amount"
is visible on constrained maps.

Input JSONs use the ghost-trace schema (collect_ghost_rollouts.py):
  {"<biome>": {"map_id": int, "rollouts": [{"steps": [{"r","c","facing","ev"}...],
                                            "correct": bool}, ...]}}

  python plot_steered_routes.py --agent ppo --map 77 \
      --conditions baseline=out/base_ppo.json suppress-mine=out/sup_ppo.json \
      --out fig_beh_ppo_77.png
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from cogniland.bridge_tunnel import tiles as T  # noqa: E402

CMAP = matplotlib.colormaps["turbo"]
BIOME_OF = {626: "lakes", 77: "rocky", 99: "balanced"}
RC = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9}


def draw_panel(ax, rec, rolls, label):
    H, W = rec.terrain.shape
    ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
    ax.set_xlim(-.5, W - .5); ax.set_ylim(H - .5, -.5)
    n = len(rolls)
    mines = builds = ok = 0
    for j, roll in enumerate(rolls):
        col = CMAP(0.06 + 0.88 * j / max(n - 1, 1))
        P = np.array([[s["r"], s["c"]] for s in roll["steps"]], float)
        ax.plot(P[:, 1], P[:, 0], color=col, lw=1.0, alpha=.75, zorder=5)
        for s in roll["steps"]:
            ev = s.get("ev")
            if ev:
                mk = "x" if ev["kind"] == "mine" else "s"
                ax.plot([ev["c"]], [ev["r"]], mk, ms=3.4, mew=1.2,
                        color="#111827", alpha=.85, zorder=6)
                if ev["kind"] == "mine":
                    mines += 1
                else:
                    builds += 1
        ok += bool(roll["correct"])
    # wrong-door / timeout counters, shown when the JSON carries the fields
    # (act2 traces do; older traces fall back to the success-only header)
    have_door = any("door" in r for r in rolls)
    wrong = sum(1 for r in rolls if not r["correct"]
                and r.get("door", "none") != "none" and not r.get("to"))
    tmo = sum(1 for r in rolls if r.get("to"))
    for cells, name in ((rec.top_goal_cells, "top"), (rec.bottom_goal_cells, "bottom")):
        good = rec.correct_target in ("either", name)
        for (r, c) in cells:
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False,
                         edgecolor="#22c55e" if good else "#ef4444",
                         lw=1.6, zorder=7))
    ax.plot(rec.spawn[1], rec.spawn[0], "o", color="white", mec="black",
            ms=5, zorder=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(label, fontsize=9)
    head = f"success {ok}/{n}    mines {mines}    builds {builds}"
    if have_door:
        head = (f"success {ok}/{n}    wrong door {wrong}    timeout {tmo}    "
                f"mines {mines}    builds {builds}")
    ax.set_title(head, loc="right", fontsize=7.6, color="#374151")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--map", type=int, required=True)
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="label=path.json, first one is the baseline row")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    rec = pool[a.map]
    biome = BIOME_OF.get(a.map, rec.category)

    conds = []
    for spec in a.conditions:
        label, path = spec.split("=", 1)
        data = json.loads(Path(path).read_text())
        entry = data[biome] if biome in data else data
        assert int(entry["map_id"]) == a.map, (entry["map_id"], a.map)
        conds.append((label, entry["rollouts"]))

    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(conds), 1,
                                 figsize=(10.4, 2.9 * len(conds)), squeeze=False)
        for ax, (label, rolls) in zip(axes[:, 0], conds):
            draw_panel(ax, rec, rolls, label)
        fig.suptitle(a.title or
                     f"{a.agent}: map {a.map} ({biome}) — baseline against "
                     f"steered behaviour ('x' = mine, '□' = build)",
                     y=1.0, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(a.out, bbox_inches="tight")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
