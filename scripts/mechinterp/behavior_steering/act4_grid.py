#!/usr/bin/env python3
"""One balanced map, every agent-method, every condition: the whole campaign
as a single grid of trajectory bundles.

Rows are the four steering arms (which are also the three intervention
surfaces: state, actuator, plan), columns are suppress none / mine / build /
both. Every cell is the same map with the same rollout seeds, so a column
comparison reads as "what the command did" and a row comparison reads as "what
the surface costs".

  PYTHONPATH=src python scripts/mechinterp/behavior_steering/act4_grid.py --map 1191
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

ACT4 = REPO / "outputs/behavior_steering/act4"
FIG = REPO / "paper/figures/behavior_steering"
CMAP = matplotlib.colormaps["turbo"]
RC = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9}
CONDS = ("baseline", "sup_mine", "sup_build", "sup_both")
CLBL = {"baseline": "suppress none (baseline)", "sup_mine": "suppress mine",
        "sup_build": "suppress build", "sup_both": "suppress both"}
ARMS = [("ppo_clamp_noorth", "PPO\nGradientClamp\n(edits the state)"),
        ("storm_logit", "STORM\nlogit bias\n(edits the scores)"),
        ("dreamer_logit", "DreamerV3\nlogit bias\n(edits the scores)"),
        ("dreamer_tilt", "DreamerV3\nimagination tilt\n(edits the plan)")]


def cell(ax, rec, rolls):
    H, W = rec.terrain.shape
    ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
    ax.set_xlim(-.5, W - .5); ax.set_ylim(H - .5, -.5)
    n = len(rolls)
    mines = builds = ok = to = 0
    top = 0
    for j, roll in enumerate(rolls):
        col = CMAP(0.06 + 0.88 * j / max(n - 1, 1))
        P = np.array([[s["r"], s["c"]] for s in roll["steps"]], float)
        ax.plot(P[:, 1], P[:, 0], color=col, lw=.9, alpha=.8, zorder=5)
        for s in roll["steps"]:
            ev = s.get("ev")
            if ev:
                ax.plot([ev["c"]], [ev["r"]],
                        "x" if ev["kind"] == "mine" else "s", ms=3.0, mew=1.1,
                        color="#111827", alpha=.9, zorder=6)
                mines += ev["kind"] == "mine"
                builds += ev["kind"] != "mine"
        ok += bool(roll["correct"]); to += bool(roll.get("to"))
        top += roll.get("door") == "top"
    for cells, name in ((rec.top_goal_cells, "top"),
                        (rec.bottom_goal_cells, "bottom")):
        good = rec.correct_target in ("either", name)
        for (r, c) in cells:
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False,
                         edgecolor="#22c55e" if good else "#ef4444", lw=1.2,
                         zorder=7))
    ax.plot(rec.spawn[1], rec.spawn[0], "o", color="white", mec="black", ms=4,
            zorder=8)
    ax.set_xticks([]); ax.set_yticks([])
    return dict(n=n, ok=ok, to=to, mines=mines, builds=builds, top=top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=int, default=1191)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    rec = pool[a.map]

    have = [(arm, lab) for arm, lab in ARMS
            if all((ACT4 / f"trace_{arm}_{c}_{a.map}.json").exists() for c in CONDS)]
    missing = [arm for arm, _ in ARMS if (arm, _) not in have]
    if missing:
        print("missing traces for:", missing)
    if not have:
        raise SystemExit("no arm has a complete set of traces for this map")

    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(have), len(CONDS),
                                 figsize=(4.1 * len(CONDS), 1.9 * len(have)),
                                 squeeze=False)
        for i, (arm, lab) in enumerate(have):
            base_top = None
            for j, cond in enumerate(CONDS):
                d = json.loads(
                    (ACT4 / f"trace_{arm}_{cond}_{a.map}.json").read_text())
                rolls = list(d.values())[0]["rollouts"]
                st = cell(axes[i][j], rec, rolls)
                if cond == "baseline":
                    base_top = st["top"]
                door = f"{st['top']}/{st['n']} top"
                if cond != "baseline" and base_top is not None:
                    door += f"  (was {base_top}/{st['n']})"
                axes[i][j].set_title(
                    f"{st['ok']}/{st['n']} ok · {st['mines']} mines · "
                    f"{st['builds']} builds · {door}",
                    fontsize=7.0, loc="left", pad=2.5,
                    color="#111827" if cond == "baseline" else "#374151")
                if i == 0:
                    axes[i][j].text(.5, 1.34, CLBL[cond], fontsize=9.6,
                                    ha="center", va="bottom", weight="bold",
                                    transform=axes[i][j].transAxes)
            axes[i][0].set_ylabel(lab, fontsize=8.2)
        fig.suptitle(
            f"Balanced map {a.map}, held out — every agent, every command, same "
            f"seeds.  'x' = mine, '□' = build; both doors are rewarded here, so\n"
            "success cannot see the decision: watch the top/bottom split "
            "instead.", y=1.005, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        out = a.out or f"act4_grid_{a.map}.png"
        fig.savefig(FIG / out, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
