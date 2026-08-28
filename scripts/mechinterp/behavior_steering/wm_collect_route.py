#!/usr/bin/env python3
"""Collect within-map route-contrast data for the world models.

The frozen dataset cannot support a through/around contrast (one episode per
map, tool use nearly universal), and the across-map intensity contrast turned
out causally inert in a Dreamer pilot -- consistent with it being a terrain-
density correlate rather than an intent. The controlled construction is
WITHIN-map: several stochastic rollouts of the same map, labelled by whether
they crossed with the tool, early-window states recorded, and the axis taken
as the average of within-map class-mean differences so map identity cancels.

Writes, per (agent, category): early-window mean state per rollout + label.

  # dreamer (conda r2dreamer)
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:r2dreamer_model:scripts/mechinterp:scripts/figures \
    python scripts/mechinterp/behavior_steering/wm_collect_route.py \
    --agent dreamer --cat rocky --maps 90 --seeds 6
  # storm (STORM_model/.venv, run from STORM_model/)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "mechinterp"))
sys.path.insert(0, str(REPO / "scripts" / "figures"))

from replay_episode import replay  # noqa: E402

OUT = REPO / "outputs/behavior_steering"
A_TOOL = {"rocky": 5, "lakes": 4}
FEAT_KEY = {"dreamer": "deter", "storm": "h"}
EARLY_MARGIN = 24        # early window: col < wall_col - 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["dreamer", "storm"])
    ap.add_argument("--cat", required=True, choices=["rocky", "lakes"])
    ap.add_argument("--maps", type=int, default=90)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    # train-split map ids of this category (test maps stay untouched)
    sys.path.insert(0, str(REPO / "scripts" / "mechinterp" / "belief_report"))
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    import csv
    cat_ids = []
    with open(REPO / f"activation_datasets/cogniland_belief/{a.agent}_episodes.csv") as fh:
        for row in csv.DictReader(fh):
            if row["category"] == a.cat:
                cat_ids.append(int(row["map_id"]))
    # reproduce the split without pandas: data.split_maps uses rng(0) over the
    # per-category id list in first-occurrence order, test_frac=1/3
    rng = np.random.default_rng(0)
    ids = rng.permutation(np.array(sorted(set(cat_ids))))
    n_te = int(round(len(ids) / 3))
    train_ids = sorted(int(i) for i in ids[n_te:])

    tool = A_TOOL[a.cat]
    key = FEAT_KEY[a.agent]
    rows = []
    for mi, mid in enumerate(train_ids[: a.maps]):
        wall = int(pool[mid].wall_col)
        for k in range(a.seeds):
            r = replay(a.agent, mid, seed=3000 + k, device=a.device)
            acts = r["actions"]
            n_tool = sum(x == tool for x in acts)
            first_tool = next((i for i, x in enumerate(acts) if x == tool), len(acts))
            cols = [p[1] for p in r["positions"][:-1]]      # col when acting
            early = [i for i in range(len(acts))
                     if cols[i] < wall - EARLY_MARGIN and i < first_tool]
            if len(early) < 3:
                continue
            F = np.stack([np.asarray(r["features"][i][key], np.float32)
                          for i in early]).mean(0)
            rows.append(dict(map_id=mid, seed=3000 + k, n_tool=n_tool,
                             success=bool(r["success"]), state=F))
        if (mi + 1) % 15 == 0:
            print(f"  {mi+1}/{min(a.maps, len(train_ids))} maps", flush=True)

    d = OUT / a.agent
    d.mkdir(parents=True, exist_ok=True)
    S = np.stack([r["state"] for r in rows])
    np.savez_compressed(d / f"route_rollouts_{a.cat}.npz",
                        states=S.astype(np.float16),
                        map_id=np.array([r["map_id"] for r in rows]),
                        seed=np.array([r["seed"] for r in rows]),
                        n_tool=np.array([r["n_tool"] for r in rows]),
                        success=np.array([r["success"] for r in rows]))
    n_thr = sum(r["n_tool"] >= 3 for r in rows)
    n_ard = sum(r["n_tool"] == 0 for r in rows)
    print(f"wrote {d}/route_rollouts_{a.cat}.npz  rollouts={len(rows)} "
          f"through(>=3)={n_thr} around(0)={n_ard}")


if __name__ == "__main__":
    main()
