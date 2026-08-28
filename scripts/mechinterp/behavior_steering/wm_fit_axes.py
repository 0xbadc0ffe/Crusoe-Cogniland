#!/usr/bin/env python3
"""Fit behaviour axes for the world models (DreamerV3, STORM) -> axes.npz.

Two families, per agent, fitted on TRAIN maps only (belief_report split, seed 0):

* TOOL axes (context-matched, Fork D's recipe):
    v_mine  = mean(state | action==MINE)  - mean(state | action!=MINE  and rock_now>0)
    v_build = mean(state | action==BUILD) - mean(state | action!=BUILD and water_now>0)
  The matched context (tile in view) keeps the axis from being a tile detector.

* ROUTE-INTENT axes (the lead hypothesis): on ROCKY maps an episode is
  `through` if it ever mines and `around` otherwise; per-episode mean state over
  the EARLY window (col_rel_wall < -24 AND strictly before the episode's first
  tool step), dm axis mu_through - mu_around with the two class-mean projection
  coordinates as steering targets. Same on LAKES maps with BUILD.

Everything an intervention needs is stored: unit axes, coordinate targets,
projection spreads (for dose/magnitude matching), sample sizes, and each axis's
cosine to the agent's belief axis (safety metric).

  conda activate crusoe
  PYTHONPATH=src:scripts/mechinterp/belief_report python \
      scripts/mechinterp/behavior_steering/wm_fit_axes.py --agent dreamer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "mechinterp" / "belief_report"))

import data as D  # noqa: E402

OUT = REPO / "outputs/behavior_steering"
A_BUILD, A_MINE = 4, 5
EARLY_CRW = -24          # route window: col_rel_wall strictly below this


def unit(v):
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v).astype(np.float32), n


def fit_route_axis(X, df, train_ids, cat, tool_action):
    """Tool-use INTENSITY axis: bottom vs top quartile of per-episode tool use.

    The frozen dataset holds one episode per map and the agents essentially
    always use the tool at least once on decisive maps (dreamer rocky min=1,
    median=9), so a binary through/around label degenerates. The graded
    version keeps the same intent semantics: states from the EARLY window
    (before the first tool step) of low-use episodes against high-use ones."""
    sub = df[df.category == cat]
    tool = sub[sub.action == tool_action].groupby("map_id")["t"].min()
    cnt = sub.groupby("map_id").apply(
        lambda g: int((g.action == tool_action).sum()), include_groups=False)
    mids = [m for m in cnt.index if m in train_ids]
    q25, q75 = np.percentile([cnt.loc[m] for m in mids], [25, 75])
    ep_states, ep_label = [], []
    for mid in mids:
        c = cnt.loc[mid]
        if q25 < c < q75:
            continue
        g = sub[sub.map_id == mid]
        first_tool = int(tool.loc[mid]) if mid in tool.index else 10**9
        rows = g[(g.col_rel_wall < EARLY_CRW) & (g.t < first_tool)]
        if len(rows) < 3:
            continue
        ep_states.append(np.asarray(X[rows.index.to_numpy()], np.float32).mean(0))
        ep_label.append(c >= q75)
    S = np.stack(ep_states)
    y = np.array(ep_label)
    if y.sum() < 20 or (~y).sum() < 20:
        return None
    v, gap = unit(S[y].mean(0) - S[~y].mean(0))
    p = S @ v
    return dict(v=v, gap=gap, q25=float(q25), q75=float(q75),
                c_through=float(p[y].mean()), c_around=float(p[~y].mean()),
                sd_through=float(p[y].std()), sd_around=float(p[~y].std()),
                n_through=int(y.sum()), n_around=int((~y).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["dreamer", "storm"])
    a = ap.parse_args()

    X, df = D.load(a.agent)
    tr, te = D.split_maps(df)
    train_ids = set(int(i) for i in tr)
    df = df.reset_index(drop=True)

    tr_rows = df.map_id.isin(train_ids).to_numpy()

    out, meta = {}, {"agent": a.agent, "dim": int(X.shape[1])}

    # ── tool axes (row-level, context-matched, train rows only) ──────────
    act = df.action.to_numpy()
    rock = df.rock_now.to_numpy() > 0
    water = df.water_now.to_numpy() > 0
    for name, a_id, ctx in (("mine", A_MINE, rock), ("build", A_BUILD, water)):
        pos = tr_rows & (act == a_id)
        neg = tr_rows & (act != a_id) & ctx
        v, gap = unit(np.asarray(X[np.flatnonzero(pos)], np.float32).mean(0)
                      - np.asarray(X[np.flatnonzero(neg)], np.float32).mean(0))
        p_all = np.asarray(X[np.flatnonzero(tr_rows)][::7], np.float32) @ v
        out[f"v_{name}"] = v
        meta[f"v_{name}"] = dict(gap=gap, n_pos=int(pos.sum()), n_neg=int(neg.sum()),
                                 proj_sd=float(p_all.std()))

    # ── route-intent axes ────────────────────────────────────────────────
    for cat, a_id, key in (("rocky", A_MINE, "route_rocky"),
                           ("lakes", A_BUILD, "route_lakes")):
        r = fit_route_axis(X, df, train_ids, cat, a_id)
        if r is None:
            meta[key] = "STARVED (fewer than 20 episodes in a class)"
            continue
        out[f"v_{key}"] = r.pop("v")
        meta[key] = r

    # ── safety: cosine to the belief axis ────────────────────────────────
    z = np.load(REPO / f"outputs/belief_report/steer_axis_{a.agent}.npz")
    vb = z["v"].astype(np.float32)
    cos = {k: float(out[k] @ vb) for k in out}
    if "v_wall" in z.files:
        vw = z["v_wall"].astype(np.float32)
        cos.update({f"{k}_vs_wall": float(out[k] @ vw) for k in out})
    meta["cos_to_belief"] = cos

    d = OUT / a.agent
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "axes.npz", **out)
    (d / "axes_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
