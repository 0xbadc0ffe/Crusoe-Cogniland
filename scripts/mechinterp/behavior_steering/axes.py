#!/usr/bin/env python3
"""Contrast-mean behaviour axes for PPO -> behavior_axes.npz (+ metadata JSON).

Axes (all fit on TRAIN maps only, difference of activation means at single
actions — the convention preferred over segment labels):

  v_mine     mean(h | a=MINE)  − mean(h | a≠MINE,  rock_now>0)
  v_build    mean(h | a=BUILD) − mean(h | a≠BUILD, water_now>0)
  v_straight mean(h | a=RIGHT) − mean(h | a∈{UP,DOWN})   (evidence+corridor)
  v_route    episode-level: mean evidence-phase h of THROUGH episodes
             (#BUILD+#MINE>0) − AROUND episodes (=0), contrasted WITHIN each
             category and then averaged, so the axis is not a belief axis in
             disguise.

For every axis: raw unit vector, cos to the belief axis v_b (the leak
predictor), and the belief-orthogonalised variant v⊥ (renormalised).

  PYTHONPATH=src:scripts/mechinterp:scripts/mechinterp/belief_report \
      python scripts/mechinterp/behavior_steering/axes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/mechinterp/belief_report"))
from data import load, split_maps  # noqa: E402

OUT = REPO / "outputs/behavior_steering"
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_BUILD, A_MINE = range(6)


def unit(v):
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v), n


def main():
    X, df = load("ppo")
    tr, te = split_maps(df)
    on_train = df["map_id"].isin(tr).to_numpy()
    act = df["action"].to_numpy()
    rock_now = df["rock_now"].to_numpy()
    water_now = df["water_now"].to_numpy()
    phase = df["phase"].to_numpy()
    cats = df["category"].to_numpy()

    def mh(mask):
        rows = np.flatnonzero(mask & on_train)
        return np.asarray(X[rows], np.float32).mean(0), len(rows)

    axes, meta = {}, {}

    # v_mine / v_build: context-matched action contrasts
    for name, a_pos, ctx in (("mine", A_MINE, rock_now > 0),
                             ("build", A_BUILD, water_now > 0)):
        pos, n_pos = mh((act == a_pos) & ctx)
        neg, n_neg = mh((act != a_pos) & ctx)
        v, norm = unit(pos - neg)
        axes[f"v_{name}"] = v
        meta[f"v_{name}"] = dict(n_pos=n_pos, n_neg=n_neg, norm=round(norm, 3))

    # v_straight: RIGHT vs vertical, before the wall only
    pre = np.isin(phase, ["evidence", "corridor"])
    pos, n_pos = mh((act == A_RIGHT) & pre)
    neg, n_neg = mh(np.isin(act, [A_UP, A_DOWN]) & pre)
    v, norm = unit(pos - neg)
    axes["v_straight"] = v
    meta["v_straight"] = dict(n_pos=n_pos, n_neg=n_neg, norm=round(norm, 3))

    # v_route: through-vs-around, per category then averaged
    tools = df.assign(is_tool=np.isin(act, [A_BUILD, A_MINE]))
    ep_tool = tools.groupby("map_id")["is_tool"].sum()
    through_ids = set(ep_tool[ep_tool > 0].index) & set(tr)
    around_ids = set(ep_tool[ep_tool == 0].index) & set(tr)
    ev = phase == "evidence"
    mids = df["map_id"].to_numpy()
    contrasts, ns = [], {}
    for c in ("balanced", "lakes", "rocky"):
        thr_rows = np.flatnonzero(ev & (cats == c) & np.isin(mids, list(through_ids)))
        ard_rows = np.flatnonzero(ev & (cats == c) & np.isin(mids, list(around_ids)))
        ns[c] = dict(through_rows=len(thr_rows), around_rows=len(ard_rows),
                     through_eps=int(len({m for m in mids[thr_rows]})),
                     around_eps=int(len({m for m in mids[ard_rows]})))
        if len(thr_rows) < 50 or len(ard_rows) < 50:
            continue
        contrasts.append(np.asarray(X[thr_rows], np.float32).mean(0)
                         - np.asarray(X[ard_rows], np.float32).mean(0))
    v, norm = unit(np.mean(contrasts, axis=0))
    axes["v_route"] = v
    meta["v_route"] = dict(norm=round(norm, 3), per_category=ns,
                           n_category_contrasts=len(contrasts))

    # belief axis + orthogonalised variants
    z = np.load(REPO / "outputs/belief_report/steer_axis_ppo.npz")
    v_b = z["v"].astype(np.float32)
    axes["v_belief"] = v_b
    kit = {}
    for name, v in list(axes.items()):
        if name == "v_belief":
            continue
        cosb = float(v @ v_b)
        vperp, _ = unit(v - cosb * v_b)
        kit[name] = v
        kit[name + "_perp"] = vperp
        meta[name].update(cos_to_belief=round(cosb, 3),
                          cos_perp_to_belief=round(float(vperp @ v_b), 6))
    kit["v_belief"] = v_b
    # projection scale per axis (train-row SD of h.v), the natural dose unit
    sub = np.flatnonzero(on_train)[::20]
    H = np.asarray(X[sub], np.float32)
    for name in list(meta):
        meta[name]["proj_sd_train"] = round(float((H @ kit[name]).std()), 3)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "behavior_axes.npz", **kit)
    (OUT / "behavior_axes_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print("wrote", OUT / "behavior_axes.npz")


if __name__ == "__main__":
    main()
