#!/usr/bin/env python3
"""Fit the WITHIN-MAP route-intent axis from stochastic rollouts.

Input: route_rollouts_<cat>.npz from wm_collect_route.py -- several sampled
rollouts per train map, early-window mean state each, labelled by tool use.
Construction: through = n_tool >= 3, around = n_tool == 0 (1-2 dropped as
ambiguous); only maps holding BOTH classes contribute; the axis is the unit
mean of within-map class-mean differences, so map identity cancels and what
remains is the stochastic intent difference.

Because the axis is built from within-map differences, absolute coordinate
targets do not transfer across maps. Steering therefore uses the DISPLACEMENT
form: +/- lambda * gap along v, where gap is the pooled within-map-centred
class separation. The npz records v, gap, spreads, and a held-out (by map)
validation AUC so the axis's cross-map generalisation is measured, not hoped.

  conda activate crusoe
  python scripts/mechinterp/behavior_steering/wm_fit_route_withinmap.py --agent dreamer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/behavior_steering"


def auc(s, y):
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def fit(agent, cat, rng):
    z = np.load(OUT / agent / f"route_rollouts_{cat}.npz")
    S = z["states"].astype(np.float32)
    mid = z["map_id"]; n_tool = z["n_tool"]
    lab = np.where(n_tool >= 3, 1, np.where(n_tool == 0, 0, -1))
    keep = lab >= 0
    S, mid, lab = S[keep], mid[keep], lab[keep].astype(bool)

    maps = [m for m in np.unique(mid)
            if lab[mid == m].any() and (~lab[mid == m]).any()]
    if len(maps) < 8:
        return None, {"error": f"only {len(maps)} split maps"}
    rng.shuffle(maps)
    n_val = max(3, len(maps) // 4)
    fit_maps, val_maps = maps[n_val:], maps[:n_val]

    diffs = []
    for m in fit_maps:
        sm, lm = S[mid == m], lab[mid == m]
        diffs.append(sm[lm].mean(0) - sm[~lm].mean(0))
    v = np.mean(diffs, 0)
    v = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

    def centred_projs(mset):
        ps, ys = [], []
        for m in mset:
            sm, lm = S[mid == m], lab[mid == m]
            p = sm @ v
            ps.append(p - p.mean()); ys.append(lm)
        return np.concatenate(ps), np.concatenate(ys)

    p_fit, y_fit = centred_projs(fit_maps)
    p_val, y_val = centred_projs(val_maps)
    gap = float(p_fit[y_fit].mean() - p_fit[~y_fit].mean())
    meta = dict(n_rollouts=int(len(lab)), n_split_maps=len(maps),
                n_fit_maps=len(fit_maps), n_val_maps=len(val_maps),
                gap=gap, sd=float(p_fit.std()),
                auc_fit=auc(p_fit, y_fit), auc_val=auc(p_val, y_val),
                n_through=int(lab.sum()), n_around=int((~lab).sum()))
    return v, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["dreamer", "storm"])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out, meta = {}, {}
    for cat in ("rocky", "lakes"):
        f = OUT / a.agent / f"route_rollouts_{cat}.npz"
        if not f.exists():
            meta[cat] = "no rollouts collected"
            continue
        v, m = fit(a.agent, cat, rng)
        meta[cat] = m
        if v is not None:
            out[f"v_{cat}"] = v
            zb = np.load(REPO / f"outputs/belief_report/steer_axis_{a.agent}.npz")
            m["cos_belief"] = float(v @ zb["v"].astype(np.float32))
            if "v_wall" in zb.files:
                m["cos_belief_wall"] = float(v @ zb["v_wall"].astype(np.float32))
    np.savez(OUT / a.agent / "route_axes_withinmap.npz", **out)
    (OUT / a.agent / "route_axes_withinmap.json").write_text(
        json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
