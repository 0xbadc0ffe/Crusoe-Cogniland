#!/usr/bin/env python3
"""Belief causality by vector ADD: h <- h + alpha * v, one write at corridor entry.

Differences from steer_belief.py, which used a coordinate SET:

* The edit is an unconditional add along the corr2 axis, so the dose does not
  depend on where the state already sits. alpha is measured in class-mean gaps,
  so alpha=1 moves the projection by exactly mean(rocky) - mean(lakes). That
  keeps PPO (gap 11.16) and Dreamer (gap 4.75) on one x axis.
* The push always runs AGAINST the map: rocky maps are pushed toward lakes,
  lakes maps toward rocky. Balanced maps have no correct door, so they are
  pushed both ways and drawn as two curves.
* alpha = 0 is a plain replay, so the leftmost point of every curve is the
  unsteered baseline.
* Transient only: a single write at the first corridor step.

The outcome is P(top door), which is an absolute quantity rather than a flip
rate: rocky maps are rewarded at the top door and lakes maps at the bottom, so
a working intervention drags every curve toward the door of the class it was
pushed towards. Episodes that time out count as "not top", and the timeout rate
is stored alongside so a curve that falls because the agent stopped moving can
be told apart from one that falls because it changed its mind.

  PYTHONPATH=src:scripts/mechinterp:scripts/mechinterp/belief_report \
      python steer_alpha.py --agent ppo --n 100 --workers 24
  (dreamer: conda r2dreamer, PYTHONPATH+=r2dreamer_model, --device cuda)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "mechinterp"))

OUT = REPO / "outputs/belief_report"
FIG = REPO / "paper/figures/belief_report"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
# (map category, direction pushed) -> curve. +1 is toward rocky, -1 toward lakes.
ARMS = [("lakes", +1), ("rocky", -1), ("balanced", +1), ("balanced", -1)]


def axis_npz(agent):
    return OUT / f"steer_axis_{agent}.npz"


def load_axis(agent, site):
    """Injection site, precomputed per map in inject_steps.json.

    corridor  first step on pure grass, col_rel_wall = -16 (corr1 entry)
    corr2     first step of the bin the axis is fitted in, col_rel_wall = -8
    wall      first step past the wall, col_rel_wall = 0
    """
    z = np.load(axis_npz(agent))
    steps = json.loads((OUT / "inject_steps.json").read_text())[agent]
    entry = {int(m): d[site] for m, d in steps.items()}
    return (z["v"].astype(np.float32), float(z["gap"]), entry,
            {c: z[c].tolist() for c in ("lakes", "rocky", "balanced")})


def add_hook(v, delta, t0):
    """One write at t0: h <- h + delta * v. delta carries the sign."""
    def hook(h, t, info):
        return h + delta * v if t == t0 else h
    return hook


def run_one(args):
    agent, mid, t0, delta, v, device = args
    from replay_episode import replay
    hook = None if delta == 0.0 else add_hook(v, delta, t0)
    r = replay(agent, mid, hook=hook, device=device)
    return dict(map_id=int(mid), door=r["door"], success=bool(r["success"]),
                steps=int(r["steps"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["ppo", "dreamer"])
    ap.add_argument("--n", type=int, default=100, help="maps per category")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--alphas", default=None,
                    help="comma-separated doses (default: the coarse ladder in ALPHAS)")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument("--site", default="corr2",
                    choices=["corridor", "corr2", "wall"],
                    help="where the single write lands")
    a = ap.parse_args()
    global ALPHAS
    if a.alphas:
        ALPHAS = [round(float(x), 4) for x in a.alphas.split(",")]

    v, gap, entry, percat = load_axis(a.agent, a.site)
    print(f"{a.agent}: corr2 axis, class-mean gap {gap:.3f} "
          f"(alpha=1 moves the projection by that much); "
          f"single write at the {a.site} step", flush=True)

    jobs = []
    for cat, sgn in ARMS:
        mids = [m for m in percat[cat][:a.n] if m in entry]
        for mid in mids:
            for al in ALPHAS:
                if al == 0.0 and (cat, sgn) == ("balanced", -1):
                    continue          # the sham is shared by the two balanced arms
                jobs.append((cat, sgn, al, int(mid), int(entry[mid])))
    print(f"{len(jobs)} episodes", flush=True)

    payload = [(a.agent, mid, t0, sgn * al * gap, v, a.device)
               for (_, sgn, al, mid, t0) in jobs]

    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(run_one, payload, chunksize=4))
    else:
        res = [run_one(p) for p in payload]

    rows = [dict(cat=c, sign=int(s), alpha=float(al), **r)
            for (c, s, al, _, _), r in zip(jobs, res)]
    # the shared sham also belongs to the downward balanced curve
    rows += [dict(r, sign=-1) for r in rows
             if r["cat"] == "balanced" and r["alpha"] == 0.0]

    path = OUT / f"steer_alpha_{a.agent}_{a.site}{('_' + a.tag) if a.tag else ''}.json"
    path.write_text(json.dumps(dict(agent=a.agent, site=a.site, gap=gap,
                                    alphas=ALPHAS, n=a.n, rows=rows), indent=1))
    print("wrote", path.name, flush=True)
    summarise(rows)


def summarise(rows):
    for cat, sgn in ARMS:
        sub = [r for r in rows if r["cat"] == cat and r["sign"] == sgn]
        if not sub:
            continue
        lab = f"{cat} -> {'rocky' if sgn > 0 else 'lakes'}"
        out = []
        for al in ALPHAS:
            g = [r for r in sub if r["alpha"] == al]
            if not g:
                continue
            top = np.mean([r["door"] == "top" for r in g])
            to = np.mean([r["door"] not in ("top", "bottom") for r in g])
            out.append(f"a={al:<4g} top={top:.2f} to={to:.2f}")
        print(f"  {lab:20s} " + "  ".join(out), flush=True)


if __name__ == "__main__":
    main()
