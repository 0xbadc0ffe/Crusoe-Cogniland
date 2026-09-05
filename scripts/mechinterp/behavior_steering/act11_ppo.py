#!/usr/bin/env python3
"""PPO+GRU steering on held-out balanced maps: the same three arms as Dreamer.

Arms: unsteered, suppress bridge, suppress tunnel. Never both -- the comparison
of interest is each skill against the untouched agent.

The method is the gated gradient clamp on the GRU hidden state, at the operating
points frozen on TRAIN-pool maps in act 5 (outputs/.../act5/operating_points.json).
Unlike the Dreamer tilt this WRITES to the carried state, which is why the
belief readback matters: we report the projection of h on the corr2 belief axis
alongside the behaviour, so the cost of the intervention is visible next to what
it buys.

PPO replays are CPU-only, so this runs on cores rather than a GPU.

  python act11_ppo.py --maps 36 --rolls 6 --workers 32

Stronger clamp than the frozen operating points (the "adversarial" reading:
push P(suppressed tool) below theta with the minimal edit, every step):

  python act11_ppo.py --rolls 10 --theta 1e-4 --iters 80 --ungated \
      --reuse-maps outputs/behavior_steering/act11/summary.json --tag strong_ungated
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ("src", "scripts/mechinterp", "scripts/mechinterp/belief_report",
          "scripts/figures", "scripts/mechinterp/behavior_steering"):
    sys.path.insert(0, str(REPO / p))

OUT = REPO / "outputs/behavior_steering/act11"
FIG = REPO / "paper/figures/behavior_steering"
TEST_PKL = REPO / "data/bridge_tunnel/forkwall6k/test.pkl"
EPS = 1e-3
# (arm label, act5 condition key, colour). None = the untouched agent.
ARMS = [("unsteered", None, "#6b7280"),
        ("suppress bridge", "sup_build", "#0e7490"),
        ("suppress tunnel", "sup_mine", "#b91c1c")]


def jobs_for(mid, arm_key, ops, rolls, seed0, theta=None, window=None,
             ungated=False, iters=25, alpha=0.5):
    """theta/window default to the act-5 operating points; `ungated` drops the
    stuck-release gate; iters/alpha are the clamp's step budget and step size."""
    if arm_key is None:
        thr, win, gated = 0.0, ops["sup_mine"]["window"], False
    else:
        thr = ops[arm_key]["theta"] if theta is None else float(theta)
        win = ops[arm_key]["window"] if window is None else int(window)
        gated = not ungated
    cond = arm_key or "baseline"
    return [(mid, seed0 + k, cond, thr, win, EPS, gated, "test", int(iters),
             float(alpha)) for k in range(rolls)]


def _num_word(n):
    return {6: "six", 8: "eight", 10: "ten", 12: "twelve"}.get(n, str(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=36)
    ap.add_argument("--rolls", type=int, default=6)
    ap.add_argument("--screen", type=int, default=130,
                    help="how many maps of the category to screen (400 = the whole test pool)")
    ap.add_argument("--category", default="balanced", choices=["balanced", "lakes", "rocky"])
    ap.add_argument("--all-eligible", action="store_true",
                    help="keep every map that passes the screen instead of sampling --maps of them")
    ap.add_argument("--grid", action="store_true", help="draw the route grid even for large map sets")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=4000)
    ap.add_argument("--figs-only", action="store_true")
    # clamp strength (None = the act-5 frozen operating points)
    ap.add_argument("--theta", type=float, default=None,
                    help="suppress P(tool) below this at every clamp-active step")
    ap.add_argument("--window", type=int, default=None,
                    help="stuck-release window (env steps); ignored with --ungated")
    ap.add_argument("--ungated", action="store_true",
                    help="never release the clamp, even when stuck")
    ap.add_argument("--iters", type=int, default=25, help="clamp max_iters")
    ap.add_argument("--alpha", type=float, default=0.5, help="clamp step size")
    ap.add_argument("--reuse-maps", default=None,
                    help="summary.json whose 'maps' list fixes the map set")
    ap.add_argument("--tag", default="", help="suffix for rows/summary/figure")
    ap.add_argument("--markers", action="store_true",
                    help="draw the tool-event glyphs (off by default)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sfx = f"_{a.tag}" if a.tag else ""
    knobs = dict(theta=a.theta, window=a.window, ungated=a.ungated,
                 iters=a.iters, alpha=a.alpha)

    import act5_ppo as A5
    from grid_fig import draw_grid
    pool = pickle.load(open(TEST_PKL, "rb"))
    ops = json.loads((REPO / "outputs/behavior_steering/act5"
                      / "operating_points.json").read_text())

    if a.figs_only:
        rows = json.loads((OUT / f"rows{sfx}.json").read_text())
        mids = sorted({r["map_id"] for r in rows})
    else:
        if a.reuse_maps:
            # same map set as an earlier run, so the grids compare panel for panel
            prev = json.loads(Path(a.reuse_maps).read_text())
            mids = [int(m) for m in prev["maps"]][:a.maps]
            print(f"reusing {len(mids)} maps from {a.reuse_maps}", flush=True)
        else:
            # screen: balanced maps whose UNSTEERED agent uses both tools, so
            # both suppressions have something to remove
            cands = [i for i, r in enumerate(pool) if r.category == a.category]
            rng = np.random.default_rng(0)
            cands = [int(cands[i]) for i in rng.permutation(len(cands))[:a.screen]]
            sc_jobs = [j for mid in cands
                       for j in jobs_for(mid, None, ops, 2, a.seed0)]
            sc = A5.run(sc_jobs, a.workers)
            score = {}
            for mid in cands:
                rs = [r for r in sc if r["map_id"] == mid]
                score[mid] = (min(np.mean([r["mines"] for r in rs]),
                                  np.mean([r["builds"] for r in rs])),
                              np.mean([r["mines"] + r["builds"] for r in rs]))
            users = [m for m in sorted(score, key=lambda m: -score[m][0])
                     if score[m][0] > 0]
            if a.all_eligible:
                mids = sorted(int(m) for m in users)
            else:
                mids = sorted(int(users[i]) for i in
                              np.random.default_rng(0).permutation(len(users))[:a.maps])
            print(f"screened {len(cands)} {a.category} test maps: {len(users)} use "
                  f"both tools, kept {len(mids)}", flush=True)

        rows = []
        for tag, key, _ in ARMS:
            jb = [j for mid in mids
                  for j in jobs_for(mid, key, ops, a.rolls, a.seed0, **knobs)]
            got = A5.run(jb, a.workers)
            for r in got:
                r["arm"] = tag
            rows += got
            print(f"  {tag:16s} {len(got)} episodes", flush=True)
        (OUT / f"rows{sfx}.json").write_text(json.dumps(rows))

    # ---- table ----------------------------------------------------------
    base = [r for r in rows if r["arm"] == "unsteered"]
    b_proj = float(np.nanmean([r["proj"] for r in base]))
    print(f"\n{'arm':16s} {'tunnel':>7s} {'bridge':>7s} {'succ':>6s} {'TO':>6s} {'wrong':>6s} "
          f"{'P(top)':>7s} {'belief':>8s} {'dbelief':>8s} {'maxP(tool)':>11s}"
          f"   median % of target")
    summary = {}
    for tag, key, _ in ARMS:
        rs = [r for r in rows if r["arm"] == tag]
        pj = float(np.nanmean([r["proj"] for r in rs]))
        which = "mines" if "tunnel" in tag else "builds" if "bridge" in tag else None
        med, fell, neff = float("nan"), 0, 0
        if which:
            v = []
            for mid in mids:
                b = [r for r in base if r["map_id"] == mid]
                s = [r for r in rs if r["map_id"] == mid]
                vb = np.mean([r[which] for r in b])
                if vb > 0:
                    v.append(100 * (np.mean([r[which] for r in s]) - vb) / vb)
            if v:
                med, fell, neff = float(np.median(v)), int(sum(1 for x in v if x < 0)), len(v)
        ptm = [r.get("p_tool_max", float("nan")) for r in rs]
        summary[tag] = dict(n=len(rs),
                            p_tool_max_mean=float(np.nanmean(ptm)),
                            p_tool_max_max=float(np.nanmax(ptm)),
                            mines=float(np.mean([r["mines"] for r in rs])),
                            builds=float(np.mean([r["builds"] for r in rs])),
                            success=float(np.mean([r["success"] for r in rs])),
                            timeout=float(np.mean([r["timeout"] for r in rs])),
                            wrong=float(np.mean([r.get("wrong", False) for r in rs])),
                            p_top=float(np.mean([r["door"] == "top" for r in rs])),
                            belief=pj, dbelief=pj - b_proj,
                            median_pct=med, fell=fell, n_eff=neff)
        s = summary[tag]
        print(f"{tag:16s} {s['mines']:7.2f} {s['builds']:7.2f} {s['success']:6.2f} "
              f"{s['timeout']:6.2f} {s['wrong']:6.2f} {s['p_top']:7.2f} {s['belief']:8.2f} "
              f"{s['dbelief']:+8.2f} {s['p_tool_max_max']:11.2e}   "
              + (f"{med:+.1f}% ({fell}/{neff})" if which else ""))
    (OUT / f"summary{sfx}.json").write_text(json.dumps(
        dict(n_maps=len(mids), rolls=a.rolls, maps=mids, ops=ops, knobs=knobs,
             category=a.category, screen=a.screen, all_eligible=a.all_eligible,
             summary=summary), indent=1))
    if not (a.grid or len(mids) <= 60):
        print(f"{len(mids)} maps: route grid skipped (pass --grid to force)"); return

    if a.theta is None:
        clamp_txt = "Gated gradient clamp at the act-5 operating points."
    else:
        clamp_txt = (f"Gradient clamp on the GRU state: P(suppressed tool) < "
                     f"{a.theta:g} at every step, "
                     + ("never released." if a.ungated else
                        f"released after {a.window if a.window is not None else 'op-point'} stuck steps."))
    draw_grid(pool, mids, rows, [(t, c) for t, _, c in ARMS],
              FIG / f"act11_ppo_grid{sfx}.png",
              f"PPO+GRU on held-out {a.category} maps, {_num_word(a.rolls)} stochastic "
              "rollouts each, identical seeds across the three arms.  "
              + clamp_txt + "  Both exits pay, so the route is a free choice.  "
              "Panel label: share of rollouts leaving by the top door"
              + (", X = tunnelled block, square = placed bridge." if a.markers else "."),
              markers=a.markers, door_pct=True)


if __name__ == "__main__":
    main()
