#!/usr/bin/env python3
"""Replay ONE chosen balanced map under act four's frozen knobs, for figures.

The test stage only archives traces for its first six maps; this reproduces the
same episodes (same conditions, same seeds, same frozen operating points) for
any map id, so a specific map -- e.g. the thesis map 99 -- can be shown.

Nothing is re-fitted: the knobs come from act4/operating_points.json.

  # PPO (conda crusoe)
  PYTHONPATH=src:scripts/mechinterp:scripts/mechinterp/belief_report:scripts/figures:scripts/mechinterp/behavior_steering \
    python scripts/mechinterp/behavior_steering/act4_trace_map.py --arm ppo_clamp --map 99 --rolls 20
  # Dreamer (conda r2dreamer) / STORM (STORM_model/.venv), same flags
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in ("src", "scripts/mechinterp", "scripts/mechinterp/belief_report",
          "scripts/figures", "scripts/mechinterp/behavior_steering"):
    sys.path.insert(0, str(REPO / p))

import act4_balanced as A4  # noqa: E402

OUT = REPO / "outputs/behavior_steering/act4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["ppo_clamp", "storm_logit", "dreamer_logit",
                             "dreamer_tilt"])
    ap.add_argument("--map", type=int, required=True)
    ap.add_argument("--rolls", type=int, default=20)
    ap.add_argument("--no-orth", action="store_true",
                    help="PPO only: run the clamp WITHOUT the project_out "
                         "correction, so traces match the un-repaired arm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-iters", type=int, default=25)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--K", type=int, default=12)
    a = ap.parse_args()

    ops = json.loads((OUT / "operating_points.json").read_text())[a.arm]["conds"]
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    rec = pool[a.map]
    assert rec.category == "balanced", f"map {a.map} is {rec.category}"
    rows = []

    if a.arm == "ppo_clamp":
        jobs = []
        for cond in A4.CONDS:
            knob = 0.0 if cond == "baseline" else ops[cond]["knob"]
            for k in range(a.rolls):
                jobs.append((a.map, A4.SEED0 + k, cond, knob,
                             a.alpha, a.max_iters, "test", not a.no_orth))
        rows = A4.run_ppo(jobs, a.workers)
    elif a.arm in ("storm_logit", "dreamer_logit"):
        agent = "storm" if a.arm == "storm_logit" else "dreamer"
        import replay_episode as RE
        act, reset = RE._get_agent(agent, a.device)
        kit = A4.wm_kit(agent)
        for cond in A4.CONDS:
            knob = 0.0 if cond == "baseline" else ops[cond]["knob"]
            for k in range(a.rolls):
                rows.append(A4.wm_logit_episode(act, reset, rec, a.map,
                                                A4.SEED0 + k, cond, knob,
                                                agent, kit))
    else:
        from act3_wm import DreamerImagination
        from replay_episode import CKPT
        D = DreamerImagination(CKPT["dreamer"]["ckpt"], a.device,
                               CKPT["dreamer"]["size"])
        for cond in A4.CONDS:
            lam = 0.0 if cond == "baseline" else ops[cond]["knob"]
            for k in range(a.rolls):
                rows.append(A4.dreamer_tilt_episode(D, rec, a.map,
                                                    A4.SEED0 + k, cond, lam,
                                                    a.M, a.K))

    for cond in A4.CONDS:
        A4.summarise([r for r in rows if r["cond"] == cond], f"map {a.map} {cond}")
    tr = {}
    for r in rows:
        tr.setdefault(r["cond"], []).append(
            dict(steps=r["trace"], correct=r["success"], door=r["door"],
                 to=r["timeout"]))
    for cond, v in tr.items():
        arm_tag = a.arm + ("_noorth" if getattr(a, "no_orth", False) else "")
        path = OUT / f"trace_{arm_tag}_{cond}_{a.map}.json"
        path.write_text(json.dumps({"balanced": dict(map_id=a.map, rollouts=v)}))
        print("wrote", path.name)


if __name__ == "__main__":
    main()
