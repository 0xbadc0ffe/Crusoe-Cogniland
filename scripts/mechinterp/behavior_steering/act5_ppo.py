#!/usr/bin/env python3
"""Act five -- the gated clamp: can PPO's steering be made to actually work?

Act four's ceiling on the model-free agent is set by TIMEOUTS, not by the
suppression itself: tightening the clamp cuts tool use hard but strands the
agent at the obstacle it may no longer clear (theta 0.1 -> 17% timeouts,
0.05 -> 33%, 0.02 -> 100% on the fit set).

The repo's steering library already ships the escape hatch and we never used
it. `StuckDetector` is a per-sample gate that fires when a trajectory stops
making progress, and `GradientClamp` accepts it as `logic=`. Wrapping it in
`Not(...)` gives:

    clamp ACTIVE while the agent is progressing
    clamp RELEASED once it has been stuck for `window` env steps

so the agent is free to spend exactly as much of the forbidden tool as it
takes to get moving again, whereupon the clamp re-arms by itself. The
prediction is "minimal feasible tooling at full success" -- a large tool cut
WITHOUT the timeouts, which is the result act one wanted and never got.

Progress is the env's own cost-to-go to the rewarded door(s), normalised:

    progress = 1 - ctg(pos) / ctg(spawn)

which is terrain-aware (it already knows a lake costs more than grass) and
saturates at 1 past the wall column, where the potential is flat by
construction -- so the endgame is unconstrained. That is documented, not
incidental.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts/mechinterp:scripts/mechinterp/belief_report:scripts/figures:scripts/mechinterp/behavior_steering \
    python scripts/mechinterp/behavior_steering/act5_ppo.py --stage fit|freeze|test
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ("src", "scripts/mechinterp", "scripts/mechinterp/belief_report",
          "scripts/figures", "scripts/mechinterp/behavior_steering"):
    sys.path.insert(0, str(REPO / p))

import act4_balanced as A4  # noqa: E402

OUT = REPO / "outputs/behavior_steering/act5"
A_BUILD, A_MINE = 4, 5
SEED0 = A4.SEED0
CONDS = A4.CONDS
GUARD_SUCCESS, GUARD_TIMEOUT = 0.90, 0.05      # pre-registered, stricter than act4


def _gated_hook(cond, thr, alpha, iters, window, eps, gated, prog_cell):
    """GradientClamp on the actor head, optionally gated to release while the
    trajectory is stuck. `prog_cell` is a one-element list the episode loop
    writes progress into before each hook call."""
    import torch
    from cogniland.bridge_tunnel.steering import (
        ClampTerm, GradientClamp, Not, StuckDetector)
    tools = A4.TOOLS_OF[cond]
    if not tools:
        return None
    idx = {"mine": A_MINE, "build": A_BUILD}
    terms = [ClampTerm(head="actor", index=idx[t], mode="suppress",
                       threshold=thr) for t in tools]
    sd = StuckDetector(window=window, eps=eps) if gated else None
    logic = Not(sd) if gated else None
    clamp = GradientClamp(A4._PPO["policy"], terms, alpha=alpha,
                          max_iters=iters, logic=logic,
                          warn_on_nonconvergence=False)

    def hook(h, t, info):
        x = torch.from_numpy(np.asarray(h, np.float32)).reshape(1, 1, -1)
        ctx = {"progress": np.array([prog_cell[0]], dtype=np.float64)}
        out_t = clamp(x, t, ctx)
        out = out_t.reshape(-1).numpy().astype(np.float32)
        if sd is not None and sd._last_mask is not None and bool(sd._last_mask[0]):
            prog_cell[1] += 1                      # steps the clamp was RELEASED
        elif len(prog_cell) > 2:
            # diagnostic: the ACTUAL post-clamp P(tool) the actor samples from
            # (shallow edit point: the edited h feeds the heads directly). The
            # max over clamp-active steps says whether the threshold was met.
            with torch.no_grad():
                pr = torch.softmax(A4._PPO["policy"].actor(out_t.reshape(1, -1)), -1)[0]
            prog_cell[2] = max(prog_cell[2], float(max(pr[idx[t_]] for t_ in tools)))
        return out
    hook.detector = sd
    return hook


def episode(job):
    """One rollout. job = (map_id, seed, cond, thr, window, eps, gated, split)."""
    import torch
    # optional trailing (iters, alpha) let a caller strengthen the clamp
    # without changing the 8-tuple every act4/act5 job list already uses
    mid, seed, cond, thr, window, eps, gated, split = job[:8]
    iters = int(job[8]) if len(job) > 8 else 25
    alpha = float(job[9]) if len(job) > 9 else 0.5
    if "act" not in A4._PPO:
        A4._ppo_init()
    pool = A4._PPO.setdefault(
        f"pool_{split}",
        pickle.load(open(A4.TRAIN_PKL if split == "train" else A4.TEST_PKL, "rb")))
    rec = pool[mid]
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS
    from replay_episode import _get_agent
    act = A4._PPO["act"]
    np.random.seed(seed); torch.manual_seed(seed)
    prog = [0.0, 0, 0.0]     # [progress, #steps released, max post-clamp P(tool)]
    rel_pre = rel_post = 0   # releases before / after the wall column
    act.set_hook(_gated_hook(cond, thr, alpha, iters, window, eps, gated, prog))
    act.set_logit_bias(None)
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset()
    _get_agent("ppo", "cpu")[1]()
    ctg0 = max(env._ctg_at(getattr(env, "_commit", 0), env._pos), 1e-9)

    mines = builds = 0
    hs, cols, trace = [], [], []
    for t in range(FORKWALL_KWARGS["max_steps"]):
        ctg = env._ctg_at(getattr(env, "_commit", 0), env._pos)
        prog[0] = float(np.clip(1.0 - ctg / ctg0, 0.0, 1.0))
        _rel0 = prog[1]
        a = act(obs, False)
        if prog[1] > _rel0:                       # the gate released this step
            if int(env._pos[1]) < int(rec.wall_col):
                rel_pre += 1
            else:
                rel_post += 1
        hs.append(act.get_state().astype(np.float32))
        cols.append(int(env._pos[1]))
        obs, _, term, trunc, info = env.step(a)
        ev = None
        if a in (A_BUILD, A_MINE) and (info.get("placed") or info.get("mined")):
            dr, dc = A4.FACE_DELTA[int(info["facing"])]
            ev = dict(kind="build" if a == A_BUILD else "mine",
                      r=int(env._pos[0] + dr), c=int(env._pos[1] + dc))
            mines += a == A_MINE; builds += a == A_BUILD
        trace.append(dict(r=int(env._pos[0]), c=int(env._pos[1]),
                          facing=int(info["facing"]), ev=ev))
        if term or trunc:
            break
    act.set_hook(None)
    steps = len(trace)
    wall = int(rec.wall_col)
    H, C = np.array(hs), np.array(cols)
    crossed = np.where(C >= wall)[0]
    stop = int(crossed[0]) if len(crossed) else len(H)
    crw = C[:stop] - wall
    m = (crw >= -8) & (crw < 0)
    proj = H[:stop] @ A4._PPO["v_bel"] if stop else np.array([])
    rb = (float(proj[m].mean()) if m.any() else
          float(proj[np.argmax(C[:stop])]) if stop else float("nan"))
    to = steps >= 799
    ok = env._pos in (env._correct_cells or set())
    top = {p[0] for p in rec.top_goal_cells}
    bot = {p[0] for p in rec.bottom_goal_cells}
    return dict(cond=cond, map_id=mid, thr=thr, window=window, gated=bool(gated),
                mines=int(mines), builds=int(builds), true_mines=int(mines),
                true_builds=int(builds), steps=steps, success=bool(ok),
                timeout=bool(to), wrong=bool((not ok) and (not to)),
                door=("top" if env._pos[0] in top else
                      "bottom" if env._pos[0] in bot else "none"),
                proj=rb, released=int(prog[1]), released_prewall=int(rel_pre),
                released_postwall=int(rel_post), iters=iters, alpha=alpha,
                p_tool_max=float(prog[2]), trace=trace)


def run(jobs, workers):
    from concurrent.futures import ProcessPoolExecutor
    out = []
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=A4._ppo_init) as ex:
        for i, r in enumerate(ex.map(episode, jobs, chunksize=4)):
            out.append(r)
            if (i + 1) % 400 == 0:
                print(f"  ... {i+1}/{len(jobs)}", flush=True)
    return out


def summarise(rows, tag):
    f = lambda k: np.mean([r[k] for r in rows])            # noqa: E731
    rel = np.mean([r.get("released_prewall", r.get("released", 0)) for r in rows])
    print(f"  {tag:34s} n={len(rows):3d} succ {f('success'):.2f} "
          f"TO {f('timeout'):.2f} mines {f('true_mines'):5.1f} "
          f"builds {f('true_builds'):5.1f} steps {f('steps'):5.0f} "
          f"rel(pre-wall) {rel:5.1f}", flush=True)


def strip(rows):
    return [{k: v for k, v in r.items() if k != "trace"} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fit", "freeze", "test"])
    ap.add_argument("--thetas", default="0.02,0.005,0.001")
    ap.add_argument("--windows", default="8,15,25")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--fit-maps", type=int, default=14)
    ap.add_argument("--fit-rolls", type=int, default=6)
    ap.add_argument("--test-maps", type=int, default=60)
    ap.add_argument("--test-rolls", type=int, default=20)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--force-ungated", action="store_true",
                    help="ABLATION: run the test at the SAME frozen thetas with "
                         "the stuck-gate switched off, isolating the gate")
    ap.add_argument("--seed-offset", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    thetas = [float(x) for x in a.thetas.split(",")]
    windows = [int(x) for x in a.windows.split(",")]

    if a.stage == "fit":
        _, ids = A4.balanced_ids("train", a.fit_maps, seed=3)
        print(f"fit on {len(ids)} TRAIN balanced maps x {a.fit_rolls} rollouts")
        jobs = [(m, SEED0 + k, "baseline", 0.0, 15, a.eps, False, "train")
                for m in ids for k in range(a.fit_rolls)]
        rows = run(jobs, a.workers)
        summarise(rows, "FIT baseline")
        for cond in CONDS[1:]:
            for thr in thetas:
                # ungated reference at the same theta, then the gated ladder
                for gated, wins in ((False, [15]), (True, windows)):
                    for w in wins:
                        jb = [(m, SEED0 + k, cond, thr, w, a.eps, gated, "train")
                              for m in ids for k in range(a.fit_rolls)]
                        sub = run(jb, a.workers)
                        rows += sub
                        summarise(sub, f"FIT {cond} θ={thr:g} "
                                       f"{'gated w=' + str(w) if gated else 'ungated'}")
        (OUT / "fit_ppo_gated.json").write_text(json.dumps(strip(rows), indent=1))
        print(f"wrote fit_ppo_gated.json ({time.time()-t0:.0f}s)")

    elif a.stage == "freeze":
        rows = json.loads((OUT / "fit_ppo_gated.json").read_text())
        ops = {"_rule": dict(guard_success=GUARD_SUCCESS,
                             guard_timeout=GUARD_TIMEOUT, eps=a.eps,
                             note="fit on TRAIN balanced maps; frozen before test;"
                                  " strongest = lowest theta, then largest window")}
        base = [r for r in rows if r["cond"] == "baseline"]
        ops["baseline_fit"] = dict(n=len(base),
                                   success=round(float(np.mean([r["success"] for r in base])), 3),
                                   mines=round(float(np.mean([r["true_mines"] for r in base])), 2),
                                   builds=round(float(np.mean([r["true_builds"] for r in base])), 2))
        for cond in CONDS[1:]:
            cands = []
            for r in rows:
                if r["cond"] != cond or not r["gated"]:
                    continue
                cands.append((r["thr"], r["window"]))
            best = None
            for thr, w in sorted(set(cands)):
                sub = [r for r in rows if r["cond"] == cond and r["gated"]
                       and r["thr"] == thr and r["window"] == w]
                s = float(np.mean([r["success"] for r in sub]))
                to = float(np.mean([r["timeout"] for r in sub]))
                if s >= GUARD_SUCCESS and to <= GUARD_TIMEOUT:
                    key = (thr, -w)                       # lowest theta, largest window
                    if best is None or key < best[0]:
                        best = (key, thr, w, s, to)
            if best is None:
                thr, w = min(thetas), max(windows)
                ops[cond] = dict(theta=thr, window=w, qualified=False,
                                 rule="NO setting met the guard rails; ladder boundary")
            else:
                _, thr, w, s, to = best
                ops[cond] = dict(theta=thr, window=w, qualified=True,
                                 fit_success=round(s, 3), fit_timeout=round(to, 3),
                                 rule="strongest inside the guard rails")
            print(f"  frozen {cond}: {ops[cond]}")
        (OUT / "operating_points.json").write_text(json.dumps(ops, indent=1))
        print("wrote operating_points.json")

    else:
        ops = json.loads((OUT / "operating_points.json").read_text())
        _, ids = A4.balanced_ids("test", a.test_maps, seed=7)
        off = a.seed_offset
        conds = ("baseline",) if a.null else CONDS
        print(f"test on {len(ids)} HELD-OUT balanced maps x {a.test_rolls} rollouts"
              + (" [NULL]" if a.null else ""))
        rows = []
        for cond in conds:
            if cond == "baseline":
                thr, w, gated = 0.0, 15, False
            else:
                thr, w = ops[cond]["theta"], ops[cond]["window"]
                gated = not a.force_ungated
            jb = [(m, SEED0 + off + k, cond, thr, w, a.eps, gated, "test")
                  for m in ids for k in range(a.test_rolls)]
            sub = run(jb, a.workers)
            rows += sub
            summarise(sub, f"TEST {cond}")
        tag = ("test_ppo_gated_null" if a.null else
               "test_ppo_ungated_same_theta" if a.force_ungated else
               "test_ppo_gated")
        (OUT / f"{tag}.json").write_text(json.dumps(strip(rows), indent=1))
        if not a.null:
            keep = set(ids[:8])
            tr = {}
            for r in rows:
                if r["map_id"] in keep:
                    tr.setdefault(f"{r['cond']}|{r['map_id']}", []).append(
                        dict(steps=r["trace"], correct=r["success"],
                             door=r["door"], to=r["timeout"]))
            for k, v in tr.items():
                cond, mid = k.split("|")
                (OUT / f"trace_ppo_gated_{cond}_{mid}.json").write_text(
                    json.dumps({"balanced": dict(map_id=int(mid), rollouts=v)}))
        print(f"wrote {tag}.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
