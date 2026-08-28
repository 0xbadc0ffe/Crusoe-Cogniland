#!/usr/bin/env python3
"""Test the repo's OWN steering method (GradientClamp) the way we tested the
textbook ones: does it deliver the commanded behaviour, and what happens to the
terminal decision?

`src/cogniland/bridge_tunnel/steering.py` (0xbadc0ffe) predates the act-1/2/3
campaigns and was never scored on the door. GradientClamp is an iterative
minimal-edit clamp: each step it takes unit-norm gradient steps on the GRU
state until P(target action) is under `threshold`, then stops -- so it edits
only as much as the constraint requires, which is a genuinely different (and a
priori safer) design than a fixed-dose push.

Per its own docstring, `threshold` -- not `alpha` -- is the depth knob, and a
lower threshold needs more iterations, so the ladder here sweeps thresholds at
fixed alpha with max_iters raised to match.

Arms, on the same held-out maps and metrics as act2/act3 so the numbers are
comparable:
  clamp_<thr>        suppress the category's tool action below thr
  clamp_<thr>_orth   the same, with the module's OWN project_out(belief axis)
                     correction -- their orthogonalisation hook, our axis
  sham               alpha=0, must reproduce the plain replay action for action

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts/mechinterp:scripts/mechinterp/belief_report:scripts/figures:scripts/mechinterp/behavior_steering \
    python scripts/mechinterp/behavior_steering/act3_gradclamp.py --stage sham|grid|qual
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

OUT = REPO / "outputs/behavior_steering/act3"
A_BUILD, A_MINE = 4, 5
TOOL_ACT = {"mine": A_MINE, "build": A_BUILD}

BEL = np.load(REPO / "outputs/belief_report/steer_axis_ppo.npz")
V_BEL = (BEL["v"] / (np.linalg.norm(BEL["v"]) + 1e-12)).astype(np.float32)


def make_clamp_hook(policy, tool, thr, alpha, max_iters, use_orth, store):
    """Wrap the repo's GradientClamp in our (h_np, t, info) hook contract."""
    import torch
    from cogniland.bridge_tunnel.steering import (
        ClampTerm, GradientClamp, project_out)

    terms = [ClampTerm(head="actor", index=TOOL_ACT[tool], mode="suppress",
                       threshold=thr)]
    corrections = ()
    if use_orth:
        corrections = (project_out(torch.from_numpy(V_BEL)),)
    clamp = GradientClamp(policy, terms, alpha=alpha, max_iters=max_iters,
                          corrections=corrections,
                          warn_on_nonconvergence=False)

    def hook(h, t, info):
        x = torch.from_numpy(np.asarray(h, np.float32)).reshape(1, 1, -1)
        with torch.no_grad():
            p_before = torch.softmax(policy.actor(x.squeeze(0)), -1)[
                0, TOOL_ACT[tool]].item()
        x2 = clamp(x, t, {})
        with torch.no_grad():
            p_after = torch.softmax(policy.actor(x2.squeeze(0)), -1)[
                0, TOOL_ACT[tool]].item()
        store.append((p_before, p_after))
        return x2.reshape(-1).numpy().astype(np.float32)

    return hook


def run_row(rec, mid, hook, store, cond, tool, cat):
    from act2_ppo import readback
    import ppo_campaign as PC
    r = PC.run_episode(rec, 1000 + mid, hook, None, want_steps=True, want_h=True)
    proj, _ = readback(r["hs"], r["trace"], int(rec.wall_col))
    to = r["steps"] >= 799
    row = dict(cond=cond, tool=tool, cat=cat, map_id=mid, mines=r["mines"],
               builds=r["builds"], steps=r["steps"], door=r["door"],
               success=r["success"], timeout=bool(to),
               wrong=bool((not r["success"]) and (not to)), proj=proj)
    if store:
        row.update(p_before=float(np.mean([a for a, _ in store])),
                   p_after=float(np.mean([b for _, b in store])),
                   p_after_max=float(np.max([b for _, b in store])),
                   n_steps_edited=int(sum(1 for a, b in store if b < a - 1e-6)))
    return row


def maps_for(cat, n):
    return [int(x) for x in BEL[cat][:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sham", "grid", "qual"])
    ap.add_argument("--thresholds", default="0.05,0.01,0.001,0.00001")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-iters", type=int, default=25)
    ap.add_argument("--n-maps", type=int, default=25)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import replay_episode as RE
    act, _ = RE._get_agent("ppo", "cpu")
    policy = act.policy
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    thrs = [float(x) for x in a.thresholds.split(",")]

    if a.stage == "sham":
        import ppo_campaign as PC
        bad, rows = 0, []
        for cat, tool in (("rocky", "mine"), ("lakes", "build")):
            for mid in maps_for(cat, 5):
                base = PC.run_episode(pool[mid], 1000 + mid, want_steps=True)
                st = []
                sham = PC.run_episode(
                    pool[mid], 1000 + mid,
                    make_clamp_hook(policy, tool, 0.05, 0.0, a.max_iters,
                                    False, st),
                    None, want_steps=True)
                same = (base["steps"] == sham["steps"]
                        and base["door"] == sham["door"]
                        and [s["c"] for s in base["trace"]] ==
                            [s["c"] for s in sham["trace"]])
                bad += not same
                rows.append(dict(cat=cat, map_id=mid, match=bool(same),
                                 steps=base["steps"], door=base["door"]))
                print(f"sham {cat} {mid:4d} {'MATCH' if same else 'DIFFERS'}",
                      flush=True)
        path = OUT / "gradclamp_sham.json"
        path.write_text(json.dumps({"ppo": dict(
            rows=rows, n=len(rows), match=sum(r["match"] for r in rows))},
            indent=1))
        print("SHAM", "PASS" if bad == 0 else f"FAIL ({bad})")
        sys.exit(1 if bad else 0)

    if a.stage == "grid":
        rows = []

        def log(sub, tag):
            f = lambda k: np.mean([r[k] for r in sub])          # noqa: E731
            extra = ""
            if "p_after" in sub[0]:
                extra = (f" p(tool) {f('p_before'):.3f}->{f('p_after'):.5f} "
                         f"max {np.max([r['p_after_max'] for r in sub]):.3f}")
            print(f"{tag:30s} n={len(sub):3d} succ {f('success'):.2f} "
                  f"WRONG {f('wrong'):.2f} TO {f('timeout'):.2f} "
                  f"mines {f('mines'):5.1f} builds {f('builds'):5.1f} "
                  f"proj {np.nanmean([r['proj'] for r in sub]):+6.2f}{extra}",
                  flush=True)

        for cat, tool in (("rocky", "mine"), ("lakes", "build")):
            ids = maps_for(cat, a.n_maps)
            base = [run_row(pool[m], m, None, [], "baseline", tool, cat)
                    for m in ids]
            rows += base
            log(base, f"{cat} baseline")
            for thr in thrs:
                for orth in (False, True):
                    sub = []
                    for mid in ids:
                        st = []
                        hk = make_clamp_hook(policy, tool, thr, a.alpha,
                                             a.max_iters, orth, st)
                        sub.append(run_row(
                            pool[mid], mid, hk, st,
                            f"clamp_{thr:g}{'_orth' if orth else ''}",
                            tool, cat))
                    rows += sub
                    log(sub, f"{cat} clamp {thr:g}{' orth' if orth else ''}")
            (OUT / "gradclamp_grid.json").write_text(json.dumps(rows, indent=1))
        print("wrote gradclamp_grid.json", len(rows), "rows")

    if a.stage == "qual":
        import ppo_campaign as PC
        from act2_ppo import readback
        for mid, cat, tool in ((626, "lakes", "build"), (77, "rocky", "mine"),
                               (99, "balanced", "mine")):
            for name, thr in [("baseline", None)] + [(f"clamp {t:g}", t)
                                                     for t in thrs]:
                rolls = []
                for k in range(20):
                    st = []
                    hk = (None if thr is None else
                          make_clamp_hook(policy, tool, thr, a.alpha,
                                          a.max_iters, False, st))
                    r = PC.run_episode(pool[mid], 2000 + k, hk, None,
                                       want_steps=True, want_h=True)
                    rolls.append(dict(steps=r["trace"], correct=bool(r["success"]),
                                      door=r["door"], to=bool(r["steps"] >= 799)))
                (OUT / f"gradclamp_qual_{name}_{mid}.json").write_text(
                    json.dumps({cat: dict(map_id=mid, rollouts=rolls)}))
                nw = sum(1 for x in rolls if not x["correct"] and not x["to"])
                print(f"map {mid} {name:14s} succ "
                      f"{sum(x['correct'] for x in rolls)}/20 wrong {nw} "
                      f"TO {sum(x['to'] for x in rolls)}", flush=True)


if __name__ == "__main__":
    main()
