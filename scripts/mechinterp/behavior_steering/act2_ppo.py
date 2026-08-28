#!/usr/bin/env python3
"""Act two -- textbook behaviour-steering methods flip the decision (PPO).

The behaviour campaign (ppo_campaign.py) asked "can we steer the behaviour
safely". This campaign asks the thesis question: apply the canonical methods
from the steering literature at doses that genuinely change tool behaviour and
rarely time out, and show the terminal decision flips to the WRONG door. The
mechanism claim: the flip is a belief corruption, visible as a shift of the
late-corridor belief coordinate h.v_belief toward the other class mean.

Methods under test (all standard in the literature):
  pg     policy-gradient steering  h' = h + eta * grad_h log pi(a_target | h)
  m1g    CAA-style contrast addition (route axis, gated coordinate-set)
  m3     SAE feature clamp (suppress the mine/build feature family)
  m1p    belief-orthogonalised contrast axis -- the MITIGATION contrast row

Every steered episode records the belief readback: the mean projection of h on
the unit belief axis over the late-corridor window (col - wall in [-8, 0), the
axis's own fitting bin), taken before the first wall crossing.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts/mechinterp:scripts/mechinterp/belief_report:scripts/figures:scripts/mechinterp/behavior_steering \
    python scripts/mechinterp/behavior_steering/act2_ppo.py --stage sham
  ... --stage grid | mech | qual
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

OUT = REPO / "outputs/behavior_steering/act2"
A_BUILD, A_MINE = 4, 5

BEL = np.load(REPO / "outputs/belief_report/steer_axis_ppo.npz")
V_BEL = (BEL["v"] / (np.linalg.norm(BEL["v"]) + 1e-12)).astype(np.float32)
MU_L, MU_R = float(BEL["mu_lakes"]), float(BEL["mu_rocky"])
MID = 0.5 * (MU_L + MU_R)


# ── the policy-gradient steering hook ────────────────────────────────────

def pg_hook(eta, target, store):
    """h' = h + eta * grad_h log pi(target | h). The adapter runs hooks under
    torch.no_grad(), so grads are re-enabled locally. `store` collects the
    cosine of each applied delta with the belief axis."""
    import torch
    import replay_episode as RE
    act, _ = RE._get_agent("ppo", "cpu")
    policy = act.policy

    def hook(h, t, info):
        with torch.enable_grad():
            ht = torch.tensor(h, dtype=torch.float32, requires_grad=True)
            logits, _ = policy._heads(ht[None])
            logp = torch.log_softmax(logits, -1)[0, target]
            g, = torch.autograd.grad(logp, ht)
        d = (eta * g.numpy()).astype(np.float32)
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            store.append(float(d @ V_BEL) / n)
        return (h + d).astype(np.float32)
    return hook


# ── delta-recording wrappers for the fixed-direction methods ─────────────

def rec_wrap(hook, store):
    def wrapped(h, t, info):
        h2 = hook(h, t, info)
        d = h2 - h
        n = float(np.linalg.norm(d))
        if n > 1e-8:
            store.append(float(d @ V_BEL) / n)
        return h2
    return wrapped


class RecCtx:
    """Same, for phase-gated hook objects (GatedCoordset)."""
    wants_ctx = True

    def __init__(self, inner, store):
        self.inner, self.store = inner, store

    def bind(self, ctx):
        return rec_wrap(self.inner.bind(ctx), self.store)


# ── belief readback ──────────────────────────────────────────────────────

def readback(hs, trace, wall):
    """Mean h.v_bel over pre-crossing steps with col-wall in [-8,0); falls
    back to the last pre-wall step if the window was never visited."""
    cols = np.array([s["c"] for s in trace[:len(hs)]])
    crossed = np.where(cols >= wall)[0]
    stop = int(crossed[0]) if len(crossed) else len(hs)
    crw = cols[:stop] - wall
    m = (crw >= -8) & (crw < 0)
    proj = np.asarray(hs[:stop], np.float32) @ V_BEL
    if m.any():
        val = float(proj[m].mean())
    elif stop:
        val = float(proj[np.argmax(cols[:stop])])
    else:
        val = float("nan")
    return val, bool(len(crossed))


def run_row(rec, mid, hook, lbias, store, cond, tool, cat):
    import ppo_campaign as PC
    r = PC.run_episode(rec, 1000 + mid, hook, lbias,
                       want_steps=True, want_h=True)
    proj, reached = readback(r["hs"], r["trace"], int(rec.wall_col))
    to = r["steps"] >= 799
    return dict(cond=cond, tool=tool, cat=cat, map_id=mid,
                mines=r["mines"], builds=r["builds"], steps=r["steps"],
                door=r["door"], success=r["success"], timeout=bool(to),
                wrong=bool((not r["success"]) and (not to)),
                proj=proj, reached=reached,
                cos_mean=float(np.mean(store)) if store else None,
                cos_sd=float(np.std(store)) if store else None,
                n_steer=len(store))


# ── stages ───────────────────────────────────────────────────────────────

def maps_for(cat, n):
    return [int(x) for x in BEL[cat][:n]]


def archive_sham(agent, rows):
    """Merge this agent's sham rows into the shared verification log; the
    three agents run from different environments, so merge, never clobber."""
    path = OUT / "sham_verify.json"
    log = json.loads(path.read_text()) if path.exists() else {}
    log[agent] = dict(rows=rows, n=len(rows),
                      match=sum(r["match"] for r in rows))
    path.write_text(json.dumps(log, indent=1))
    print("archived", path)


def stage_sham():
    """eta=0 PG hook must reproduce the plain replay action-for-action."""
    import ppo_campaign as PC
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    bad, rows = 0, []
    for cat, tgt in (("rocky", A_MINE), ("lakes", A_BUILD)):
        for mid in maps_for(cat, 5):
            base = PC.run_episode(pool[mid], 1000 + mid, want_steps=True)
            store = []
            sham = PC.run_episode(pool[mid], 1000 + mid,
                                  pg_hook(0.0, tgt, store), None,
                                  want_steps=True)
            same = ([s["c"] for s in base["trace"]] == [s["c"] for s in sham["trace"]]
                    and [s["r"] for s in base["trace"]] == [s["r"] for s in sham["trace"]]
                    and base["steps"] == sham["steps"]
                    and base["door"] == sham["door"])
            bad += not same
            rows.append(dict(cat=cat, map_id=mid, match=bool(same),
                             steps=base["steps"], door=base["door"]))
            print(f"sham {cat} map {mid:4d}: {'MATCH' if same else 'DIFFERS'} "
                  f"steps {base['steps']} vs {sham['steps']}", flush=True)
    archive_sham("ppo", rows)
    print("SHAM", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)


def stage_grid(doses, n_maps):
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    rows = []

    def log(sub, tag):
        n = len(sub)
        print(f"{tag:26s} n={n:3d} succ {np.mean([r['success'] for r in sub]):.2f} "
              f"WRONG {np.mean([r['wrong'] for r in sub]):.2f} "
              f"TO {np.mean([r['timeout'] for r in sub]):.2f} "
              f"mines {np.mean([r['mines'] for r in sub]):5.1f} "
              f"builds {np.mean([r['builds'] for r in sub]):5.1f} "
              f"proj {np.nanmean([r['proj'] for r in sub]):+6.2f}", flush=True)

    for cat, tool, tgt in (("rocky", "mine", A_MINE), ("lakes", "build", A_BUILD)):
        maps = maps_for(cat, n_maps)
        base = [run_row(pool[mid], mid, None, None, [], "baseline", tool, cat)
                for mid in maps]
        rows += base
        log(base, f"{cat} baseline")
        for eta in doses:
            sub = []
            for mid in maps:
                store = []
                sub.append(run_row(pool[mid], mid, pg_hook(eta, tgt, store),
                                   None, store, f"pg_{eta:+.2f}", tool, cat))
            rows += sub
            log(sub, f"{cat} pg eta={eta:+.2f}")
        (OUT / "ppo_pg_grid.json").write_text(json.dumps(rows))
    print("wrote", OUT / "ppo_pg_grid.json", len(rows), "rows")


def stage_mech(n_maps):
    """The other textbook methods, rerun WITH belief readback."""
    import ppo_campaign as PC
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    rows = []
    for cat, tool in (("rocky", "mine"), ("lakes", "build")):
        maps = maps_for(cat, n_maps)
        for cond in ("m1g_1.5", "m3_sup", "m1p_1"):
            sub = []
            for mid in maps:
                hook, lb = PC.make_condition(cond, tool)
                store = []
                if getattr(hook, "wants_ctx", False):
                    hook = RecCtx(hook, store)
                elif hook is not None:
                    hook = rec_wrap(hook, store)
                sub.append(run_row(pool[mid], mid, hook, lb, store,
                                   cond, tool, cat))
            rows += sub
            n = len(sub)
            print(f"{cat} {cond:8s} n={n} succ {np.mean([r['success'] for r in sub]):.2f} "
                  f"WRONG {np.mean([r['wrong'] for r in sub]):.2f} "
                  f"TO {np.mean([r['timeout'] for r in sub]):.2f} "
                  f"cos {np.mean([r['cos_mean'] for r in sub if r['cos_mean'] is not None]):+.2f}",
                  flush=True)
            (OUT / "ppo_textbook_mech.json").write_text(json.dumps(rows))
    print("wrote", OUT / "ppo_textbook_mech.json", len(rows), "rows")


def stage_qual(map_id, conds):
    """Figure-7.5 seeds (2000+k, 20 rollouts) on one thesis map, ghost schema
    with door/timeout fields so the panel can count wrong doors. Conditions:
    'baseline', 'pg_<eta>', or any ppo_campaign condition name (m3_sup, ...)."""
    import ppo_campaign as PC
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    rec = pool[map_id]
    cat = rec.category
    tool = "mine" if cat in ("rocky", "balanced") else "build"
    tgt = A_MINE if tool == "mine" else A_BUILD
    for label in conds:
        rolls = []
        for k in range(20):
            store = []
            if label == "baseline":
                hook, lb = None, None
            elif label.startswith("pg_"):
                hook, lb = pg_hook(float(label[3:]), tgt, store), None
            else:
                hook, lb = PC.make_condition(label, tool)
            r = PC.run_episode(rec, 2000 + k, hook, lb, want_steps=True)
            to = r["steps"] >= 799
            rolls.append(dict(steps=r["trace"], correct=r["success"],
                              door=r["door"], to=bool(to)))
        out = {cat: dict(map_id=int(map_id), rollouts=rolls)}
        p = OUT / f"ppo_qual_{map_id}_{label}.json"
        p.write_text(json.dumps(out))
        ok = sum(r["correct"] for r in rolls)
        wr = sum((not r["correct"]) and r["door"] != "none" and not r["to"]
                 for r in rolls)
        to = sum(r["to"] for r in rolls)
        mines = sum(1 for r in rolls for s in r["steps"]
                    if s["ev"] and s["ev"]["kind"] == "mine")
        builds = sum(1 for r in rolls for s in r["steps"]
                     if s["ev"] and s["ev"]["kind"] == "build")
        print(f"map {map_id} {label:10s} succ {ok}/20 wrong {wr} to {to} "
              f"mines {mines} builds {builds} -> {p.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["sham", "grid", "mech", "qual"])
    ap.add_argument("--doses", default="-0.75,-0.5,-0.25,-0.1,0.1,0.25,0.5,0.75")
    ap.add_argument("--n-maps", type=int, default=50)
    ap.add_argument("--map", type=int, default=77)
    ap.add_argument("--conds", default="baseline,pg_-0.50")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "sham":
        stage_sham()
    elif a.stage == "grid":
        stage_grid([float(x) for x in a.doses.split(",")], a.n_maps)
    elif a.stage == "mech":
        stage_mech(min(a.n_maps, 25))
    else:
        stage_qual(a.map, a.conds.split(","))


if __name__ == "__main__":
    main()
