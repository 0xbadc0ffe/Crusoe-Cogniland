#!/usr/bin/env python3
"""Behaviour-steering campaign for the world models (DreamerV3, STORM).

Methods (per agent, chosen after pilots -- see WM_BEHAVIOR.md):
  route-set   coordinate-set on the within-map route-intent axis, applied over
              the pre-obstacle window (Dreamer: deter hook; STORM: per-step
              payload). Doses are multiples of the fitted class gap.
  tool-add    per-step additive displacement along the context-matched tool
              axis (the perseveration negative from the PPO prep, measured
              honestly here too).
  logit       soft bias on the actor logits (Dreamer wrapper, verified
              byte-identical at zero bias; STORM jit variant if attempted).

Arms:
  grid        held-out map grid, all directions x doses
  controls    sham + random matched-displacement directions
  qual        maps 626/77/99 with the figure-7.5 seeds -> ghost-schema JSONs

Every episode is a seed-exact replay; tool counts in the GRID are counted from
action ids (stated in the report); the QUAL traces record true env events.

  # dreamer (conda r2dreamer)
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:r2dreamer_model:scripts/mechinterp:scripts/figures \
    python scripts/mechinterp/behavior_steering/wm_campaign.py --agent dreamer --arm grid
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

from replay_episode import replay, _get_agent  # noqa: E402

OUT = REPO / "outputs/behavior_steering"
A_BUILD, A_MINE = 4, 5
QUAL_MAPS = {626: "lakes", 77: "rocky", 99: "balanced"}


def load_kit(agent):
    """Axes + targets + per-map corridor-entry steps + test-map lists."""
    z = np.load(OUT / agent / "axes.npz")
    meta = json.loads((OUT / agent / "axes_meta.json").read_text())
    zb = np.load(REPO / f"outputs/belief_report/steer_axis_{agent}.npz")
    entry = dict(zip(zb["entry_keys"].tolist(), zb["entry_vals"].tolist()))
    test_ids = {c: [int(i) for i in zb[c]] for c in ("lakes", "rocky", "balanced")}
    kit = dict(entry=entry, test=test_ids, meta=meta)
    for k in z.files:
        kit[k] = z[k].astype(np.float32)
    # within-map route axes, if fitted
    f = OUT / agent / "route_axes_withinmap.npz"
    if f.exists():
        zw = np.load(f)
        for k in zw.files:
            kit[f"wm_{k}"] = zw[k]
        kit["wm_meta"] = json.loads(
            (OUT / agent / "route_axes_withinmap.json").read_text())
    return kit


def dreamer_hook(v, mode, t0, delta=None, target=None):
    """deter-space intervention over the pre-obstacle window [2, t0)."""
    v = v.astype(np.float32)

    def hook(d, t, info):
        on = (t == 2) if mode == "transient" else (2 <= t < t0)
        if not on:
            return d
        if target is not None:                       # coordinate set
            return d + (target - float(d @ v)) * v
        return d + delta * v                         # additive displacement
    return hook


def storm_payload(v, mode, t0, delta=None, target=None):
    """per-step payload; gate(t) -> (on, c, rel). mode: win = [0,t0), all."""
    c = float(target) if target is not None else float(delta)
    rel = target is None
    if mode == "all":
        gate = lambda t: (True, c, rel)                      # noqa: E731
    else:
        gate = lambda t: (t < t0, c, rel)                    # noqa: E731
    return {"v": v.astype(np.float32), "gate": gate}


def run_one(agent, mid, device, seed=None, hook=None, payload=None, lbias=None):
    r = replay(agent, mid, device=device, seed=seed,
               hook=(payload if agent == "storm" and payload is not None else hook),
               logit_bias=lbias)
    acts = r["actions"]
    return dict(map_id=mid, success=bool(r["success"]), door=r["door"],
                steps=r["steps"],
                mines=int(sum(a == A_MINE for a in acts)),
                builds=int(sum(a == A_BUILD for a in acts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["dreamer", "storm"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--methods", default="route:1,2|logit:1.5,3")
    ap.add_argument("--qual-conds", default="{}")
    a = ap.parse_args()
    parsed = []
    for m in a.methods.split("|"):
        name, vals = m.split(":")
        key = "lam" if name != "logit" else "bias"
        parsed.append((name, [{key: float(x)} for x in vals.split(",")]))
    a.methods = parsed
    kit = load_kit(a.agent)
    dev = a.device
    OUTA = OUT / a.agent
    OUTA.mkdir(parents=True, exist_ok=True)

    def route_axis(cat):
        key = "wm_v_rocky" if cat in ("rocky", "balanced") else "wm_v_lakes"
        gap = kit["wm_meta"]["rocky" if cat in ("rocky", "balanced") else "lakes"]["gap"]
        return kit[key].astype(np.float32), float(gap)

    def make_intervention(agent, method, direction, cat, mid, lam=1.0, bias=3.0):
        """-> dict(hook=, payload=, lbias=) for run_one."""
        t0 = int(kit["entry"].get(mid, 60))
        sgn = -1.0 if direction.startswith("sup") else +1.0
        if method == "route":
            v, gap = route_axis(cat)
            delta = sgn * lam * gap
            if direction == "sup-both" and cat == "balanced":
                v2, gap2 = kit["wm_v_lakes"].astype(np.float32), \
                           float(kit["wm_meta"]["lakes"]["gap"])
                if agent == "storm":
                    # storm payload supports one axis; combine into one vector
                    w = (-lam * gap) * v + (-lam * gap2) * v2
                    n = float(np.linalg.norm(w))
                    return dict(payload=storm_payload(w / n, "win", t0, delta=n))
                def hook(d, t, info, v=v, v2=v2):
                    if 2 <= t < t0:
                        d = d + (-lam * gap) * v + (-lam * gap2) * v2
                    return d
                return dict(hook=hook)
            if agent == "storm":
                return dict(payload=storm_payload(v, "win", t0, delta=delta))
            return dict(hook=dreamer_hook(v, "window", t0, delta=delta))
        if method == "tooladd":
            if direction == "sup-both":
                vm = kit["v_mine"].astype(np.float32)
                vb = kit["v_build"].astype(np.float32)
                w = -lam * (kit["meta"]["v_mine"]["proj_sd"] * vm
                            + kit["meta"]["v_build"]["proj_sd"] * vb)
                n = float(np.linalg.norm(w)) + 1e-12
                if agent == "storm":
                    return dict(payload=storm_payload(w / n, "all", t0, delta=n))
                return dict(hook=lambda d, t, info, w=w: d + w)
            key = "v_mine" if "mine" in direction or cat != "lakes" else "v_build"
            v = kit[key].astype(np.float32)
            sd = kit["meta"][key]["proj_sd"]
            delta = sgn * lam * sd
            if agent == "storm":
                return dict(payload=storm_payload(v, "all", t0, delta=delta))
            def hook(d, t, info, v=v):
                return d + delta * v
            return dict(hook=hook)
        if method == "logit":
            tool = A_MINE if ("mine" in direction or cat == "rocky") else A_BUILD
            b = np.zeros(6, np.float32)
            b[tool] = sgn * bias
            if direction == "sup-both":
                b[A_MINE] = b[A_BUILD] = -bias
            return dict(lbias=lambda t, b=b: b)
        raise ValueError(method)

    def grid_arm():
        rows = []
        jobs = []
        for cat, ids, dirs in (
                ("rocky", kit["test"]["rocky"][:40], ["sup-mine", "inc-mine"]),
                ("lakes", kit["test"]["lakes"][:40], ["sup-build", "inc-build"]),
                ("balanced", kit["test"]["balanced"][:25], ["sup-mine", "sup-both"])):
            for mid in ids:
                jobs.append((cat, mid, None, None, {}))          # baseline
                for d in dirs:
                    for meth, params in a.methods:
                        for pv in params:
                            jobs.append((cat, mid, meth, d, pv))
        print(f"grid: {len(jobs)} episodes", flush=True)
        for i, (cat, mid, meth, d, pv) in enumerate(jobs):
            if meth is None:
                r = run_one(a.agent, mid, dev)
                r.update(method="baseline", direction="-", params={})
            else:
                iv = make_intervention(a.agent, meth, d, cat, mid, **pv)
                r = run_one(a.agent, mid, dev, **iv)
                r.update(method=meth, direction=d, params=pv)
            r.update(cat=cat)
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)
        (OUTA / "grid.json").write_text(json.dumps(rows))
        print("wrote grid.json", len(rows))

    def controls_arm():
        """Controls for both shipped methods on 15 rocky maps.
        logit family: sham (zero bias through the steered path) + specificity
        (same-magnitude bias on the WRONG tool: build on rocky). tooladd
        family: sham (lam=0 through the hook path) + 5 random unit directions
        at the matched displacement |delta| = lam*sd_mine."""
        rows = []
        rng = np.random.default_rng(0)
        v = kit["v_mine"].astype(np.float32)
        sd = float(kit["meta"]["v_mine"]["proj_sd"])
        lam = 2.0
        D_ = v.shape[0]
        for mid in kit["test"]["rocky"][:15]:
            base = run_one(a.agent, mid, dev)
            base.update(method="baseline", direction="-", params={}, cat="rocky")
            rows.append(base)
            r = run_one(a.agent, mid, dev,
                        lbias=lambda t: np.zeros(6, np.float32))
            r.update(method="sham-logit", direction="-",
                     params=dict(bias=0.0), cat="rocky")
            rows.append(r)
            b = np.zeros(6, np.float32); b[A_BUILD] = -3.0
            r = run_one(a.agent, mid, dev, lbias=lambda t, b=b: b)
            r.update(method="wrongtool-logit", direction="sup-build",
                     params=dict(bias=3.0), cat="rocky")
            rows.append(r)
            iv = make_intervention(a.agent, "tooladd", "sup-mine", "rocky",
                                   mid, lam=0.0)
            r = run_one(a.agent, mid, dev, **iv)
            r.update(method="sham-tooladd", direction="sup-mine",
                     params=dict(lam=0.0), cat="rocky")
            rows.append(r)
            for k in range(5):
                rv = rng.standard_normal(D_).astype(np.float32)
                rv /= np.linalg.norm(rv)
                delta = -lam * sd
                if a.agent == "storm":
                    iv = dict(payload=storm_payload(rv, "all", 0, delta=delta))
                else:
                    iv = dict(hook=lambda d, t, info, rv=rv, delta=delta:
                              d + delta * rv)
                r = run_one(a.agent, mid, dev, **iv)
                r.update(method="random", direction="matched",
                         params=dict(dir=k, lam=lam), cat="rocky")
                rows.append(r)
        (OUTA / "controls.json").write_text(json.dumps(rows))
        print("wrote controls.json", len(rows))

    def qual_arm():
        from cogniland.bridge_tunnel.env import BridgeTunnelEnv
        from paper_rollouts import FORKWALL_KWARGS
        FACE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        act, reset = _get_agent(a.agent, dev)
        conds = json.loads(a.qual_conds)
        pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
        for label, spec in conds.items():
            out = {}
            for mid, cat in QUAL_MAPS.items():
                if spec.get("cats") and cat not in spec["cats"]:
                    continue
                rec = pool[mid]
                rolls = []
                for k in range(20):
                    np.random.seed(2000 + k)
                    try:
                        import torch
                        torch.manual_seed(2000 + k)
                    except Exception:
                        pass
                    iv = {}
                    if spec.get("method"):
                        iv = make_intervention(a.agent, spec["method"],
                                               spec["direction"], cat, mid,
                                               **spec.get("params", {}))
                    if hasattr(act, "set_hook"):
                        act.set_hook(iv.get("payload") if a.agent == "storm"
                                     else iv.get("hook"))
                    if hasattr(act, "set_logit_bias"):
                        act.set_logit_bias(iv.get("lbias"))
                    if hasattr(act, "set_seed"):
                        act.set_seed(2000 + k)
                    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
                    obs, _ = env.reset(); reset()
                    steps = [dict(r=int(env._pos[0]), c=int(env._pos[1]),
                                  facing=int(env._facing), ev=None)]
                    for t in range(FORKWALL_KWARGS["max_steps"]):
                        ac = act(obs, False)
                        obs, _, term, trunc, info = env.step(ac)
                        ev = None
                        if ac in (A_BUILD, A_MINE) and (info.get("placed") or info.get("mined")):
                            dr, dc = FACE_DELTA[int(info["facing"])]
                            ev = dict(kind="build" if ac == A_BUILD else "mine",
                                      r=int(env._pos[0] + dr), c=int(env._pos[1] + dc))
                        steps.append(dict(r=int(env._pos[0]), c=int(env._pos[1]),
                                          facing=int(info["facing"]), ev=ev))
                        if term or trunc:
                            break
                    fr = env._pos
                    topr = {p[0] for p in rec.top_goal_cells}
                    botr = {p[0] for p in rec.bottom_goal_cells}
                    rolls.append(dict(
                        steps=steps,
                        correct=bool(env._pos in (env._correct_cells or set())),
                        door=("top" if fr[0] in topr else
                              "bottom" if fr[0] in botr else "none"),
                        to=bool(len(steps) - 1 >= 799)))
                out[cat] = dict(map_id=int(mid), rollouts=rolls)
                nt = sum(any(s["ev"] for s in r["steps"]) for r in rolls)
                ok = sum(r["correct"] for r in rolls)
                print(f"  {label:14s} map {mid:4d}: tool-users {nt}/20 correct {ok}/20",
                      flush=True)
            (OUTA / f"qual_{label}.json").write_text(json.dumps(out))
        if hasattr(act, "set_hook"):
            act.set_hook(None)
        if hasattr(act, "set_logit_bias"):
            act.set_logit_bias(None)

    if a.arm == "grid":
        grid_arm()
    elif a.arm == "controls":
        controls_arm()
    elif a.arm == "qual":
        qual_arm()
    else:
        raise SystemExit(f"unknown arm {a.arm}")


if __name__ == "__main__":
    main()
