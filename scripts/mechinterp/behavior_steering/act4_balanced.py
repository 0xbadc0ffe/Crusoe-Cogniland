#!/usr/bin/env python3
"""Act four -- behavioural steering on BALANCED maps, fit on train, tested held out.

Balanced maps are the showcase for behaviour steering: both doors are rewarded
and several route types exist (bridge the water, tunnel the rock, or go round),
so the behaviour axis and the decision axis separate. Success cannot see the
decision here -- which is exactly why the decision must be measured directly,
as a per-map PAIRED shift in the door split.

Protocol, and the thing that makes this campaign different from acts 1-3: every
knob (clamp threshold, tilt lambda, logit bias) is FIT on balanced maps drawn
from the TRAINING pool, frozen into act4/operating_points.json, and only then
evaluated once on held-out balanced maps from the test pool. Nothing is retuned
after the freeze.

Behaviour metric (user-specified): SUCCESSFUL tool usages only -- events where
a block was really mined or placed (info["mined"] / info["placed"]), never
action presses, which repeat when blocked. Each map's own baseline is 100% and
the report gives the per-map percentage change.

Arms, one per agent x intervention surface:
  ppo_clamp      GradientClamp on the actor head + the module's own
                 project_out(belief axis) correction          (state-level)
  storm_logit    soft actor-logit bias                        (actuator-level)
  dreamer_logit  soft actor-logit bias                        (actuator-level)
  dreamer_tilt   imagination tilt, log pi' = log pi - lam*E[tools]  (plan-level)

  CUDA_VISIBLE_DEVICES= python act4_balanced.py --stage fit  --arm ppo_clamp
  python act4_balanced.py --stage freeze
  CUDA_VISIBLE_DEVICES= python act4_balanced.py --stage test --arm ppo_clamp
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ("src", "scripts/mechinterp", "scripts/mechinterp/belief_report",
          "scripts/figures", "scripts/mechinterp/behavior_steering"):
    sys.path.insert(0, str(REPO / p))

OUT = REPO / "outputs/behavior_steering/act4"
TRAIN_PKL = REPO / "data/bridge_tunnel/forkwall6k/train.pkl"
TEST_PKL = REPO / "data/bridge_tunnel/forkwall6k/test.pkl"
A_BUILD, A_MINE = 4, 5
FACE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
CONDS = ("baseline", "sup_mine", "sup_build", "sup_both")
TOOLS_OF = {"baseline": (), "sup_mine": ("mine",), "sup_build": ("build",),
            "sup_both": ("mine", "build")}
SEED0 = 2000                      # the campaign's rollout-seed convention

# selection rule, pre-registered: strongest knob whose FIT-set guard rails hold
GUARD_SUCCESS, GUARD_TIMEOUT = 0.85, 0.10


def balanced_ids(split, n, seed=0):
    """Deterministic sample of balanced map ids from one pool."""
    pool = pickle.load(open(TRAIN_PKL if split == "train" else TEST_PKL, "rb"))
    ids = [i for i, r in enumerate(pool) if r.category == "balanced"]
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(ids))[:n]
    return pool, sorted(int(ids[i]) for i in pick)


def _door(env, rec):
    fr = env._pos
    top = {p[0] for p in rec.top_goal_cells}
    bot = {p[0] for p in rec.bottom_goal_cells}
    return "top" if fr[0] in top else "bottom" if fr[0] in bot else "none"


# ── PPO: GradientClamp + project_out ─────────────────────────────────────

_PPO = {}


def _ppo_init():
    """Per-process agent, one torch thread (we parallelise over episodes)."""
    import torch
    torch.set_num_threads(1)
    import replay_episode as RE
    act, _ = RE._get_agent("ppo", "cpu")
    _PPO["act"] = act
    _PPO["policy"] = act.policy
    z = np.load(REPO / "outputs/belief_report/steer_axis_ppo.npz")
    _PPO["v_bel"] = (z["v"] / (np.linalg.norm(z["v"]) + 1e-12)).astype(np.float32)


def _clamp_hook(cond, thr, alpha, max_iters, orth=True):
    """suppress-both uses TWO independent terms (each tool constrained on its
    own probability), not one group term on their sum -- the stricter reading.

    `orth` toggles the module's own project_out correction on the belief axis.
    orth=True is the campaign default; orth=False is the ABLATION: the same
    constraint, satisfied to the same threshold, with the belief component of
    the edit left in. Because the clamp iterates until P(tool) < threshold,
    both variants meet the same behavioural target by construction, so the
    comparison isolates the correction, not the dose."""
    import torch
    from cogniland.bridge_tunnel.steering import (
        ClampTerm, GradientClamp, project_out)
    tools = TOOLS_OF[cond]
    if not tools:
        return None
    idx = {"mine": A_MINE, "build": A_BUILD}
    terms = [ClampTerm(head="actor", index=idx[t], mode="suppress",
                       threshold=thr) for t in tools]
    clamp = GradientClamp(
        _PPO["policy"], terms, alpha=alpha, max_iters=max_iters,
        corrections=((project_out(torch.from_numpy(_PPO["v_bel"])),)
                     if orth else ()),
        warn_on_nonconvergence=False)

    def hook(h, t, info):
        x = torch.from_numpy(np.asarray(h, np.float32)).reshape(1, 1, -1)
        return clamp(x, t, {}).reshape(-1).numpy().astype(np.float32)
    return hook


def ppo_episode(job):
    """One PPO rollout. job = (map_id, seed, cond, thr, alpha, iters, split)
    or the 8-tuple with a trailing `orth` flag (default True)."""
    import torch
    if len(job) == 8:
        mid, seed, cond, thr, alpha, iters, split, orth = job
    else:
        mid, seed, cond, thr, alpha, iters, split = job
        orth = True
    if "act" not in _PPO:
        _ppo_init()
    pool = _PPO.setdefault(
        f"pool_{split}",
        pickle.load(open(TRAIN_PKL if split == "train" else TEST_PKL, "rb")))
    rec = pool[mid]
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS
    act = _PPO["act"]
    np.random.seed(seed); torch.manual_seed(seed)
    act.set_hook(_clamp_hook(cond, thr, alpha, iters, orth))
    act.set_logit_bias(None)
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset()
    from replay_episode import _get_agent
    _get_agent("ppo", "cpu")[1]()                   # reset the GRU state
    mines = builds = 0
    hs, cols, trace = [], [], []
    for t in range(FORKWALL_KWARGS["max_steps"]):
        a = act(obs, False)
        hs.append(act.get_state().astype(np.float32))
        cols.append(int(env._pos[1]))
        obs, _, term, trunc, info = env.step(a)
        ev = None
        if a in (A_BUILD, A_MINE) and (info.get("placed") or info.get("mined")):
            dr, dc = FACE_DELTA[int(info["facing"])]
            ev = dict(kind="build" if a == A_BUILD else "mine",
                      r=int(env._pos[0] + dr), c=int(env._pos[1] + dc))
            if a == A_MINE:
                mines += 1
            else:
                builds += 1
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
    proj = H[:stop] @ _PPO["v_bel"] if stop else np.array([])
    rb = (float(proj[m].mean()) if m.any() else
          float(proj[np.argmax(C[:stop])]) if stop else float("nan"))
    return dict(agent="ppo", method="clamp", map_id=mid, seed=seed, cond=cond,
                knob=0.0 if cond == "baseline" else thr, split=split, steps=steps, door=_door(env, rec),
                success=bool(env._pos in (env._correct_cells or set())),
                timeout=bool(steps >= 799), mines=mines, builds=builds,
                proj=rb, trace=trace)


def run_ppo(jobs, workers):
    from concurrent.futures import ProcessPoolExecutor
    rows = []
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_ppo_init) as ex:
        for i, r in enumerate(ex.map(ppo_episode, jobs, chunksize=4)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"  ... {i + 1}/{len(jobs)} episodes", flush=True)
    return rows


# ── world models ─────────────────────────────────────────────────────────

def wm_logit_episode(act, reset, rec, mid, seed, cond, bias, agent, kit=None):
    """Soft actor-logit bias, driven through the env so TRUE events are counted."""
    import torch
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS
    b = np.zeros(6, np.float32)
    for tool in TOOLS_OF[cond]:
        b[A_MINE if tool == "mine" else A_BUILD] = -abs(bias)
    np.random.seed(seed); torch.manual_seed(seed)
    if hasattr(act, "set_hook"):
        act.set_hook(None)
    act.set_logit_bias(None if not TOOLS_OF[cond] else (lambda t, _b=b: _b))
    if hasattr(act, "set_seed"):
        act.set_seed(seed)
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset(); reset()
    mines = builds = 0
    feats, cols, trace = [], [], []
    for t in range(FORKWALL_KWARGS["max_steps"]):
        a = act(obs, False)
        if kit is not None and hasattr(act, "get_features"):
            feats.append(act.get_features())
            cols.append(int(env._pos[1]))
        obs, _, term, trunc, info = env.step(a)
        ev = None
        if a in (A_BUILD, A_MINE) and (info.get("placed") or info.get("mined")):
            dr, dc = FACE_DELTA[int(info["facing"])]
            ev = dict(kind="build" if a == A_BUILD else "mine",
                      r=int(env._pos[0] + dr), c=int(env._pos[1] + dc))
            if a == A_MINE:
                mines += 1
            else:
                builds += 1
        trace.append(dict(r=int(env._pos[0]), c=int(env._pos[1]),
                          facing=int(info["facing"]), ev=ev))
        if term or trunc:
            break
    act.set_logit_bias(None)
    steps = len(trace)
    return dict(agent=agent, method="logit", map_id=mid, seed=seed, cond=cond,
                knob=0.0 if cond == "baseline" else float(bias), steps=steps, door=_door(env, rec),
                success=bool(env._pos in (env._correct_cells or set())),
                timeout=bool(steps >= 799), mines=mines, builds=builds,
                proj=_wm_readback(kit, feats, cols, int(rec.wall_col)),
                trace=trace)


def _wm_readback(kit, feats, cols, wall):
    if kit is None or not feats:
        return None
    H = np.stack([np.asarray(f[kit["feat"]], np.float32) for f in feats])
    C = np.asarray(cols[:len(H)])
    crw = C - wall
    if kit["prewall"]:
        crossed = np.where(C >= wall)[0]
        stop = int(crossed[0]) if len(crossed) else len(H)
        H, C, crw = H[:stop], C[:stop], crw[:stop]
    if not len(H):
        return None
    m = (crw >= kit["win"][0]) & (crw < kit["win"][1])
    proj = H @ kit["v_bel"]
    return float(proj[m].mean()) if m.any() else float(proj[np.argmax(C)])


def dreamer_tilt_episode(D, rec, mid, seed, cond, lam, M, K):
    """Imagination tilt; cond -> tool argument of tilt_logits."""
    import torch
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS
    from tensordict import TensorDict
    tools = TOOLS_OF[cond]
    tool = "both" if len(tools) == 2 else (tools[0] if tools else "mine")
    use_lam = 0.0 if not tools else lam
    ag = D.agent
    np.random.seed(seed); torch.manual_seed(seed)
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset()
    state = ag.get_initial_state(1)
    first = True
    mines = builds = 0
    deters, cols, trace = [], [], []
    for t in range(FORKWALL_KWARGS["max_steps"]):
        stoch, deter = D.posterior(state, obs, first)
        lg, _ = D.tilt_logits(stoch, deter, tool, use_lam, M, K)
        a = int(torch.distributions.Categorical(logits=lg).sample())
        deters.append(deter.detach().cpu().numpy().reshape(-1).astype(np.float32))
        cols.append(int(env._pos[1]))
        oh = torch.zeros(1, D.A, device=deter.device); oh[0, a] = 1.0
        state = TensorDict({"stoch": stoch, "deter": deter, "prev_action": oh},
                           batch_size=(1,))
        first = False
        obs, _, term, trunc, info = env.step(a)
        ev = None
        if a in (A_BUILD, A_MINE) and (info.get("placed") or info.get("mined")):
            dr, dc = FACE_DELTA[int(info["facing"])]
            ev = dict(kind="build" if a == A_BUILD else "mine",
                      r=int(env._pos[0] + dr), c=int(env._pos[1] + dc))
            if a == A_MINE:
                mines += 1
            else:
                builds += 1
        trace.append(dict(r=int(env._pos[0]), c=int(env._pos[1]),
                          facing=int(info["facing"]), ev=ev))
        if term or trunc:
            break
    steps = len(trace)
    z = np.load(REPO / "outputs/belief_report/steer_axis_dreamer.npz")
    v = z["v"].astype(np.float32); v /= np.linalg.norm(v) + 1e-12
    D_ = np.array(deters); C = np.asarray(cols)
    crw = C - int(rec.wall_col)
    m = (crw >= -8) & (crw < 0)
    proj = D_ @ v
    return dict(agent="dreamer", method="tilt", map_id=mid, seed=seed,
                cond=cond, knob=float(use_lam), steps=steps,
                door=_door(env, rec),
                success=bool(env._pos in (env._correct_cells or set())),
                timeout=bool(steps >= 799), mines=mines, builds=builds,
                proj=float(proj[m].mean()) if m.any() else None, trace=trace)


def wm_kit(agent):
    rb = {"dreamer": dict(feat="deter", win=(-8, 0), prewall=True, key="v"),
          "storm": dict(feat="h", win=(0, 3), prewall=False, key="v_wall")}[agent]
    z = np.load(REPO / f"outputs/belief_report/steer_axis_{agent}.npz")
    v = z[rb["key"]].astype(np.float32)
    rb["v_bel"] = v / (np.linalg.norm(v) + 1e-12)
    return rb


# ── stages ───────────────────────────────────────────────────────────────

def summarise(rows, tag):
    f = lambda k: float(np.mean([r[k] for r in rows]))          # noqa: E731
    print(f"  {tag:34s} n={len(rows):4d} succ {f('success'):.2f} "
          f"TO {f('timeout'):.2f} mines {f('mines'):6.1f} "
          f"builds {f('builds'):6.1f} steps {f('steps'):5.0f}", flush=True)
    return dict(n=len(rows), success=f("success"), timeout=f("timeout"),
                mines=f("mines"), builds=f("builds"), steps=f("steps"))


def strip(rows):
    """Rows without traces, for the compact per-episode JSON."""
    return [{k: v for k, v in r.items() if k != "trace"} for r in rows]


def stage_fit(a):
    OUT.mkdir(parents=True, exist_ok=True)
    _, ids = balanced_ids("train", a.fit_maps, seed=11)
    print(f"fit arm={a.arm} on {len(ids)} TRAIN balanced maps "
          f"x {a.fit_rolls} rollouts", flush=True)
    rows, t0 = [], time.time()

    if a.arm == "ppo_clamp":
        jobs = []
        for thr in [float(x) for x in a.ladder.split(",")]:
            for cond in CONDS:
                if cond == "baseline" and thr != float(a.ladder.split(",")[0]):
                    continue                       # baseline is knob-free
                for mid in ids:
                    for k in range(a.fit_rolls):
                        jobs.append((mid, SEED0 + k, cond, thr, a.alpha,
                                     a.max_iters, "train"))
        rows = run_ppo(jobs, a.workers)
    else:
        rows = wm_fit_rows(a, ids)

    (OUT / f"fit_{a.arm}.json").write_text(json.dumps(strip(rows), indent=1))
    print(f"  wall-clock {time.time() - t0:.0f}s", flush=True)
    tab = {}
    for cond in CONDS:
        for knob in sorted({r["knob"] for r in rows if r["cond"] == cond}):
            sub = [r for r in rows if r["cond"] == cond and r["knob"] == knob]
            if sub:
                tab[f"{cond}|{knob:g}"] = summarise(sub, f"{cond} knob={knob:g}")
    (OUT / f"fit_{a.arm}_summary.json").write_text(json.dumps(tab, indent=1))


def wm_fit_rows(a, ids):
    pool = pickle.load(open(TRAIN_PKL, "rb"))
    ladder = [float(x) for x in a.ladder.split(",")]
    rows = []
    if a.arm in ("storm_logit", "dreamer_logit"):
        agent = a.arm.split("_")[0]
        import replay_episode as RE
        act, reset = RE._get_agent(agent, a.device)
        kit = wm_kit(agent)
        for bias in ladder:
            for cond in CONDS:
                if cond == "baseline" and bias != ladder[0]:
                    continue
                for mid in ids:
                    for k in range(a.fit_rolls):
                        rows.append(wm_logit_episode(
                            act, reset, pool[mid], mid, SEED0 + k, cond,
                            bias, agent, kit))
                kb = 0.0 if cond == "baseline" else float(bias)
                summarise([r for r in rows if r["cond"] == cond
                           and r["knob"] == kb], f"{cond} bias={bias:g}")
    else:                                            # dreamer_tilt
        from act3_wm import DreamerImagination
        from replay_episode import CKPT
        D = DreamerImagination(CKPT["dreamer"]["ckpt"], a.device,
                               CKPT["dreamer"]["size"])
        for lam in ladder:
            for cond in CONDS:
                if cond == "baseline" and lam != ladder[0]:
                    continue
                for mid in ids:
                    for k in range(a.fit_rolls):
                        rows.append(dreamer_tilt_episode(
                            D, pool[mid], mid, SEED0 + k, cond, lam, a.M, a.K))
                kb = 0.0 if cond == "baseline" else float(lam)
                summarise([r for r in rows if r["cond"] == cond
                           and r["knob"] == kb], f"{cond} lam={lam:g}")
    return rows


def stage_freeze(a):
    """Apply the pre-registered rule to every fit summary, write the frozen
    operating points. Strongest knob = largest tool reduction that keeps
    success >= 0.85 and timeout <= 0.10 on the FIT set."""
    ops = {}
    for f in sorted(OUT.glob("fit_*_summary.json")):
        arm = f.name[len("fit_"):-len("_summary.json")]
        tab = json.loads(f.read_text())
        base = tab.get("baseline|0") or next(
            (v for k, v in tab.items() if k.startswith("baseline|")), None)
        if base is None:
            continue
        ops[arm] = {"baseline_fit": base, "conds": {}}
        for cond in CONDS[1:]:
            cands = []
            for k, v in tab.items():
                c, knob = k.split("|")
                if c != cond:
                    continue
                # rank by the reduction in the TARGETED tool(s) only: total
                # tool use mixes in substitution, which would let a knob that
                # merely swaps one tool for the other look "strong"
                keys = [t + "s" for t in TOOLS_OF[cond]]
                tgt_b = sum(base[k] for k in keys)
                tgt_v = sum(v[k] for k in keys)
                tot_b = base["mines"] + base["builds"]
                tot_v = v["mines"] + v["builds"]
                cands.append((float(knob), v, (tgt_b - tgt_v) / max(tgt_b, 1e-9),
                              (tot_b - tot_v) / max(tot_b, 1e-9)))
            ok = [c for c in cands
                  if c[1]["success"] >= GUARD_SUCCESS
                  and c[1]["timeout"] <= GUARD_TIMEOUT]
            if ok:
                pick = max(ok, key=lambda c: c[2])
                why = "strongest knob inside the guard rails"
            else:
                pick = max(cands, key=lambda c: c[1]["success"])
                why = ("NO knob satisfied the guard rails on the fit set; "
                       "took the most conservative (highest fit success)")
            ops[arm]["conds"][cond] = dict(
                knob=pick[0], fit_success=pick[1]["success"],
                fit_timeout=pick[1]["timeout"], fit_target_cut=pick[2],
                fit_total_cut=pick[3], n_candidates=len(cands),
                n_inside_guards=len(ok), rule=why)
            print(f"{arm:14s} {cond:10s} knob={pick[0]:<8g} "
                  f"fit succ {pick[1]['success']:.2f} TO {pick[1]['timeout']:.2f} "
                  f"target cut {pick[2]:+.1%} total {pick[3]:+.1%}  [{why}]",
                  flush=True)
    ops["_rule"] = dict(guard_success=GUARD_SUCCESS, guard_timeout=GUARD_TIMEOUT,
                        note="fit on TRAIN balanced maps; frozen before test")
    (OUT / "operating_points.json").write_text(json.dumps(ops, indent=1))
    print("wrote operating_points.json")


def stage_test(a):
    ops = json.loads((OUT / "operating_points.json").read_text())[a.arm]
    _, ids = balanced_ids("test", a.test_maps, seed=7)
    print(f"test arm={a.arm} on {len(ids)} HELD-OUT balanced maps "
          f"x {a.test_rolls} rollouts; knobs {[(c, v['knob']) for c, v in ops['conds'].items()]}",
          flush=True)
    t0 = time.time()
    conds = ("baseline",) if a.null else CONDS
    off = a.seed_offset
    if a.arm == "ppo_clamp":
        jobs = []
        for cond in conds:
            thr = 0.0 if cond == "baseline" else ops["conds"][cond]["knob"]
            for mid in ids:
                for k in range(a.test_rolls):
                    jobs.append((mid, SEED0 + off + k, cond, thr, a.alpha,
                                 a.max_iters, "test", not a.no_orth))
        rows = run_ppo(jobs, a.workers)
    else:
        pool = pickle.load(open(TEST_PKL, "rb"))
        rows = []
        if a.arm in ("storm_logit", "dreamer_logit"):
            agent = a.arm.split("_")[0]
            import replay_episode as RE
            act, reset = RE._get_agent(agent, a.device)
            kit = wm_kit(agent)
            for cond in conds:
                bias = 0.0 if cond == "baseline" else ops["conds"][cond]["knob"]
                for mid in ids:
                    for k in range(a.test_rolls):
                        rows.append(wm_logit_episode(
                            act, reset, pool[mid], mid, SEED0 + off + k, cond,
                            bias, agent, kit))
                summarise([r for r in rows if r["cond"] == cond], f"TEST {cond}")
        else:
            from act3_wm import DreamerImagination
            from replay_episode import CKPT
            D = DreamerImagination(CKPT["dreamer"]["ckpt"], a.device,
                                   CKPT["dreamer"]["size"])
            for cond in conds:
                lam = 0.0 if cond == "baseline" else ops["conds"][cond]["knob"]
                for mid in ids:
                    for k in range(a.test_rolls):
                        rows.append(dreamer_tilt_episode(
                            D, pool[mid], mid, SEED0 + off + k, cond, lam,
                            a.M, a.K))
                summarise([r for r in rows if r["cond"] == cond], f"TEST {cond}")
    dt = time.time() - t0
    for cond in conds:
        summarise([r for r in rows if r["cond"] == cond], f"TEST {cond}")
    tag = f"test_{a.arm}_null" if a.null else f"test_{a.arm}"
    if getattr(a, "no_orth", False):
        tag = tag.replace(f"test_{a.arm}", f"test_{a.arm}_noorth")
    (OUT / f"{tag}.json").write_text(json.dumps(strip(rows), indent=1))
    # traces for later cherry-picking: first 6 maps, all conditions
    keep = set() if a.null else set(ids[:6])
    tr = {}
    for r in rows:
        if r["map_id"] in keep:
            tr.setdefault(f"{r['cond']}|{r['map_id']}", []).append(
                dict(steps=r["trace"], correct=r["success"], door=r["door"],
                     to=r["timeout"]))
    for k, v in tr.items():
        cond, mid = k.split("|")
        (OUT / f"trace_{a.arm}_{cond}_{mid}.json").write_text(
            json.dumps({"balanced": dict(map_id=int(mid), rollouts=v)}))
    meta = json.loads((OUT / "run_meta.json").read_text()) if (
        OUT / "run_meta.json").exists() else {}
    meta[a.arm] = dict(test_maps=ids, test_rolls=a.test_rolls,
                       wall_clock_s=round(dt, 1),
                       s_per_episode=round(dt / max(len(rows), 1), 2),
                       M=a.M, K=a.K, alpha=a.alpha, max_iters=a.max_iters)
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  wall-clock {dt:.0f}s ({dt / max(len(rows),1):.2f}s/episode), "
          f"{len(tr)} trace files", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fit", "freeze", "test"])
    ap.add_argument("--arm", default="ppo_clamp",
                    choices=["ppo_clamp", "storm_logit", "dreamer_logit",
                             "dreamer_tilt"])
    ap.add_argument("--ladder", default="0.1,0.05,0.02,0.01,0.005")
    ap.add_argument("--fit-maps", type=int, default=10)
    ap.add_argument("--fit-rolls", type=int, default=4)
    ap.add_argument("--test-maps", type=int, default=30)
    ap.add_argument("--test-rolls", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-iters", type=int, default=25)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="shift the rollout seeds; used to run a NULL baseline "
                         "(same condition, different seeds) so the door-shift "
                         "statistic has an empirical noise floor")
    ap.add_argument("--no-orth", action="store_true",
                    help="ABLATION: run the PPO clamp WITHOUT project_out")
    ap.add_argument("--null", action="store_true",
                    help="test stage: run ONLY the baseline condition, writing "
                         "test_<arm>_null.json")
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--K", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"fit": stage_fit, "freeze": stage_freeze, "test": stage_test}[a.stage](a)


if __name__ == "__main__":
    main()
