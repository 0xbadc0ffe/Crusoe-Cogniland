#!/usr/bin/env python3
"""State-level clamping on a WORLD MODEL: the missing cell of the design.

Act four steers PPO at the state level (GradientClamp on the GRU state) and the
world models at the actuator level (logit bias) or the plan level (imagination
tilt). That confounds the intervention surface with the agent: "state edits move
the decision" cannot be separated from "PPO is the one being edited".

This closes it. Dreamer is steered the same way PPO is: push the CARRIED state
until the actor's probability of the tool action falls below a threshold, then
stop -- a minimal, closed-loop edit, not a fixed displacement.

    deter <- deter - alpha * d/d(deter) log pi(tool | stoch, deter)
    repeat while P(tool) > threshold, at most max_iters

The edited deter is carried forward, exactly as PPO's edited h is. Because the
clamp iterates to a probability target, this arm and the logit arm can be made
to satisfy the SAME behavioural constraint, so a difference in the door
statistic is attributable to the surface.

Dreamer's actor reads feat = concat(stoch, deter) computed AFTER the observation
update, so the adapter hook (which fires before) cannot see the probability it
must clamp; this uses the hand-driven loop from act three instead.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:r2dreamer_model:scripts/mechinterp:scripts/figures:scripts/mechinterp/behavior_steering \
    python scripts/mechinterp/behavior_steering/act4_wm_clamp.py --stage fit|test
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
for p in ("src", "scripts/mechinterp", "scripts/figures",
          "scripts/mechinterp/behavior_steering"):
    sys.path.insert(0, str(REPO / p))

OUT = REPO / "outputs/behavior_steering/act4"
A_BUILD, A_MINE = 4, 5
TOOLS_OF = {"baseline": (), "sup_mine": ("mine",), "sup_build": ("build",),
            "sup_both": ("mine", "build")}
IDX = {"mine": A_MINE, "build": A_BUILD}
FACE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
SEED0 = 2000


def dream_tool_cost(D, stoch, deter, tools, K=12, gamma=0.95):
    """Expected discounted tool use in the model's OWN dream, as a
    DIFFERENTIABLE function of the carried state.

    The imagination tilt (act three) computes this same forecast and uses it to
    re-weight the actor's logits. Here it is instead used as an objective on the
    state itself: the quantity is identical, the surface is not. Actions inside
    the dream are taken with the actor's straight-through sample so gradients
    flow through the whole rollout into `deter`, and the cost is the actor's
    PROBABILITY of the tool action at each imagined step (not a hard count),
    which keeps it differentiable."""
    import torch
    ag = D.agent
    idxs = [IDX[t] for t in tools]
    st, dt = stoch, deter
    cost = 0.0
    disc = 1.0
    for k in range(K):
        feat = ag._frozen_rssm.get_feat(st, dt)
        dist = ag._frozen_actor(feat)
        p = torch.softmax(dist.logits.reshape(-1), -1)
        cost = cost + disc * sum(p[i] for i in idxs)
        a = dist.rsample()                      # straight-through, keeps grad
        st, dt = ag._frozen_rssm.img_step(st, dt, a)
        disc = disc * gamma
    return cost


def clamp_deter_dream(D, stoch, deter, tools, thr, alpha=0.5, max_iters=8,
                      K=12):
    """Your fourth cell: plan ahead with the world model, then edit the
    ACTIVATIONS so the dream contains less tool use.

        deter <- deter - alpha * d/d(deter) E[dreamed tool use]

    stopping when the dreamed cost falls below `thr` (a cost budget over K
    imagined steps, so its scale differs from the single-step probability
    threshold used by the immediate clamp)."""
    import torch
    d = deter.detach().clone()
    if not tools:
        return deter, 0, 0.0
    for it in range(max_iters):
        d = d.detach().requires_grad_(True)
        cost = dream_tool_cost(D, stoch, d, tools, K)
        if float(cost) <= thr:
            return d.detach(), it, float(cost)
        g, = torch.autograd.grad(cost, d)
        n = torch.linalg.vector_norm(g)
        if float(n) < 1e-9:
            break
        d = (d - alpha * g / n).detach()
    with torch.no_grad():
        c = float(dream_tool_cost(D, stoch, d, tools, K))
    return d.detach(), max_iters, c


def clamp_deter(D, stoch, deter, tools, thr, alpha=0.5, max_iters=25):
    """Minimal edit: step deter down the actor's own gradient until every
    named tool action is below `thr`. Returns (deter', n_iters, p_after)."""
    import torch
    ag = D.agent
    d = deter.detach().clone()
    idxs = [IDX[t] for t in tools]
    if not idxs:
        return deter, 0, 0.0
    for it in range(max_iters):
        d = d.detach().requires_grad_(True)
        feat = ag._frozen_rssm.get_feat(stoch, d)
        logits = ag._frozen_actor(feat).logits.reshape(-1)
        p = torch.softmax(logits, -1)
        viol = [i for i in idxs if float(p[i]) > thr]
        if not viol:
            return d.detach(), it, float(max(p[i] for i in idxs))
        obj = torch.log(sum(p[i] for i in viol) + 1e-12)
        g, = torch.autograd.grad(obj, d)
        n = torch.linalg.vector_norm(g)
        if float(n) < 1e-9:
            break
        d = (d - alpha * g / n).detach()
    with torch.no_grad():
        feat = ag._frozen_rssm.get_feat(stoch, d)
        p = torch.softmax(ag._frozen_actor(feat).logits.reshape(-1), -1)
    return d.detach(), max_iters, float(max(p[i] for i in idxs))


def episode(D, rec, mid, seed, cond, thr, alpha, max_iters, v_bel, mid_pt,
            objective="immediate", K=12):
    import torch
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS
    from tensordict import TensorDict
    ag = D.agent
    tools = TOOLS_OF[cond]
    np.random.seed(seed); torch.manual_seed(seed)
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset()
    state = ag.get_initial_state(1)
    first = True
    mines = builds = 0
    trace, projs, cols, p_after = [], [], [], []
    for t in range(FORKWALL_KWARGS["max_steps"]):
        stoch, deter = D.posterior(state, obs, first)
        if tools and thr > 0:
            if objective == "dream":
                deter, _, pa = clamp_deter_dream(D, stoch, deter, tools, thr,
                                                 alpha, max_iters, K)
            else:
                deter, _, pa = clamp_deter(D, stoch, deter, tools, thr, alpha,
                                           max_iters)
            p_after.append(pa)
        with torch.no_grad():
            feat = ag._frozen_rssm.get_feat(stoch, deter)
            lg = ag._frozen_actor(feat).logits.reshape(-1)
            a = int(torch.distributions.Categorical(logits=lg).sample())
        projs.append(float(deter.detach().cpu().numpy().reshape(-1) @ v_bel))
        cols.append(int(env._pos[1]))
        oh = torch.zeros(1, 6, device=deter.device); oh[0, a] = 1.0
        state = TensorDict({"stoch": stoch, "deter": deter, "prev_action": oh},
                           batch_size=(1,))
        first = False
        obs, _, term, trunc, info = env.step(a)
        ev = None
        if a in (A_BUILD, A_MINE) and (info.get("placed") or info.get("mined")):
            dr, dc = FACE_DELTA[int(info["facing"])]
            ev = dict(kind="build" if a == A_BUILD else "mine",
                      r=int(env._pos[0] + dr), c=int(env._pos[1] + dc))
            mines += a == A_MINE; builds += a == A_BUILD
        trace.append(dict(r=int(env._pos[0]), c=int(env._pos[1]),
                          facing=int(info["facing"]), ev=ev))
        if term or trunc:
            break
    steps = len(trace)
    to = steps >= 799
    ok = env._pos in (env._correct_cells or set())
    top = {p[0] for p in rec.top_goal_cells}
    bot = {p[0] for p in rec.bottom_goal_cells}
    rel = np.asarray(cols) - int(rec.wall_col)
    m = (rel >= -8) & (rel < 0)
    return dict(cond=cond, map_id=mid, mines=int(mines), builds=int(builds),
                true_mines=int(mines), true_builds=int(builds), steps=steps,
                success=bool(ok), timeout=bool(to),
                wrong=bool((not ok) and (not to)),
                door=("top" if env._pos[0] in top else
                      "bottom" if env._pos[0] in bot else "none"),
                proj=float(np.mean(np.array(projs)[m])) if m.any() else None,
                midpoint=mid_pt,
                p_after=float(np.mean(p_after)) if p_after else None,
                trace=trace)


def balanced_ids(split, n, seed=0):
    pkl = REPO / f"data/bridge_tunnel/forkwall6k/{split}.pkl"
    pool = pickle.load(open(pkl, "rb"))
    ids = [i for i, r in enumerate(pool) if r.category == "balanced"]
    rng = np.random.default_rng(seed)
    return pool, [int(x) for x in rng.permutation(ids)[:n]]


def summarise(rows, tag):
    f = lambda k: np.mean([r[k] for r in rows])            # noqa: E731
    pa = [r["p_after"] for r in rows if r.get("p_after") is not None]
    print(f"  {tag:28s} n={len(rows):3d} succ {f('success'):.2f} "
          f"TO {f('timeout'):.2f} mines {f('true_mines'):5.1f} "
          f"builds {f('true_builds'):5.1f} steps {f('steps'):5.0f}"
          + (f" p(tool)->{np.mean(pa):.4f}" if pa else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fit", "test"])
    ap.add_argument("--ladder", default="0.05,0.02,0.01")
    ap.add_argument("--fit-maps", type=int, default=8)
    ap.add_argument("--fit-rolls", type=int, default=4)
    ap.add_argument("--test-maps", type=int, default=20)
    ap.add_argument("--test-rolls", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-iters", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--objective", default="immediate",
                    choices=["immediate", "dream"],
                    help="what the state edit minimises: the current action's "
                         "probability, or the tool use the model foresees")
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--seed-offset", type=int, default=0)
    a = ap.parse_args()

    from act3_wm import DreamerImagination
    from replay_episode import CKPT
    D = DreamerImagination(CKPT["dreamer"]["ckpt"], a.device,
                           CKPT["dreamer"]["size"])
    z = np.load(REPO / "outputs/belief_report/steer_axis_dreamer.npz")
    v = z["v"].astype(np.float32); v /= np.linalg.norm(v) + 1e-12
    mid_pt = 0.5 * (float(z["mu_lakes"]) + float(z["mu_rocky"]))
    t0 = time.time()

    if a.stage == "fit":
        pool, ids = balanced_ids("train", a.fit_maps, seed=3)
        thrs = [float(x) for x in a.ladder.split(",")]
        rows = []
        base = [episode(D, pool[m], m, SEED0 + k, "baseline", 0.0, a.alpha,
                        a.max_iters, v, mid_pt, a.objective, a.K)
                for m in ids for k in range(a.fit_rolls)]
        rows += base
        summarise(base, "FIT baseline")
        for cond in ("sup_mine", "sup_build", "sup_both"):
            for thr in thrs:
                sub = [episode(D, pool[m], m, SEED0 + k, cond, thr, a.alpha,
                               a.max_iters, v, mid_pt, a.objective, a.K)
                       for m in ids for k in range(a.fit_rolls)]
                for r in sub:
                    r["thr"] = thr
                rows += sub
                summarise(sub, f"FIT {cond} thr={thr:g}")
        (OUT / f"fit_dreamer_clamp_{a.objective}.json").write_text(json.dumps(
            [{k: x for k, x in r.items() if k != "trace"} for r in rows],
            indent=1))
        # frozen operating point: strongest threshold inside the guard rails
        ops = {}
        for cond in ("sup_mine", "sup_build", "sup_both"):
            cands = []
            for thr in thrs:
                sub = [r for r in rows if r["cond"] == cond and r.get("thr") == thr]
                s, to = np.mean([r["success"] for r in sub]), np.mean(
                    [r["timeout"] for r in sub])
                if s >= 0.85 and to <= 0.10:
                    cands.append((thr, s, to))
            pick = min(cands)[0] if cands else min(thrs)
            ops[cond] = dict(knob=pick, n_inside_guards=len(cands),
                             rule="strongest threshold inside the guard rails"
                             if cands else "ladder boundary, no knob qualified")
            print(f"  frozen {cond}: thr={pick:g} "
                  f"({len(cands)}/{len(thrs)} inside guards)")
        p = OUT / f"operating_points_dreamer_clamp_{a.objective}.json"
        p.write_text(json.dumps(ops, indent=1))
        print("wrote", p.name)
    else:
        ops = json.loads(
            (OUT / f"operating_points_dreamer_clamp_{a.objective}.json").read_text())
        pool, ids = balanced_ids("test", a.test_maps, seed=7)
        off = a.seed_offset
        conds = ("baseline",) if a.null else (
            "baseline", "sup_mine", "sup_build", "sup_both")
        rows = []
        for cond in conds:
            thr = 0.0 if cond == "baseline" else ops[cond]["knob"]
            sub = [episode(D, pool[m], m, SEED0 + off + k, cond, thr, a.alpha,
                           a.max_iters, v, mid_pt, a.objective, a.K)
                   for m in ids for k in range(a.test_rolls)]
            rows += sub
            summarise(sub, f"TEST {cond}")
        tag = f"test_dreamer_clamp_{a.objective}_null" if a.null else f"test_dreamer_clamp_{a.objective}"
        (OUT / f"{tag}.json").write_text(json.dumps(
            [{k: x for k, x in r.items() if k != "trace"} for r in rows],
            indent=1))
        if not a.null:
            keep = set(ids[:6])
            tr = {}
            for r in rows:
                if r["map_id"] in keep:
                    tr.setdefault(f"{r['cond']}|{r['map_id']}", []).append(
                        dict(steps=r["trace"], correct=r["success"],
                             door=r["door"], to=r["timeout"]))
            for k, vv in tr.items():
                cond, mid = k.split("|")
                (OUT / f"trace_dreamer_clamp_{a.objective}_{cond}_{mid}.json").write_text(
                    json.dumps({"balanced": dict(map_id=int(mid), rollouts=vv)}))
        print(f"wrote {tag}.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
