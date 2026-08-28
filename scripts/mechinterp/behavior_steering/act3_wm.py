#!/usr/bin/env python3
"""Act three, part B -- steering a world model through its own imagination.

Bush et al. need a probe because a model-free agent cannot be asked what it
will do next. A world model can: roll its latent dynamics forward under its own
actor and count the tool actions. So part B replaces the probe with the model's
own forecast, and replaces state editing with a re-weighting of the policy:

  B1 estimator   P_hat(tool within K) from M imagined rollouts, calibrated
                 against what the replayed episode actually did.
  B2 tilt        log pi'(a) = log pi(a) - lam * E[imagined tool events | a],
                 the expectation from A x M imagined rollouts in one batch.
                 No latent is ever edited, so the belief cannot be displaced by
                 the intervention -- the readback checks that.

At lam = 0 `tilt_logits` short-circuits to the actor's own logits and consumes
no imagination RNG, so the lam=0 arm reproduces the plain agent draw for draw
(archived in act3/dreamer_sham.json). For lam > 0 the imagined rollouts do
consume torch RNG, so steered arms match plain in distribution, not draw for
draw -- state that when comparing.

Two earlier designs were measured and discarded; do not resurrect them. An
unanchored MPC (argmax of imagined return - lam*tools) is a WORSE policy than
the actor at lam=0 (21.7 mines against 8.7, 182 steps against 102), and an
actor-log-prob anchor at beta=8 over-anchors so that every lam reproduces the
agent exactly. The `--beta` flag is a leftover of the second and is unused.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:r2dreamer_model:scripts/mechinterp:scripts/figures:scripts/mechinterp/behavior_steering \
    python scripts/mechinterp/behavior_steering/act3_wm.py --agent dreamer --stage calib|grid|qual|sham
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ("src", "scripts/mechinterp", "scripts/figures",
          "scripts/mechinterp/behavior_steering"):
    sys.path.insert(0, str(REPO / p))

OUT = REPO / "outputs/behavior_steering/act3"
A_BUILD, A_MINE = 4, 5
TOOL_ACT = {"mine": A_MINE, "build": A_BUILD}
VIEW, N_TILES = 21, 9
# facing id -> delta, mirroring env._FACE_DELTA (as collect_ghost_rollouts)
FACE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


# ── Dreamer: latent imagination ──────────────────────────────────────────

class DreamerImagination:
    """Drives a Dreamer episode by hand so we can imagine from the live
    posterior each step. Mirrors make_dreamer's encoding exactly; the plain
    arm reproduces the adapter's action rule (actor mode)."""

    def __init__(self, ckpt, device="cuda", size="size25M"):
        import torch
        from paper_rollouts import make_dreamer
        self.torch = torch
        act, _ = make_dreamer(ckpt, device, size, sampled=False)
        self.agent = act.agent
        self.device = device
        self.A = 6

    # -- encoding, identical to the adapter --
    def _trans(self, obs, first):
        import torch
        from tensordict import TensorDict
        oh = np.zeros((VIEW, VIEW, N_TILES), dtype=np.float32)
        rr, cc = np.indices((VIEW, VIEW))
        oh[rr, cc, np.asarray(obs["minimap"], dtype=np.int64)] = 1.0
        vec = np.concatenate([oh.reshape(-1),
                              np.asarray(obs["scalars"], dtype=np.float32)])
        return TensorDict({
            "vector": torch.as_tensor(vec, device=self.device,
                                      dtype=torch.float32)[None],
            "is_first": torch.tensor([first], device=self.device)},
            batch_size=(1,))

    def posterior(self, state, obs, first):
        """One observation step -> (stoch, deter) posterior at time t."""
        ag = self.agent
        trans = self._trans(obs, first)
        p = ag.preprocess(trans)
        embed = ag._frozen_encoder(p)
        stoch, deter, _ = ag._frozen_rssm.obs_step(
            state["stoch"], state["deter"], state["prev_action"], embed,
            trans["is_first"])
        return stoch, deter

    def _roll(self, stoch, deter, first_action=None, K=15):
        """Imagine K steps under the frozen actor. `first_action` forces t=0:
        an int forces one action on every row, an index tensor forces a
        different action per row (that is how all six candidates are scored in
        one batch). Returns discounted return and discounted tool counts."""
        import torch
        ag = self.agent
        B = deter.shape[0]
        gamma = 0.99
        ret = torch.zeros(B, device=deter.device)
        disc = torch.ones(B, device=deter.device)
        mine_n = torch.zeros(B, device=deter.device)
        build_n = torch.zeros(B, device=deter.device)
        for k in range(K):
            feat = ag._frozen_rssm.get_feat(stoch, deter)
            if k == 0 and first_action is not None:
                a = torch.zeros(B, self.A, device=deter.device)
                if torch.is_tensor(first_action):
                    a[torch.arange(B, device=deter.device), first_action] = 1.0
                else:
                    a[:, first_action] = 1.0
            else:
                a = ag._frozen_actor(feat).rsample()
            idx = a.argmax(-1)
            mine_n += disc * (idx == A_MINE).float()
            build_n += disc * (idx == A_BUILD).float()
            stoch, deter = ag._frozen_rssm.img_step(stoch, deter, a)
            nfeat = ag._frozen_rssm.get_feat(stoch, deter)
            r = ag._frozen_reward(nfeat).mode()
            c = ag._frozen_cont(nfeat).mean
            ret = ret + disc * r.reshape(B)
            disc = disc * gamma * c.reshape(B)
        v = ag._frozen_value(ag._frozen_rssm.get_feat(stoch, deter)).mode()
        ret = ret + disc * v.reshape(B)
        return ret, mine_n, build_n

    def _expand(self, stoch, deter, n):
        s = stoch.expand(n, *stoch.shape[1:]).contiguous()
        d = deter.expand(n, *deter.shape[1:]).contiguous()
        return s, d

    def forecast(self, stoch, deter, M=8, K=15):
        """B1: P_hat(tool within K) under the agent's own policy."""
        import torch
        with torch.no_grad():
            s, d = self._expand(stoch, deter, M)
            _, mn, bn = self._roll(s, d, None, K)
        return float((mn > 0).float().mean()), float((bn > 0).float().mean())

    def tilt_logits(self, stoch, deter, tool, lam, M=8, K=15):
        """The plan-level intervention, as a re-weighting of the agent's OWN
        policy by what it imagines each first action leads to:

            log pi'(a) = log pi(a) - lam * E[ imagined tool events | a ]

        The expectation comes from A x M imagined rollouts in one batch. No
        latent is ever edited, and at lam = 0 the logits ARE the actor's, so
        the lam=0 arm reproduces the plain agent draw for draw. Returns
        (tilted logits, mean imagined penalty per action)."""
        import torch
        with torch.no_grad():
            feat = self.agent._frozen_rssm.get_feat(stoch, deter)
            logp = torch.log_softmax(
                self.agent._frozen_actor(feat).logits.reshape(-1), -1)
            if lam == 0.0:
                return logp, torch.zeros_like(logp)
            st, dt = self._expand(stoch, deter, self.A * M)
            first = torch.arange(self.A, device=dt.device).repeat_interleave(M)
            _, mn, bn = self._roll(st, dt, first, K)
            pen = mn + bn if tool == "both" else (mn if tool == "mine" else bn)
            pen = pen.reshape(self.A, M).mean(1)
        return logp - lam * pen, pen


def run_dreamer_episode(D, rec, seed, mode, tool="mine", lam=0.0, M=8, K=15,
                        want_forecast=False, want_trace=False, beta=8.0):
    """mode: 'plain' (actor mode, the adapter's rule) | 'mpc'."""
    import torch
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS
    ag = D.agent
    np.random.seed(seed); torch.manual_seed(seed)
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset()
    state = ag.get_initial_state(1)
    first = True
    acts, deters, fc, trace = [], [], [], []
    for t in range(FORKWALL_KWARGS["max_steps"]):
        stoch, deter = D.posterior(state, obs, first)
        # ONE action rule for both arms: sample the (possibly tilted) actor
        # logits, so plain and mpc differ ONLY by the imagined-tool penalty
        lg, _ = D.tilt_logits(stoch, deter, tool,
                              lam if mode == "mpc" else 0.0, M, K)
        a = int(torch.distributions.Categorical(logits=lg).sample())
        if want_forecast:
            fc.append(D.forecast(stoch, deter, M, K))
        deters.append(deter.detach().cpu().numpy().reshape(-1).astype(np.float32))
        oh = torch.zeros(1, D.A, device=deter.device); oh[0, a] = 1.0
        from tensordict import TensorDict
        state = TensorDict({"stoch": stoch, "deter": deter, "prev_action": oh},
                           batch_size=(1,))
        first = False
        acts.append(int(a))
        obs, _, term, trunc, info = env.step(a)
        if want_trace:
            # tool events exactly as collect_ghost_rollouts records them: the
            # tool acts on the faced cell, and a tool action never turns the
            # agent, so info["facing"] is the facing it acted with
            ev = None
            if a in (A_BUILD, A_MINE) and (info.get("placed") or info.get("mined")):
                dr, dc = FACE_DELTA[int(info["facing"])]
                ev = dict(kind="build" if a == A_BUILD else "mine",
                          r=int(env._pos[0] + dr), c=int(env._pos[1] + dc))
            trace.append(dict(r=int(env._pos[0]), c=int(env._pos[1]),
                              facing=int(info["facing"]), ev=ev))
        if term or trunc:
            break
    fr = env._pos
    top = {p[0] for p in rec.top_goal_cells}
    bot = {p[0] for p in rec.bottom_goal_cells}
    to = len(acts) >= 799
    ok = env._pos in (env._correct_cells or set())
    return dict(steps=len(acts), actions=acts,
                mines=int(sum(a == A_MINE for a in acts)),
                builds=int(sum(a == A_BUILD for a in acts)),
                success=bool(ok), timeout=bool(to),
                wrong=bool((not ok) and (not to)),
                door="top" if fr[0] in top else "bottom" if fr[0] in bot else "none",
                deters=np.array(deters), forecast=fc, trace=trace)


# ── belief readback (same definition as act2) ────────────────────────────

def readback_deter(deters, cols, wall_col):
    z = np.load(REPO / "outputs/belief_report/steer_axis_dreamer.npz")
    v = z["v"].astype(np.float32); v /= np.linalg.norm(v) + 1e-12
    mu_l, mu_r = float(z["mu_lakes"]), float(z["mu_rocky"])
    rel = np.asarray(cols) - wall_col
    m = (rel >= -8) & (rel < 0)
    if not m.any():
        return float("nan"), 0.5 * (mu_l + mu_r)
    return float((deters[m] @ v).mean()), 0.5 * (mu_l + mu_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="dreamer", choices=["dreamer"])
    ap.add_argument("--stage", required=True,
                    choices=["calib", "grid", "qual", "sham"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-maps", type=int, default=25)
    ap.add_argument("--n-calib", type=int, default=12)
    ap.add_argument("--lams", default="0,10,30")
    ap.add_argument("--arms", default="mine:1,mine:3,both:3",
                    help="comma list of tool:lambda for the grid stage")
    ap.add_argument("--beta", type=float, default=8.0,
                    help="actor-anchor weight in the MPC score")
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--K", type=int, default=15)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from replay_episode import CKPT, replay
    D = DreamerImagination(CKPT["dreamer"]["ckpt"], a.device, CKPT["dreamer"]["size"])
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    BEL = np.load(REPO / "outputs/belief_report/steer_axis_dreamer.npz")
    maps = {c: [int(x) for x in BEL[c]] for c in ("rocky", "lakes", "balanced")}

    if a.stage == "sham":
        # lam=0 must reproduce the plain agent action for action
        rows, bad = [], 0
        for mid in maps["rocky"][:3]:
            base = run_dreamer_episode(D, pool[mid], 1000 + mid, "plain",
                                       M=a.M, K=a.K)
            sham = run_dreamer_episode(D, pool[mid], 1000 + mid, "mpc",
                                       "mine", 0.0, a.M, a.K)
            same = base["actions"] == sham["actions"]
            bad += not same
            rows.append(dict(map_id=mid, match=bool(same), steps=base["steps"],
                             mines=base["mines"], door=base["door"]))
            print(f"sham map {mid}: {'MATCH' if same else 'DIFFERS'} "
                  f"({base['steps']} vs {sham['steps']} steps)", flush=True)
        (OUT / "dreamer_sham.json").write_text(json.dumps(
            dict(dreamer=dict(rows=rows, n=len(rows),
                              match=sum(r["match"] for r in rows))), indent=1))
        print("SHAM", "PASS" if bad == 0 else f"FAIL ({bad})")
        sys.exit(1 if bad else 0)

    elif a.stage == "calib":
        from sklearn.metrics import roc_auc_score
        P, Y = {"mine": [], "build": []}, {"mine": [], "build": []}
        rows = []
        # mining happens on rocky maps, building on lakes maps: calibrate each
        # forecast where its event actually occurs, else the label is all-zero
        todo = ([("rocky", m) for m in maps["rocky"][:a.n_calib]] +
                [("lakes", m) for m in maps["lakes"][:a.n_calib]])
        for cat, mid in todo:
            r = run_dreamer_episode(D, pool[mid], 1000 + mid, "plain",
                                    M=a.M, K=a.K, want_forecast=True)
            acts = np.array(r["actions"])
            for tool in ("mine", "build"):
                aa = (acts == TOOL_ACT[tool]).astype(int)
                for t in range(len(acts)):
                    hi = min(t + 1 + a.K, len(acts))
                    Y[tool].append(int(aa[t + 1:hi].sum() > 0))
                    P[tool].append(r["forecast"][t][0 if tool == "mine" else 1])
            rows.append(dict(cat=cat, map_id=mid, steps=r["steps"],
                             mines=r["mines"], builds=r["builds"],
                             success=r["success"]))
            print(f"calib {cat:6s} map {mid:4d} steps {r['steps']:3d} "
                  f"mines {r['mines']:3d} builds {r['builds']:3d}", flush=True)
        out = dict(K=a.K, M=a.M, n_episodes=len(rows), rows=rows)
        for tool in ("mine", "build"):
            y, p = np.array(Y[tool]), np.array(P[tool])
            out[tool] = dict(n=int(len(y)), base_rate=round(float(y.mean()), 4),
                             auc=round(float(roc_auc_score(y, p)), 4)
                             if 0 < y.mean() < 1 else None,
                             corr=round(float(np.corrcoef(y, p)[0, 1]), 4)
                             if y.std() > 0 and p.std() > 0 else None)
            print(tool, out[tool], flush=True)
        (OUT / "dreamer_calib.json").write_text(json.dumps(out, indent=1))

    elif a.stage == "grid":
        # arms: "tool:lam" pairs, e.g. mine:1,mine:3,both:3
        arms = []
        for spec in a.arms.split(","):
            t, l = spec.split(":")
            arms.append((t, float(l)))
        rows = []
        cat = "rocky"
        for mid in maps[cat][:a.n_maps]:
            rec = pool[mid]
            # the plain agent, driven by MY loop, so its readback is comparable
            p = run_dreamer_episode(D, rec, 1000 + mid, "plain", M=a.M, K=a.K,
                                    want_trace=True)
            proj, mid_pt = readback_deter(p["deters"],
                                          [s["c"] for s in p["trace"]],
                                          int(rec.wall_col))
            rows.append(dict(cond="plain", cat=cat, map_id=mid, mines=p["mines"],
                             builds=p["builds"], steps=p["steps"], door=p["door"],
                             success=p["success"], timeout=p["timeout"],
                             wrong=p["wrong"], proj=proj, midpoint=mid_pt))
            for tool, lam in arms:
                r = run_dreamer_episode(D, rec, 1000 + mid, "mpc", tool, lam,
                                        a.M, a.K, want_trace=True, beta=a.beta)
                proj, mid_pt = readback_deter(r["deters"],
                                              [s["c"] for s in r["trace"]],
                                              int(rec.wall_col))
                rows.append(dict(cond=f"mpc_{tool}_lam{lam:g}", cat=cat,
                                 map_id=mid, mines=r["mines"],
                                 builds=r["builds"], steps=r["steps"],
                                 door=r["door"], success=r["success"],
                                 timeout=r["timeout"], wrong=r["wrong"],
                                 proj=proj, midpoint=mid_pt))
            print(f"map {mid:4d} done", flush=True)
            (OUT / "dreamer_mpc_grid.json").write_text(json.dumps(rows, indent=1))
        # summary
        import collections
        by = collections.defaultdict(list)
        for r in rows:
            by[r["cond"]].append(r)
        for cond, sub in sorted(by.items()):
            f = lambda k: np.mean([x[k] for x in sub])
            pr = [x["proj"] for x in sub if x.get("proj") is not None
                  and np.isfinite(x["proj"])]
            print(f"{cond:16s} n={len(sub):3d} succ {f('success'):.2f} "
                  f"WRONG {f('wrong'):.2f} TO {f('timeout'):.2f} "
                  f"mines {f('mines'):6.1f} builds {f('builds'):6.1f} "
                  f"proj {np.mean(pr) if pr else float('nan'):+6.2f}", flush=True)

    elif a.stage == "qual":
        lams = [float(x) for x in a.lams.split(",")]
        for mid, tool_q in ((77, "mine"), (99, "both")):
            rec = pool[mid]
            cat = {77: "rocky", 99: "balanced"}[mid]
            for lam in lams:
                rolls = []
                for k in range(20):
                    r = run_dreamer_episode(D, rec, 2000 + k, "mpc", tool_q, lam,
                                            a.M, a.K, want_trace=True,
                                            beta=a.beta)
                    rolls.append(dict(steps=r["trace"], correct=r["success"],
                                      door=r["door"], to=r["timeout"]))
                nm = sum(1 for x in rolls if not x["correct"] and not x["to"])
                (OUT / f"dreamer_qual_mpc{tool_q}{lam:g}_{mid}.json").write_text(
                    json.dumps({cat: dict(map_id=mid, rollouts=rolls)}))
                print(f"map {mid} lam {lam:g}: succ "
                      f"{sum(x['correct'] for x in rolls)}/20 wrong {nm} "
                      f"TO {sum(x['to'] for x in rolls)}", flush=True)


if __name__ == "__main__":
    main()
