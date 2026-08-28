#!/usr/bin/env python3
"""Act three, part A -- horizon-probe plan steering on PPO (Bush et al. adapted).

Bush et al. (ICLR 2025) steer a Sokoban DRC by adding future-behaviour probe
vectors to its spatial state. This is the honest adaptation to our flat-state
POMDP agent: probe P(tool within the next n steps) from h, then intervene
  lin_add   h' = h -/+ eta * w_hat            (their Eq. 1 / CAA analogue)
  lin_set   move along w_hat until sigma(w.h'+b) = p_target   (coordinate-SET)
  mlp_grad  h' = h -/+ eta * grad_h logit_mlp(h)              (PPLM analogue)
each x {plain, orth} where orth removes the belief-axis component of the
APPLIED delta every step. The question: does the orth arm still steer the
behaviour (separable plan representation) or go inert (plan entangled with
belief)? Readback + door outcome decide.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts/mechinterp:scripts/mechinterp/belief_report:scripts/figures:scripts/mechinterp/behavior_steering \
    python scripts/mechinterp/behavior_steering/act3_ppo.py --stage probes|sham|pilot|grid|qual
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
HORIZONS = (5, 15, 30)


# ── labels ───────────────────────────────────────────────────────────────

def future_labels(df, tool, n):
    """y[i] = 1 iff any step t+1..t+n of the same episode has the tool action."""
    a = (df["action"].to_numpy() == TOOL_ACT[tool]).astype(np.int64)
    y = np.zeros(len(df), dtype=np.int8)
    for _, g in df.groupby("ep", sort=False):
        idx = g.index.to_numpy()
        ai = a[idx]
        c = np.concatenate([[0], np.cumsum(ai)])
        T = len(idx)
        for off in range(T):
            hi = min(off + n, T - 1)
            y[idx[off]] = 1 if (c[hi + 1] - c[off + 1]) > 0 else 0
    return y


def stage_probes():
    """Train horizon probes (linear + MLP) and the paper-style controls."""
    import data as D
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, df = D.load("ppo")
    X = np.asarray(X, np.float32)
    tr_ids, te_ids = D.split_maps(df)
    m_tr = df["map_id"].isin(tr_ids).to_numpy()
    m_te = df["map_id"].isin(te_ids).to_numpy()

    out = {"auc": {}, "base_rate": {}}
    probes = {}
    for tool in ("mine", "build"):
        for n in HORIZONS:
            y = future_labels(df, tool, n)
            lr = LogisticRegression(max_iter=2000, C=1.0)
            lr.fit(X[m_tr], y[m_tr])
            auc = roc_auc_score(y[m_te], lr.decision_function(X[m_te]))
            out["auc"][f"lin_{tool}_{n}"] = round(float(auc), 4)
            out["base_rate"][f"{tool}_{n}"] = round(float(y.mean()), 4)
            probes[f"w_{tool}_{n}"] = lr.coef_[0].astype(np.float32)
            probes[f"b_{tool}_{n}"] = np.float32(lr.intercept_[0])
            print(f"lin {tool} n={n:2d} AUC {auc:.4f} base {y.mean():.3f}",
                  flush=True)

    # primary horizon per tool: best test AUC
    prim = {t: max(HORIZONS, key=lambda n: out["auc"][f"lin_{t}_{n}"])
            for t in ("mine", "build")}
    out["primary_horizon"] = prim

    # MLP probes at the primary horizons
    torch.manual_seed(0)
    for tool in ("mine", "build"):
        n = prim[tool]
        y = future_labels(df, tool, n)
        net = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(),
                                  torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xt = torch.tensor(X[m_tr]); yt = torch.tensor(y[m_tr], dtype=torch.float32)
        lossf = torch.nn.BCEWithLogitsLoss()
        for ep in range(6):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 4096):
                b = perm[i:i + 4096]
                opt.zero_grad()
                loss = lossf(net(Xt[b]).squeeze(-1), yt[b])
                loss.backward(); opt.step()
        with torch.no_grad():
            z = net(torch.tensor(X[m_te])).squeeze(-1).numpy()
        auc = roc_auc_score(y[m_te], z)
        out["auc"][f"mlp_{tool}_{n}"] = round(float(auc), 4)
        torch.save(net.state_dict(), OUT / f"mlp_probe_{tool}.pt")
        print(f"mlp {tool} n={n:2d} AUC {auc:.4f}", flush=True)

    # random norm-matched probe control (their random-probe baseline)
    rng = np.random.default_rng(0)
    for tool in ("mine", "build"):
        n = prim[tool]
        w = probes[f"w_{tool}_{n}"]
        wr = rng.standard_normal(128).astype(np.float32)
        wr *= np.linalg.norm(w) / np.linalg.norm(wr)
        probes[f"wrand_{tool}"] = wr
        y = future_labels(df, tool, n)
        out["auc"][f"rand_{tool}"] = round(float(
            roc_auc_score(y[m_te], X[m_te] @ wr)), 4)

    # cos(probe axis, belief axis) -- the a-priori leak prediction
    for tool in ("mine", "build"):
        w = probes[f"w_{tool}_{prim[tool]}"]
        out[f"cos_w_belief_{tool}"] = round(float(
            (w / np.linalg.norm(w)) @ V_BEL), 4)

    np.savez(OUT / "probes.npz", **probes)
    (OUT / "probe_meta.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def stage_obs_probe():
    """Paper-style baseline: the same labels predicted from the RAW observation
    (flattened one-hot-free minimap int8 + scalars), on a map subsample."""
    import data as D
    import torch
    import replay_episode as RE
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS

    _, df = D.load("ppo")
    tr_ids, te_ids = D.split_maps(df)
    rng = np.random.default_rng(1)
    sub_tr = rng.choice(tr_ids, 150, replace=False)
    sub_te = rng.choice(te_ids, 75, replace=False)
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    act, reset = RE._get_agent("ppo", "cpu")
    meta = json.loads((OUT / "probe_meta.json").read_text())

    def collect(ids):
        F, A, EP = [], [], []
        for k, mid in enumerate(ids):
            rec = pool[int(mid)]
            np.random.seed(1000 + int(mid)); torch.manual_seed(1000 + int(mid))
            act.set_hook(None); act.set_logit_bias(None)
            env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
            obs, _ = env.reset(); reset()
            for t in range(FORKWALL_KWARGS["max_steps"]):
                F.append(np.concatenate([
                    np.asarray(obs["minimap"], np.float32).reshape(-1),
                    np.asarray(obs["scalars"], np.float32)]))
                a = act(obs, False)
                A.append(int(a)); EP.append(k)
                obs, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
        return np.array(F, np.float32), np.array(A), np.array(EP)

    res = {}
    for split, ids in (("train", sub_tr), ("test", sub_te)):
        res[split] = collect(ids)
        print(f"obs-probe {split}: {len(res[split][0])} rows", flush=True)
    import pandas as pd
    for tool in ("mine", "build"):
        n = meta["primary_horizon"][tool]
        aucs = {}
        Ftr, Atr, Etr = res["train"]; Fte, Ate, Ete = res["test"]
        def lab(A_, E_):
            d = pd.DataFrame({"ep": E_, "action": A_})
            return future_labels(d, tool, n)
        ytr, yte = lab(Atr, Etr), lab(Ate, Ete)
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(Ftr, ytr)
        auc = roc_auc_score(yte, lr.decision_function(Fte))
        meta["auc"][f"obs_{tool}_{n}"] = round(float(auc), 4)
        print(f"obs {tool} n={n} AUC {auc:.4f}", flush=True)
    meta["obs_probe_maps"] = dict(train=150, test=75)
    (OUT / "probe_meta.json").write_text(json.dumps(meta, indent=1))


# ── hooks ────────────────────────────────────────────────────────────────

def load_kit():
    z = np.load(OUT / "probes.npz")
    meta = json.loads((OUT / "probe_meta.json").read_text())
    kit = {"meta": meta}
    for tool in ("mine", "build"):
        n = meta["primary_horizon"][tool]
        w = z[f"w_{tool}_{n}"].astype(np.float32)
        kit[tool] = dict(w=w, b=float(z[f"b_{tool}_{n}"]),
                         w_hat=w / np.linalg.norm(w), wn=float(np.linalg.norm(w)),
                         wrand=z[f"wrand_{tool}"].astype(np.float32), n=n)
    return kit


def orth(d):
    return d - float(d @ V_BEL) * V_BEL


def make_hook(kind, tool, dose, kit, store, use_orth, mlp=None):
    """Returns hook(h, t, info) -> h'. Records per step into `store`:
    probe value c pre-edit, |delta|, cos(applied delta, belief axis)."""
    K = kit[tool]

    def apply(h, d):
        if use_orth:
            d = orth(d)
        n = float(np.linalg.norm(d))
        z = float(K["w"] @ h + K["b"])
        store.append(dict(c=1.0 / (1.0 + np.exp(-z)), dn=n,
                          cos=(float(d @ V_BEL) / n) if n > 1e-9 else 0.0))
        return (h + d).astype(np.float32)

    if kind == "lin_add":            # +dose incentivises, -dose suppresses
        def hook(h, t, info):
            return apply(h, dose * K["w_hat"])
    elif kind == "lin_set":          # one-sided move to p_target along w_hat
        p_t = dose                   # dose IS the target probability
        z_t = float(np.log(p_t / (1 - p_t)))
        cap = 3.0                    # per-step displacement cap, h units

        def hook(h, t, info):
            z = float(K["w"] @ h + K["b"])
            step = (z_t - z) / K["wn"]
            if p_t < 0.5:
                step = min(step, 0.0)        # suppress: only push down
            else:
                step = max(step, 0.0)
            step = float(np.clip(step, -cap, cap))
            return apply(h, step * K["w_hat"])
        hook.cap = cap
    elif kind == "mlp_grad":         # PPLM: gradient of the MLP logit
        import torch

        def hook(h, t, info):
            with torch.enable_grad():
                ht = torch.tensor(h, dtype=torch.float32, requires_grad=True)
                zt = mlp(ht[None]).squeeze()
                g, = torch.autograd.grad(zt, ht)
            return apply(h, dose * g.numpy().astype(np.float32))
    elif kind == "rand_dir":         # matched-norm fixed random direction
        rng = np.random.default_rng(int(dose * 1000) % 2**31)
        u = rng.standard_normal(128).astype(np.float32)
        u /= np.linalg.norm(u)

        def hook(h, t, info, _u=u):
            return apply(h, abs(dose) * _u)
    elif kind == "rand_probe":       # their random-probe control, lin_add form
        wr = K["wrand"] / np.linalg.norm(K["wrand"])

        def hook(h, t, info):
            return apply(h, dose * wr)
    else:
        raise ValueError(kind)
    return hook


def load_mlp(tool):
    import torch
    net = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 1))
    net.load_state_dict(torch.load(OUT / f"mlp_probe_{tool}.pt"))
    net.eval()
    return net


# ── episode wrapper (act2 readback + probe stats) ────────────────────────

def run_row(rec, mid, hook, store, cond, tool, cat):
    from act2_ppo import readback
    import ppo_campaign as PC
    r = PC.run_episode(rec, 1000 + mid, hook, None, want_steps=True, want_h=True)
    proj, reached = readback(r["hs"], r["trace"], int(rec.wall_col))
    to = r["steps"] >= 799
    row = dict(cond=cond, tool=tool, cat=cat, map_id=mid,
               mines=r["mines"], builds=r["builds"], steps=r["steps"],
               door=r["door"], success=r["success"], timeout=bool(to),
               wrong=bool((not r["success"]) and (not to)), proj=proj)
    if store:
        row.update(c_mean=float(np.mean([s["c"] for s in store])),
                   dn_mean=float(np.mean([s["dn"] for s in store])),
                   cos_mean=float(np.mean([s["cos"] for s in store])))
    return row


def maps_for(cat, n):
    return [int(x) for x in BEL[cat][:n]]


# ── stages ───────────────────────────────────────────────────────────────

def stage_sham():
    kit = load_kit()
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    import ppo_campaign as PC
    bad, rows = 0, []
    for cat, tool in (("rocky", "mine"), ("lakes", "build")):
        for mid in maps_for(cat, 5):
            base = PC.run_episode(pool[mid], 1000 + mid, want_steps=True)
            sham = PC.run_episode(pool[mid], 1000 + mid,
                                  make_hook("lin_add", tool, 0.0, kit, [], False),
                                  None, want_steps=True)
            same = (base["steps"] == sham["steps"] and base["door"] == sham["door"]
                    and [s["c"] for s in base["trace"]] ==
                        [s["c"] for s in sham["trace"]])
            bad += not same
            rows.append(dict(cat=cat, map_id=mid, match=bool(same),
                             steps=base["steps"], door=base["door"]))
            print(f"sham {cat} {mid:4d} {'MATCH' if same else 'DIFFERS'}", flush=True)
    (OUT / "sham_verify.json").write_text(json.dumps(
        {"ppo": dict(rows=rows, n=len(rows), match=sum(r["match"] for r in rows))},
        indent=1))
    print("SHAM", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)


def stage_pilot():
    kit = load_kit()
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    mlp = {t: load_mlp(t) for t in ("mine", "build")}
    rows = []
    for cat, tool in (("rocky", "mine"), ("lakes", "build")):
        ids = maps_for(cat, 5)
        arms = ([("lin_add", -e) for e in (0.5, 1.0, 2.0, 4.0)] +
                [("lin_set", p) for p in (0.05,)] +
                [("mlp_grad", -e) for e in (0.5, 2.0, 8.0, 32.0)])
        for kind, dose in arms:
            for o in (False, True):
                sub = []
                for mid in ids:
                    st = []
                    hk = make_hook(kind, tool, dose, kit, st, o,
                                   mlp=mlp[tool] if kind == "mlp_grad" else None)
                    sub.append(run_row(pool[mid], mid, hk, st,
                                       f"{kind}_{dose}{'_orth' if o else ''}",
                                       tool, cat))
                rows += sub
                s = lambda k: np.mean([r[k] for r in sub])
                print(f"{cat:6s} {kind:9s} d={dose:+7.2f} orth={int(o)} "
                      f"succ {s('success'):.2f} wrong {s('wrong'):.2f} "
                      f"TO {s('timeout'):.2f} mines {s('mines'):5.1f} "
                      f"builds {s('builds'):5.1f} c {s('c_mean'):.2f} "
                      f"|d| {s('dn_mean'):5.2f} cos {s('cos_mean'):+.2f} "
                      f"proj {s('proj'):+6.2f}", flush=True)
    (OUT / "pilot.json").write_text(json.dumps(rows, indent=1))


def stage_grid(doses_add, doses_mlp, n_maps):
    kit = load_kit()
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    mlp = {t: load_mlp(t) for t in ("mine", "build")}
    rows = []

    def log(sub, tag):
        s = lambda k: np.mean([r[k] for r in sub])
        print(f"{tag:34s} n={len(sub):3d} succ {s('success'):.2f} "
              f"WRONG {s('wrong'):.2f} TO {s('timeout'):.2f} "
              f"mines {s('mines'):5.1f} builds {s('builds'):5.1f} "
              f"c {s('c_mean') if 'c_mean' in sub[0] else float('nan'):.2f} "
              f"proj {s('proj'):+6.2f}", flush=True)

    for cat, tool in (("rocky", "mine"), ("lakes", "build")):
        ids = maps_for(cat, n_maps)
        base = [run_row(pool[m], m, None, [], "baseline", tool, cat) for m in ids]
        rows += base
        log(base, f"{cat} baseline")
        arms = []
        for e in doses_add:
            arms += [("lin_add", -e, False), ("lin_add", -e, True)]
        for p in (0.05, 0.005):
            arms += [("lin_set", p, False), ("lin_set", p, True)]
        for e in doses_mlp:
            arms += [("mlp_grad", -e, False), ("mlp_grad", -e, True)]
        # controls at the mid lin_add dose
        arms += [("rand_dir", -doses_add[1], False),
                 ("rand_probe", -doses_add[1], False)]
        # incentive, single dose each (rocky only, secondary)
        if cat == "rocky":
            arms += [("lin_add", +doses_add[1], False),
                     ("mlp_grad", +doses_mlp[1], False)]
        for kind, dose, o in arms:
            sub = []
            for mid in ids:
                st = []
                hk = make_hook(kind, tool, dose, kit, st, o,
                               mlp=mlp[tool] if kind == "mlp_grad" else None)
                sub.append(run_row(pool[mid], mid, hk, st,
                                   f"{kind}_{dose:+g}{'_orth' if o else ''}",
                                   tool, cat))
            rows += sub
            log(sub, f"{cat} {kind} {dose:+g}{' orth' if o else ''}")
    (OUT / "ppo_grid.json").write_text(json.dumps(rows, indent=1))
    print("wrote ppo_grid.json", len(rows), "rows")


def stage_qual(arms, n_roll=20):
    """Ghost-schema traces + per-step probe series on maps 626/77/99."""
    kit = load_kit()
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    mlp = {t: load_mlp(t) for t in ("mine", "build")}
    import ppo_campaign as PC
    from act2_ppo import readback
    for mid, cat, tool in ((626, "lakes", "build"), (77, "rocky", "mine"),
                           (99, "balanced", "mine")):
        for name, kind, dose, o in [("baseline", None, 0, False)] + arms:
            rolls, series = [], []
            for k in range(n_roll):
                st = []
                hk = (None if kind is None else
                      make_hook(kind, tool, dose, kit, st, o,
                                mlp=mlp[tool] if kind == "mlp_grad" else None))
                r = PC.run_episode(pool[mid], 2000 + k, hk, None,
                                   want_steps=True, want_h=True)
                proj, _ = readback(r["hs"], r["trace"], int(pool[mid].wall_col))
                to = r["steps"] >= 799
                rolls.append(dict(steps=r["trace"], correct=bool(r["success"]),
                                  door=r["door"], to=bool(to)))
                # probe series: recompute c(h_t) from stored h (works for
                # baseline too, where no hook ran)
                K = kit[tool]
                z = r["hs"] @ K["w"] + K["b"]
                series.append([round(float(v), 4)
                               for v in 1.0 / (1.0 + np.exp(-z))])
            path = OUT / f"qual_{name}_{mid}.json"
            path.write_text(json.dumps(
                {cat: dict(map_id=mid, rollouts=rolls)}))
            (OUT / f"qualseries_{name}_{mid}.json").write_text(json.dumps(series))
            nw = sum(1 for r in rolls if not r["correct"] and not r["to"])
            print(f"map {mid} {name:24s} succ "
                  f"{sum(r['correct'] for r in rolls)}/{n_roll} wrong {nw} "
                  f"TO {sum(r['to'] for r in rolls)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["probes", "obs_probe", "sham", "pilot", "grid", "qual"])
    ap.add_argument("--doses-add", default="0.5,1.0,2.0")
    ap.add_argument("--doses-mlp", default="2.0,8.0")
    ap.add_argument("--n-maps", type=int, default=25)
    ap.add_argument("--qual-arms", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "probes":
        stage_probes()
    elif a.stage == "obs_probe":
        stage_obs_probe()
    elif a.stage == "sham":
        stage_sham()
    elif a.stage == "pilot":
        stage_pilot()
    elif a.stage == "grid":
        stage_grid([float(x) for x in a.doses_add.split(",")],
                   [float(x) for x in a.doses_mlp.split(",")], a.n_maps)
    elif a.stage == "qual":
        arms = []
        for spec in filter(None, a.qual_arms.split(";")):
            name, kind, dose, o = spec.split(",")
            arms.append((name, kind, float(dose), o == "1"))
        stage_qual(arms)


if __name__ == "__main__":
    main()
