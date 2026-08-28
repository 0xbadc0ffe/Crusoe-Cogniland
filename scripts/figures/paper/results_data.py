#!/usr/bin/env python3
"""Gather every number the Results chapter needs into one JSON.

Runs the held-out evaluation for the three released agents (belief-free PPO,
DreamerV3, STORM) and for every PPO seed in the sweep (recurrent belief-free and
feed-forward), all on the same 1200-map held-out pool with the TRUE door metric.
Feed-forward and recurrent seeds give the spread for the recurrence figure.

  crusoe:     PYTHONPATH=src python scripts/figures/paper/results_data.py --arm ppo
  r2dreamer:  PYTHONPATH=src:r2dreamer_model ...            --arm dreamer
  storm venv: PYTHONPATH=.:..:../src ...                     --arm storm
  crusoe:     ...                                            --arm ppo-seeds

Merged into paper/figures/forkwall_paper/results.json.
"""
from __future__ import annotations
import argparse, glob, json, math, os, pickle, sys, collections
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts" / "figures"))
CATS = ("balanced", "lakes", "rocky")


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0., c-h), min(1., c+h))


def eval_checkpoint_ppo(ckpt, pool, ids):
    import torch
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from cogniland.bridge_tunnel.policy import PPOGRUPolicy
    from paper_rollouts import FORKWALL_KWARGS
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    probe = BridgeTunnelEnv(seed=0, **FORKWALL_KWARGS); probe.reset()
    pol = PPOGRUPolicy.from_checkpoint(ck, probe.observation_space); pol.eval()
    rec = pol.gru is not None
    np.random.seed(0); torch.manual_seed(0)
    c = collections.Counter(); n = collections.Counter()
    wrong = to = 0; lens = []
    for i in ids:
        r = pool[i]; e = BridgeTunnelEnv(seed=0, map_record=r, **FORKWALL_KWARGS)
        obs, _ = e.reset(); h = torch.zeros(1, 1, 128)
        for t in range(FORKWALL_KWARGS["max_steps"]):
            tt = {k: torch.as_tensor(np.asarray(v))[None] for k, v in obs.items()}
            with torch.no_grad():
                a, _, _, _, h = pol.get_action_and_value(tt, h, torch.zeros(1))
            obs, _, term, trunc, _ = e.step(int(a))
            if term or trunc: break
        n[r.category] += 1; lens.append(t + 1)
        if e._pos in (e._correct_cells or set()): c[r.category] += 1
        elif e._step_count < e.max_steps: wrong += 1
        else: to += 1
    N = sum(n.values()); ok = sum(c.values())
    dn = n["lakes"] + n["rocky"]; dok = c["lakes"] + c["rocky"]
    lo, hi = wilson(ok, N); dlo, dhi = wilson(dok, dn)
    return dict(recurrent=rec, episodes=N, success=ok/N, ci=[lo, hi],
                decisive=dok/dn, decisive_ci=[dlo, dhi],
                wrong=wrong/N, timeout=to/N, mean_len=float(np.mean(lens)),
                per_cat={k: c[k]/n[k] for k in n})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["ppo", "dreamer", "storm", "ppo-seeds"])
    ap.add_argument("--out", default=str(REPO / "paper/figures/forkwall_paper/results.json"))
    ap.add_argument("--maps", default=str(REPO / "data/bridge_tunnel/forkwall6k/test.pkl"))
    a = ap.parse_args()
    pool = pickle.load(open(a.maps, "rb"))
    ids = list(range(len(pool)))
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(outp.read_text()) if outp.exists() else {}

    if a.arm == "ppo":
        res["ppo"] = eval_checkpoint_ppo(str(REPO / "final_models/ppo/ppo_plain_noaux.pt"), pool, ids)
        print("ppo:", res["ppo"]["success"], res["ppo"]["decisive"])
    elif a.arm == "ppo-seeds":
        seeds = {"recurrent": [], "feedforward": []}
        for d in sorted(glob.glob(str(REPO / "outputs/ppo_noaux/*/final.pt"))):
            name = os.path.basename(os.path.dirname(d))
            r = eval_checkpoint_ppo(d, pool, ids)
            key = "recurrent" if r["recurrent"] else "feedforward"
            r["name"] = name
            seeds[key].append(r)
            print(f"  {name:20s} {'REC' if r['recurrent'] else 'FF':3s} "
                  f"succ={r['success']*100:.2f} dec={r['decisive']*100:.2f}")
        res["ppo_seeds"] = seeds
    elif a.arm == "dreamer":
        from paper_rollouts import make_dreamer, FORKWALL_KWARGS
        res["dreamer"] = _eval_generic(make_dreamer(str(REPO/"final_models/dreamer/dreamer_25M_bl64.pt"),
                        "cuda", "size25M", sampled=True), pool, ids)
        print("dreamer:", res["dreamer"]["success"])
    elif a.arm == "storm":
        from paper_rollouts import make_storm
        res["storm"] = _eval_generic(make_storm(str(REPO/"final_models/storm"), 624489, sampled=True),
                                     pool, ids)
        print("storm:", res["storm"]["success"])
    outp.write_text(json.dumps(res, indent=1)); print("merged ->", outp)


def _eval_generic(actreset, pool, ids):
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS
    act, reset = actreset
    np.random.seed(0)
    try:
        import torch; torch.manual_seed(0)
    except Exception: pass
    c = collections.Counter(); n = collections.Counter(); wrong = to = 0; lens = []
    for i in ids:
        r = pool[i]; e = BridgeTunnelEnv(seed=0, map_record=r, **FORKWALL_KWARGS)
        obs, _ = e.reset(); reset()
        if hasattr(act, "set_seed"): act.set_seed(0)
        for t in range(FORKWALL_KWARGS["max_steps"]):
            obs, _, term, trunc, _ = e.step(act(obs, False))
            if term or trunc: break
        n[r.category] += 1; lens.append(t + 1)
        if e._pos in (e._correct_cells or set()): c[r.category] += 1
        elif e._step_count < e.max_steps: wrong += 1
        else: to += 1
    N = sum(n.values()); ok = sum(c.values())
    dn = n["lakes"] + n["rocky"]; dok = c["lakes"] + c["rocky"]
    lo, hi = wilson(ok, N); dlo, dhi = wilson(dok, dn)
    return dict(recurrent=True, episodes=N, success=ok/N, ci=[lo, hi],
                decisive=dok/dn, decisive_ci=[dlo, dhi], wrong=wrong/N, timeout=to/N,
                mean_len=float(np.mean(lens)), per_cat={k: c[k]/n[k] for k in n})


if __name__ == "__main__":
    main()
