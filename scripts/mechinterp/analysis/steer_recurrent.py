#!/usr/bin/env python3
"""Can we steer from the recurrent hidden h? (follow-up to E4)

Persistent additive injection into the carried h fails (accumulates off-manifold,
reach→0). This tests recurrent schedules that avoid accumulation, all along the
actor decision direction, pre-commit only, on held-out balanced maps:

  recurrent persist   h ← h + αv every pre-commit step           (the failing ref)
  recurrent firstK    inject only for the first K pre-commit steps, then relax
  recurrent clamp     h ← h + αv then renormalise to the pre-injection ‖h‖
                      (rotate the state, stay on-manifold)
  actor (ref)         add to the actor input only (the working E4 method)

Measures commit control + reach + belief projection. Logs one W&B run.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .bundle import ActivationBundle
from . import geometry as G, style

CAT = ["rocky", "balanced", "lakes"]
A_BUILD, A_MINE = 4, 5


def _split(map_ids):
    u = np.unique(map_ids); rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(u, int(len(u) * 0.7), replace=False))
    return tr, np.setdiff1d(u, tr)


def _eval_eps(bundle, eval_maps, per_cat):
    m = bundle.maps; rng = np.random.default_rng(1)
    cats, seeds = m["category"], m["map_seed"]
    eps = []
    for c in CAT:
        mids = [i for i in range(len(cats)) if cats[i] == c and i in set(eval_maps)]
        for mid in rng.choice(mids, min(per_cat, len(mids)), replace=False) if mids else []:
            for tid in rng.choice(60, 2, replace=False):
                eps.append((mid, m["terrain"][mid], m["spawn"][mid],
                            int(seeds[mid]) * 100000 + int(tid), c))
    return eps


def rollout(pol, MiniEnv, terr, spawn, seed, vec, alpha, schedule, K, view, max_steps,
            belief_axis, device):
    import torch
    env = MiniEnv(terr, spawn, variant="btc", view_size=view, max_steps=max_steps)
    torch.manual_seed(seed)
    obs = env.reset()
    h = torch.zeros(1, 1, pol.gru_hidden, device=device)
    iv = None if (alpha == 0 or vec is None) else torch.from_numpy((alpha * vec).astype(np.float32)).view(1, 1, -1).to(device)
    proj = []; n_pre = 0
    for _ in range(max_steps):
        o = {k: torch.from_numpy(np.asarray(v)[None]).to(device) for k, v in obs.items()}
        feat = pol._encode(o).reshape(1, 1, -1)
        y, h = pol.gru(feat, h)
        precommit = env.commit == 0
        yin = y
        if iv is not None and precommit:
            if schedule == "actor":
                yin = y + iv
            elif schedule == "persist":
                h = h + iv
            elif schedule == "firstK":
                if n_pre < K:
                    h = h + iv
            elif schedule == "clamp":
                nrm = h.norm()
                h = h + iv
                h = h * (nrm / h.norm())
            yin = h if schedule != "actor" else yin
        if precommit:
            proj.append(float(h.squeeze().detach().cpu().numpy() @ belief_axis)); n_pre += 1
        logits = pol.actor(yin.squeeze(0))
        a = int(torch.distributions.Categorical(logits=logits).sample()[0])
        obs, reached, done = env.step(a)
        if done:
            break
    return env.commit, bool(reached), (np.mean(proj) if proj else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="activation_datasets/btc_ppo")
    ap.add_argument("--checkpoint", default="released_models/bridge_tunnel_commit/ppo_commit_onehot.pt")
    ap.add_argument("--per-cat", type=int, default=10)
    ap.add_argument("--alphas", type=float, nargs="*", default=[-8, -4, -2, 2, 4, 8])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--schedules", nargs="*", default=["actor", "firstK"],
                    help="which recurrent schedules to run (subset of actor/firstK/persist/clamp)")
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--out-dir", default="outputs/analysis_steer_rec")
    ap.add_argument("--wandb-project", default="bridge_tunnel_geometry")
    ap.add_argument("--wandb-mode", default="online")
    ap.add_argument("--run-name", default="steering-recurrent-btc")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import torch, wandb, pandas as pd
    style.apply_theme()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    b = ActivationBundle(a.dataset)
    sys.path.insert(0, str(b.path))
    from env_min import MiniBridgeTunnelEnv
    from policy_min import PPOGRUPolicy

    lab = b.labels
    tr_maps, ev_maps = _split(lab["map_id"].to_numpy())
    sub = lab[lab["map_id"].isin(tr_maps)].sample(
        min(a.rows, int((lab["map_id"].isin(tr_maps)).sum())), random_state=0)
    Xtr = b.load_activations("gru_h", sub["row_id"]); cat = sub["category"].to_numpy()
    belief_axis = G.unit(Xtr[cat == "lakes"].mean(0) - Xtr[cat == "rocky"].mean(0))

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    pol = PPOGRUPolicy.from_checkpoint(ckpt, b.view_size, b.manifest["n_scalars"], device=a.device)
    torch.set_grad_enabled(False)
    Wa = pol.actor.weight.detach().cpu().numpy()
    vec = G.unit(Wa[A_BUILD] - Wa[A_MINE])

    eps = _eval_eps(b, ev_maps, a.per_cat)
    print(f"{len(eps)} eval episodes on {len(ev_maps)} held-out maps")
    run = wandb.init(project=a.wandb_project, mode=a.wandb_mode, name=a.run_name,
                     tags=["btc", "ppo", "steering", "recurrent"],
                     config=dict(alphas=a.alphas, K=a.K, per_cat=a.per_cat, n_eps=len(eps)))

    schedules = a.schedules
    rows = []

    def cond(schedule, alpha):
        rec = [rollout(pol, MiniBridgeTunnelEnv, t[1], t[2], t[3], vec, alpha, schedule,
                       a.K, b.view_size, b.manifest["max_steps"], belief_axis, a.device) + (t[4],)
               for t in eps]
        df = pd.DataFrame(rec, columns=["commit", "reach", "proj", "cat"])
        bal = df[df.cat == "balanced"]
        r = dict(schedule=schedule, alpha=alpha, reach=float(df.reach.mean()),
                 p_build_bal=float((bal.commit == 1).mean()),
                 p_mine_bal=float((bal.commit == 2).mean()),
                 reach_bal=float(bal.reach.mean()),
                 belief_proj=float(np.nanmean(df.proj)))
        return r

    rows.append({**cond("actor", 0.0), "schedule": "baseline"})
    print(f"  baseline: reach={rows[0]['reach']:.2f} build|bal={rows[0]['p_build_bal']:.2f} "
          f"mine|bal={rows[0]['p_mine_bal']:.2f}")
    for s in schedules:
        for al in a.alphas:
            r = cond(s, al); rows.append(r)
            print(f"  {s:9s} a={al:+5.1f} reach={r['reach']:.2f} reach|bal={r['reach_bal']:.2f} "
                  f"build|bal={r['p_build_bal']:.2f} mine|bal={r['p_mine_bal']:.2f} "
                  f"belief_proj={r['belief_proj']:+.1f}")

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "steering_recurrent.csv", index=False)
    run.log({"recurrent/table": wandb.Table(dataframe=tab)})
    _fig(run, out, tab, rows[0])
    run.finish(); print("DONE")


def _fig(run, out, tab, base):
    import wandb
    scheds = [s for s in tab.schedule.unique() if s != "baseline"]
    colors = dict(zip(["actor", "persist", "firstK", "clamp"],
                      ["#2a9d4a", "#d1495b", "#f0892b", "#1f5fd0"]))
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for s in scheds:
        d = tab[tab.schedule == s].sort_values("alpha")
        ax[0].plot(d.alpha, d.p_build_bal - d.p_mine_bal, "-o", color=colors.get(s), label=s)
        ax[1].plot(d.alpha, d.reach_bal, "-o", color=colors.get(s), label=s)
    ax[0].axhline(base["p_build_bal"] - base["p_mine_bal"], ls="--", color="#888", label="baseline")
    ax[0].set_title("commit control (balanced):  P(build)−P(mine)")
    ax[1].axhline(base["reach_bal"], ls="--", color="#888", label="baseline")
    ax[1].set_title("reach (balanced) — does it still solve?"); ax[1].set_ylim(0, 1.03)
    for x in ax:
        x.set_xlabel("injection α"); x.grid(True, color=style.GRIDC)
        x.set_facecolor(style.PANEL); x.legend(fontsize=8)
    fig.suptitle("Steering from the recurrent h: schedules vs the actor-input reference",
                 fontweight="bold")
    fig.tight_layout()
    p = out / "recurrent__control_and_reach.png"
    fig.savefig(p, bbox_inches="tight", dpi=150); run.log({"recurrent/control_and_reach": wandb.Image(str(p))})
    plt.close(fig)


if __name__ == "__main__":
    main()
