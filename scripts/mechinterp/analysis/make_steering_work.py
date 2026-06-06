#!/usr/bin/env python3
"""Make skill steering actually work — redesigned intervention (E4).

The E3 failure (constant additive bias to the recurrent hidden, DoM direction)
broke navigation and never flipped the commit. Fixes here:

  * inject into the ACTOR INPUT (the GRU *output* y), NOT the recurrent hidden h
    -> no recurrent compounding, navigation stays intact;
  * steer along the actor's own decision direction W_actor[build]-W_actor[mine]
    -> the exact axis that moves the build-vs-mine logits;
  * inject only PRE-COMMIT (the commit is irreversible / event-driven);
  * evaluate per category — BALANCED maps (both water+rock present) are the real
    test, where the agent genuinely chooses.

Compares: actor-input vs recurrent injection site; actor-decision vs DoM
direction; + a belief-orthogonalised actor direction. Measures, per condition:
commit distribution, commit-flip rates, reach (does it still solve the env), and
a non-saturating belief projection. Logs one W&B run.

    python -m scripts.mechinterp.analysis.make_steering_work \
        --checkpoint released_models/bridge_tunnel_commit/ppo_commit_onehot.pt \
        --wandb-mode online
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .bundle import ActivationBundle
from . import geometry as G, style

CAT = ["rocky", "balanced", "lakes"]
A_BUILD, A_MINE = 4, 5


def _train_eval_maps(map_ids):
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
            for tid in rng.choice(60, 2, replace=False):       # 2 trajs per map
                eps.append((mid, m["terrain"][mid], m["spawn"][mid],
                            int(seeds[mid]) * 100000 + int(tid), c))
    return eps


def rollout(pol, MiniEnv, terr, spawn, seed, inj, site, view, max_steps, belief_axis, device):
    import torch
    env = MiniEnv(terr, spawn, variant="btc", view_size=view, max_steps=max_steps)
    torch.manual_seed(seed)
    obs = env.reset()
    h = torch.zeros(1, 1, pol.gru_hidden, device=device)
    proj = []
    iv = None if inj is None else torch.from_numpy(inj).view(1, 1, -1).to(device)
    for _ in range(max_steps):
        o = {k: torch.from_numpy(np.asarray(v)[None]).to(device) for k, v in obs.items()}
        feat = pol._encode(o).reshape(1, 1, -1)
        y, h = pol.gru(feat, h)
        precommit = env.commit == 0
        yin = y
        if iv is not None and precommit:                       # steer only pre-commit
            if site == "actor":
                yin = y + iv                                   # actor input only
            elif site == "recurrent":
                yin = y + iv; h = h + iv                       # contaminates state (E3 way)
        if precommit:
            proj.append(float(h.squeeze().detach().cpu().numpy() @ belief_axis))
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
    ap.add_argument("--per-cat", type=int, default=12, help="held-out maps per category (×2 trajs)")
    ap.add_argument("--alphas", type=float, nargs="*", default=[-6, -3, -1.5, 1.5, 3, 6])
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--out-dir", default="outputs/analysis_steer")
    ap.add_argument("--wandb-project", default="bridge_tunnel_geometry")
    ap.add_argument("--wandb-mode", default="online")
    ap.add_argument("--run-name", default="steering-v2-btc")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import torch, wandb, pandas as pd
    style.apply_theme()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    b = ActivationBundle(a.dataset)
    sys.path.insert(0, str(b.path))
    from env_min import MiniBridgeTunnelEnv
    from policy_min import PPOGRUPolicy

    # ---- directions in the gru_h / actor-input space (128) ----
    lab = b.labels
    tr_maps, ev_maps = _train_eval_maps(lab["map_id"].to_numpy())
    sub = lab[lab["map_id"].isin(tr_maps)].sample(min(a.rows, (lab["map_id"].isin(tr_maps)).sum()),
                                                  random_state=0)
    Xtr = b.load_activations("gru_h", sub["row_id"])
    cat = sub["category"].to_numpy(); sk = sub["final_commit"].to_numpy()
    cen = lambda lab_, c: Xtr[lab_ == c].mean(0)
    belief_axis = G.unit(cen(cat, "lakes") - cen(cat, "rocky"))     # for the readout projection
    dom_skill = G.unit(cen(sk, "build") - cen(sk, "mine"))

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    pol = PPOGRUPolicy.from_checkpoint(ckpt, b.view_size, b.manifest["n_scalars"], device=a.device)
    torch.set_grad_enabled(False)
    Wa = pol.actor.weight.detach().cpu().numpy()                    # (6,128)
    actor_skill = G.unit(Wa[A_BUILD] - Wa[A_MINE])                  # the decision direction
    # belief-preserving actor direction: remove belief axis component
    actor_skill_orth = G.unit(actor_skill - (actor_skill @ belief_axis) * belief_axis)

    print(f"cos(actor_skill, belief_axis)={actor_skill@belief_axis:+.3f}  "
          f"cos(actor_skill, dom_skill)={actor_skill@dom_skill:+.3f}")

    methods = [
        ("actor · decision", "actor", actor_skill),
        ("recurrent · decision", "recurrent", actor_skill),
        ("actor · DoM", "actor", dom_skill),
        ("actor · decision⊥belief", "actor", actor_skill_orth),
    ]
    eps = _eval_eps(b, ev_maps, a.per_cat)
    print(f"{len(eps)} eval episodes on {len(ev_maps)} held-out maps")

    run = wandb.init(project=a.wandb_project, mode=a.wandb_mode, name=a.run_name,
                     tags=["btc", "ppo", "steering", "v2"],
                     config=dict(alphas=a.alphas, per_cat=a.per_cat, n_eps=len(eps)))

    def run_condition(method, site, vec, alpha):
        inj = None if alpha == 0 else (alpha * vec).astype(np.float32)
        rec = []
        for (mid, terr, spawn, seed, c) in eps:
            commit, reached, proj = rollout(pol, MiniBridgeTunnelEnv, terr, spawn, seed,
                                            inj, site, b.view_size, b.manifest["max_steps"],
                                            belief_axis, a.device)
            rec.append((mid, c, seed, commit, reached, proj))
        return rec

    # baseline (per-episode, for flip rates)
    base = run_condition("baseline", "actor", actor_skill, 0.0)
    base_commit = {(r[0], r[2]): r[3] for r in base}
    rows = []

    def summarise(method, alpha, rec):
        df = pd.DataFrame(rec, columns=["mid", "cat", "seed", "commit", "reach", "proj"])
        row = dict(method=method, alpha=alpha,
                   reach=float(df["reach"].mean()),
                   p_build=float((df["commit"] == 1).mean()),
                   p_mine=float((df["commit"] == 2).mean()),
                   p_none=float((df["commit"] == 0).mean()),
                   skill_axis=float((df["commit"] == 1).mean() - (df["commit"] == 2).mean()),
                   belief_proj=float(np.nanmean(df["proj"])))
        for c in CAT:
            d = df[df.cat == c]
            row[f"p_build_{c}"] = float((d["commit"] == 1).mean())
            row[f"p_mine_{c}"] = float((d["commit"] == 2).mean())
            row[f"reach_{c}"] = float(d["reach"].mean())
        # flip rates vs baseline
        flips_to_build = flips_to_mine = nb = nm = 0
        for r in rec:
            bc = base_commit.get((r[0], r[2]))
            if bc == 2: nm += 1; flips_to_build += int(r[3] == 1)
            if bc == 1: nb += 1; flips_to_mine += int(r[3] == 2)
        row["flip_mine→build"] = flips_to_build / max(nm, 1)
        row["flip_build→mine"] = flips_to_mine / max(nb, 1)
        return row

    rows.append(summarise("baseline", 0.0, base))
    print(f"  baseline: reach={rows[0]['reach']:.2f} build={rows[0]['p_build']:.2f} "
          f"mine={rows[0]['p_mine']:.2f}")
    for method, site, vec in methods:
        for al in a.alphas:
            rec = run_condition(method, site, vec, al)
            r = summarise(method, al, rec); rows.append(r)
            print(f"  {method:24s} a={al:+5.1f} reach={r['reach']:.2f} "
                  f"build={r['p_build']:.2f} mine={r['p_mine']:.2f} "
                  f"build|bal={r['p_build_balanced']:.2f} mine|bal={r['p_mine_balanced']:.2f} "
                  f"belief_proj={r['belief_proj']:+.2f}")

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "steering_v2.csv", index=False)
    run.log({"steering/table": wandb.Table(dataframe=tab)})
    _figs(run, out, tab, rows[0])
    (out / "steering_v2_summary.json").write_text(tab.to_json(orient="records", indent=2))
    run.finish()
    print("\nDONE")


def _figs(run, out, tab, base):
    import wandb
    methods = [m for m in tab["method"].unique() if m != "baseline"]
    colors = dict(zip(methods, ["#2a9d4a", "#d1495b", "#f0892b", "#1f5fd0"]))

    # 1) does steering work: skill axis & reach vs alpha (all methods)
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for m in methods:
        s = tab[tab.method == m].sort_values("alpha")
        ax[0].plot(s["alpha"], s["skill_axis"], "-o", color=colors[m], label=m)
        ax[1].plot(s["alpha"], s["reach"], "-o", color=colors[m], label=m)
    ax[0].axhline(base["skill_axis"], ls="--", color="#888", label="baseline")
    ax[0].set_title("commit control:  P(build) − P(mine)"); ax[0].set_xlabel("injection α")
    ax[1].axhline(base["reach"], ls="--", color="#888", label="baseline reach")
    ax[1].set_title("does it still solve the env?  reach"); ax[1].set_xlabel("injection α")
    ax[1].set_ylim(0, 1.03)
    for a_ in ax:
        a_.grid(True, color=style.GRIDC); a_.set_facecolor(style.PANEL); a_.legend(fontsize=8)
    fig.tight_layout(); _save(run, out, "steering/control_and_reach", fig)

    # 2) balanced-map commit composition vs alpha for the primary method
    prim = "actor · decision"
    s = tab[tab.method == prim].sort_values("alpha")
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(s["alpha"], s["p_build_balanced"], "-o", color="#ffd000", label="P(build)")
    ax.plot(s["alpha"], s["p_mine_balanced"], "-o", color="#a800e6", label="P(mine)")
    ax.plot(s["alpha"], s["reach_balanced"], "--s", color="#2a9d4a", label="reach")
    ax.set_title(f"BALANCED maps — {prim}\n(steer toward build →  /  toward mine ←)")
    ax.set_xlabel("injection α"); ax.set_ylim(0, 1.03)
    ax.grid(True, color=style.GRIDC); ax.set_facecolor(style.PANEL); ax.legend()
    fig.tight_layout(); _save(run, out, "steering/balanced_commit_control", fig)

    # 3) selectivity: does belief move (recurrent vs actor site)?
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for m in methods:
        s = tab[tab.method == m].sort_values("alpha")
        ax.plot(s["alpha"], s["belief_proj"], "-o", color=colors[m], label=m)
    ax.axhline(base["belief_proj"], ls="--", color="#888", label="baseline")
    ax.set_title("belief projection (pre-commit) vs α\n(actor-site should stay flat; recurrent should drift)")
    ax.set_xlabel("injection α"); ax.set_ylabel("h · belief axis")
    ax.grid(True, color=style.GRIDC); ax.set_facecolor(style.PANEL); ax.legend(fontsize=8)
    fig.tight_layout(); _save(run, out, "steering/belief_projection", fig)


def _save(run, out, key, fig):
    import wandb
    p = out / (key.replace("/", "__") + ".png")
    fig.savefig(p, bbox_inches="tight", dpi=150)
    run.log({key: wandb.Image(str(p))})
    plt.close(fig)


if __name__ == "__main__":
    main()
