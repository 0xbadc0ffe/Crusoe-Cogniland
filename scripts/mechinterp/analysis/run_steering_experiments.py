#!/usr/bin/env python3
"""Three experiments testing belief↔skill separability in the BTC PPO+GRU agent.

  E1  subspace test   — does skill survive removing the belief subspace (and vice
                        versa)? + principal angles between the two subspaces.
  E2  off-type test   — does a build-vs-mine probe generalise to the rare
                        against-type commits (build-on-rocky / mine-on-lakes)?
                        Compared on full vs belief-removed features.
  E3  causal steering — inject 3 candidate skill directions into gru_h during
                        rollouts and measure Δskill (commit flips) vs Δbelief
                        (decoded). The belief-preserving direction should move
                        skill at low belief drift.

Everything operates in the gru_h(128) activation space. Probes / directions are
fit on TRAIN maps; E2 evaluates and E3 rolls out on HELD-OUT maps. Logs one W&B
run. Usage:

    python -m scripts.mechinterp.analysis.run_steering_experiments \
        --dataset activation_datasets/btc_ppo \
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from .bundle import ActivationBundle
from . import geometry as G, style

SKILL = ["none", "build", "mine"]
CAT = ["rocky", "balanced", "lakes"]
DIR_COLORS = {"global DoM": "#d1495b", "balanced DoM": "#1f5fd0", "belief-orth": "#2a9d4a"}


# ----------------------------------------------------------------- helpers
def onb(vectors):
    """Orthonormal basis (128xk) spanning the given vectors."""
    M = np.stack([G.unit(v) for v in vectors]).T
    Q, _ = np.linalg.qr(M)
    return Q


def remove_subspace(X, Q):
    """Project the columns of basis Q out of rows of X."""
    return X - (X @ Q) @ Q.T


def fit_probe(X, y, groups, seed=0, balanced=True):
    tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed).split(X, groups=groups))
    pipe = Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000,
                                              class_weight="balanced" if balanced else None))])
    pipe.fit(X[tr], y[tr])
    return pipe, tr, te


def acc(pipe, X, y):
    return float((pipe.predict(X) == y).mean())


# ----------------------------------------------------------------- E1
def exp1_subspace(run, X, cat, skill, groups, outdir):
    print("\n=== E1: subspace separability ===")
    cen = lambda lab, c: X[lab == c].mean(0)
    belief_dirs = [cen(cat, "lakes") - cen(cat, "rocky"),
                   cen(cat, "balanced") - 0.5 * (cen(cat, "lakes") + cen(cat, "rocky"))]
    skill_dirs = [cen(skill, "build") - cen(skill, "mine"),
                  0.5 * (cen(skill, "build") + cen(skill, "mine")) - cen(skill, "none")]
    Bb, Bs = onb(belief_dirs), onb(skill_dirs)
    ang = G.principal_angles(belief_dirs, skill_dirs)

    Xnb = remove_subspace(X, Bb)   # belief removed
    Xns = remove_subspace(X, Bs)   # skill removed
    out = {}
    # skill decodability: full vs belief-removed
    p, tr, te = fit_probe(X, skill, groups)
    out["skill_acc_full"] = acc(p, X[te], skill[te])
    p2, _, te2 = fit_probe(Xnb, skill, groups)
    out["skill_acc_belief_removed"] = acc(p2, Xnb[te2], skill[te2])
    # belief decodability: full vs skill-removed
    pb, _, teb = fit_probe(X, cat, groups)
    out["belief_acc_full"] = acc(pb, X[teb], cat[teb])
    pb2, _, teb2 = fit_probe(Xns, cat, groups)
    out["belief_acc_skill_removed"] = acc(pb2, Xns[teb2], cat[teb2])
    out["principal_angles_deg"] = ang.round(1).tolist()
    out["min_principal_angle_deg"] = float(ang.min())
    print(json.dumps(out, indent=2))
    run.summary.update({f"E1/{k}": v for k, v in out.items() if np.isscalar(v)})

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = ["skill\n(full)", "skill\n(belief removed)", "belief\n(full)", "belief\n(skill removed)"]
    vals = [out["skill_acc_full"], out["skill_acc_belief_removed"],
            out["belief_acc_full"], out["belief_acc_skill_removed"]]
    cols = ["#1f5fd0", "#7fb0e6", "#d1495b", "#e6a0aa"]
    ax.bar(labels, vals, color=cols, edgecolor="white")
    ax.axhline(1/3, ls="--", color="#888", label="chance (3-class)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("held-out accuracy")
    ax.set_title(f"E1 decodability after removing the other subspace\n"
                 f"min principal angle = {out['min_principal_angle_deg']:.0f}°")
    ax.legend(); fig.tight_layout()
    _logfig(run, outdir, "E1/decodability_after_removal", fig)
    return out


# ----------------------------------------------------------------- E2
def exp2_offtype(run, X, df, groups, outdir):
    print("\n=== E2: off-type generalisation ===")
    bm = np.isin(df["final_commit"].to_numpy(), ["build", "mine"])
    Xb = X[bm]; y = (df["final_commit"].to_numpy()[bm] == "build").astype(int)
    cat = df["category"].to_numpy()[bm]
    g = groups[bm]
    # on-type = build&lakes or mine&rocky ; off-type = build&rocky or mine&lakes
    build = y == 1
    ontype = (build & (cat == "lakes")) | (~build & (cat == "rocky"))
    offtype = (build & (cat == "rocky")) | (~build & (cat == "lakes"))

    # belief subspace from THIS subset
    cen = lambda c: Xb[cat == c].mean(0)
    Bb = onb([cen("lakes") - cen("rocky"),
              cen("balanced") - 0.5 * (cen("lakes") + cen("rocky"))])
    Xb_nb = remove_subspace(Xb, Bb)

    out = {}
    for name, feats in [("full", Xb), ("belief_removed", Xb_nb)]:
        p, tr, te = fit_probe(feats, y, g, balanced=True)
        te_on = te[ontype[te]]; te_off = te[offtype[te]]
        out[f"{name}_acc_ontype"] = acc(p, feats[te_on], y[te_on])
        out[f"{name}_acc_offtype"] = acc(p, feats[te_off], y[te_off])
        out[f"{name}_n_offtype_test"] = int(len(te_off))
    print(json.dumps(out, indent=2))
    run.summary.update({f"E2/{k}": v for k, v in out.items()})

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    groups_lbl = ["on-type", "off-type"]
    full = [out["full_acc_ontype"], out["full_acc_offtype"]]
    rem = [out["belief_removed_acc_ontype"], out["belief_removed_acc_offtype"]]
    x = np.arange(2); w = 0.36
    ax.bar(x - w/2, full, w, label="full gru_h", color="#d1495b", edgecolor="white")
    ax.bar(x + w/2, rem, w, label="belief-removed", color="#2a9d4a", edgecolor="white")
    ax.axhline(0.5, ls="--", color="#888", label="chance (binary)")
    for i, (a, b) in enumerate(zip(full, rem)):
        ax.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(groups_lbl); ax.set_ylim(0, 1.05)
    ax.set_ylabel("build-vs-mine accuracy")
    ax.set_title("E2 skill probe: on-type vs against-type commits")
    ax.legend(); fig.tight_layout()
    _logfig(run, outdir, "E2/offtype_generalisation", fig)
    return out


# ----------------------------------------------------------------- E3
def _belief_probe(X, cat, groups, seed=0):
    """3-class belief probe -> signed score s = P(lakes)-P(rocky), + true-acc fn."""
    p, tr, te = fit_probe(X, cat, groups, seed=seed, balanced=True)
    cls = list(p.named_steps["c"].classes_)
    il, ir = cls.index("lakes"), cls.index("rocky")
    score = lambda Z: p.predict_proba(Z)[:, il] - p.predict_proba(Z)[:, ir]
    return p, score


def exp3_steering(run, bundle, ckpt_path, X, df, groups, outdir, n_ep, alphas, device):
    import torch
    print("\n=== E3: causal steering selectivity ===")
    sys.path.insert(0, str(bundle.path))
    from env_min import MiniBridgeTunnelEnv          # noqa: E402
    from policy_min import PPOGRUPolicy               # noqa: E402

    cat = df["category"].to_numpy(); skill = df["final_commit"].to_numpy()
    cen = lambda lab, c, M=X: M[lab == c].mean(0)

    # belief subspace + readout probe (fit on TRAIN maps only)
    tr_mask = np.isin(groups, _train_maps(groups))
    Xtr, cattr, gtr = X[tr_mask], cat[tr_mask], groups[tr_mask]
    Bb = onb([cen(cattr, "lakes", Xtr) - cen(cattr, "rocky", Xtr),
              cen(cattr, "balanced", Xtr) - 0.5 * (cen(cattr, "lakes", Xtr) + cen(cattr, "rocky", Xtr))])
    _, belief_score = _belief_probe(Xtr, cattr, gtr)

    # directions (gru_h space), all scaled to the global gap norm
    sk_tr = skill[tr_mask]
    d_glob = cen(sk_tr, "build", Xtr) - cen(sk_tr, "mine", Xtr)
    gnorm = float(np.linalg.norm(d_glob))
    balm = cattr == "balanced"
    d_bal = (Xtr[balm & (sk_tr == "build")].mean(0) - Xtr[balm & (sk_tr == "mine")].mean(0))
    d_orth = d_glob - (d_glob @ Bb) @ Bb.T
    dirs = {"global DoM": G.unit(d_glob), "balanced DoM": G.unit(d_bal),
            "belief-orth": G.unit(d_orth)}
    b_axis = cen(cattr, "lakes", Xtr) - cen(cattr, "rocky", Xtr)
    for k, v in dirs.items():
        print(f"  dir {k:13s} cos(belief lakes-rocky)={G.cosine(v, b_axis):+.3f}")

    # held-out eval episodes, balanced across categories
    eval_maps = np.setdiff1d(np.unique(groups), _train_maps(groups))
    eps = _eval_episodes(bundle, eval_maps, n_ep)
    print(f"  {len(eps)} eval episodes on {len(eval_maps)} held-out maps")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pol = PPOGRUPolicy.from_checkpoint(ckpt, bundle.view_size,
                                       bundle.manifest["n_scalars"], device=device)
    torch.set_grad_enabled(False)

    rows = []
    conditions = [("baseline", 0.0, np.zeros(128, np.float32))]
    for name, d in dirs.items():
        for a in alphas:
            conditions.append((name, a, (a * gnorm * d).astype(np.float32)))

    for ci, (name, a, vec) in enumerate(conditions):
        inj = torch.from_numpy(vec).view(1, 1, -1).to(device) if np.any(vec) else None
        commits, beliefs, reaches = [], [], []
        for (terr, spawn, seed, true_cat) in eps:
            env = MiniBridgeTunnelEnv(terr, spawn, variant="btc",
                                      view_size=bundle.view_size, max_steps=bundle.manifest["max_steps"])
            torch.manual_seed(seed)
            obs = env.reset(); h = torch.zeros(1, 1, pol.gru_hidden, device=device)
            hs = []
            for _ in range(bundle.manifest["max_steps"]):
                o = {k: torch.from_numpy(np.asarray(v)[None]).to(device) for k, v in obs.items()}
                logits, _, h = pol.step(o, h, inject=inj)
                hs.append(h.squeeze().cpu().numpy())
                act = int(torch.distributions.Categorical(logits=logits).sample()[0])
                obs, reached, done = env.step(act)
                if done:
                    break
            commits.append(env.commit)
            reaches.append(reached)
            beliefs.append(float(np.mean(belief_score(np.stack(hs)))))
        commits = np.array(commits)
        rows.append(dict(direction=name, alpha=a,
                         p_none=float((commits == 0).mean()),
                         p_build=float((commits == 1).mean()),
                         p_mine=float((commits == 2).mean()),
                         skill_axis=float((commits == 1).mean() - (commits == 2).mean()),
                         decoded_belief=float(np.mean(beliefs)),
                         reach=float(np.mean(reaches))))
        print(f"  [{ci+1}/{len(conditions)}] {name:13s} a={a:+.1f}  "
              f"build={rows[-1]['p_build']:.2f} mine={rows[-1]['p_mine']:.2f} "
              f"belief={rows[-1]['decoded_belief']:+.3f} reach={rows[-1]['reach']:.2f}")

    base = next(r for r in rows if r["direction"] == "baseline")
    for r in rows:
        r["d_skill"] = r["skill_axis"] - base["skill_axis"]
        r["d_belief"] = r["decoded_belief"] - base["decoded_belief"]
    _log_steering(run, outdir, rows, base)
    return rows


def _train_maps(groups):
    u = np.unique(groups)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(u, int(len(u) * 0.7), replace=False))


def _eval_episodes(bundle, eval_maps, n_ep):
    m = bundle.maps; rng = np.random.default_rng(1)
    cats = m["category"]; seeds = m["map_seed"]
    per_cat = max(1, n_ep // 3)
    eps = []
    for c in CAT:
        mids = [i for i in range(len(cats)) if cats[i] == c and m_in(eval_maps, i, bundle)]
        for mid in rng.choice(mids, min(per_cat, len(mids)), replace=False) if mids else []:
            tid = int(rng.integers(0, 60))
            eps.append((m["terrain"][mid], m["spawn"][mid],
                        int(seeds[mid]) * 100000 + tid, c))
    return eps


def m_in(eval_maps, map_idx, bundle):
    return map_idx in set(eval_maps)


def _log_steering(run, outdir, rows, base):
    import wandb, pandas as pd
    tab = pd.DataFrame(rows)
    run.log({"E3/table": wandb.Table(dataframe=tab)})
    tab.to_csv(outdir / "E3_steering.csv", index=False)

    # selectivity scatter: |Δbelief| vs |Δskill|
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for name, col in DIR_COLORS.items():
        sub = tab[tab.direction == name].sort_values("alpha")
        ax.plot(sub["d_belief"].abs(), sub["d_skill"].abs(), "-o", color=col, label=name, ms=6)
    ax.scatter([0], [0], c="k", s=40, zorder=5, label="baseline")
    ax.set_xlabel("|Δ decoded belief|  (lower = belief preserved)")
    ax.set_ylabel("|Δ skill|  (higher = stronger skill steering)")
    ax.set_title("E3 steering selectivity — skill change vs belief drift")
    ax.grid(True, color=style.GRIDC); ax.set_facecolor(style.PANEL); ax.legend()
    fig.tight_layout(); _logfig(run, outdir, "E3/selectivity", fig)

    # response curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, col in DIR_COLORS.items():
        sub = tab[tab.direction == name].sort_values("alpha")
        axes[0].plot(sub["alpha"], sub["skill_axis"], "-o", color=col, label=name)
        axes[1].plot(sub["alpha"], sub["decoded_belief"], "-o", color=col, label=name)
    for ax, ttl, yl in [(axes[0], "skill axis  P(build)−P(mine)", "skill"),
                        (axes[1], "decoded belief  P(lakes)−P(rocky)", "belief")]:
        ax.axhline(base["skill_axis"] if yl == "skill" else base["decoded_belief"],
                   ls="--", color="#888", label="baseline")
        ax.set_xlabel("injection α (× global gap norm)"); ax.set_title(ttl)
        ax.grid(True, color=style.GRIDC); ax.set_facecolor(style.PANEL); ax.legend()
    fig.tight_layout(); _logfig(run, outdir, "E3/response_curves", fig)


def _logfig(run, outdir, key, fig):
    import wandb
    p = outdir / (key.replace("/", "__") + ".png")
    fig.savefig(p, bbox_inches="tight", dpi=150)
    run.log({key: wandb.Image(str(p))})
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="activation_datasets/btc_ppo")
    ap.add_argument("--checkpoint", default="released_models/bridge_tunnel_commit/ppo_commit_onehot.pt")
    ap.add_argument("--source", default="gru_h")
    ap.add_argument("--rows", type=int, default=120000, help="rows for probe/direction fitting")
    ap.add_argument("--n-ep", type=int, default=90, help="held-out steering episodes")
    ap.add_argument("--alphas", type=float, nargs="*", default=[-2, -1, -0.5, 0.5, 1, 2])
    ap.add_argument("--out-dir", default="outputs/analysis_exp")
    ap.add_argument("--wandb-project", default="bridge_tunnel_geometry")
    ap.add_argument("--wandb-mode", default="online")
    ap.add_argument("--run-name", default="geometry-experiments-btc")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import wandb
    style.apply_theme()
    outdir = Path(a.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    b = ActivationBundle(a.dataset)
    rng = np.random.default_rng(0)
    # stratified rows across category x final_commit
    lab = b.labels
    frac = a.rows / len(lab); take = []
    for idx in lab.groupby(["category", "final_commit"], observed=True).indices.values():
        k = min(len(idx), max(1, int(round(len(idx) * frac))))
        take.append(rng.choice(idx, k, replace=False))
    S = lab.iloc[np.sort(np.concatenate(take))].reset_index(drop=True)
    X = b.load_activations(a.source, S["row_id"])
    cat = S["category"].to_numpy(); skill = S["final_commit"].to_numpy()
    groups = S["map_id"].to_numpy()
    print(f"fit rows={len(S)}  source={a.source}({X.shape[1]}d)")

    run = wandb.init(project=a.wandb_project, mode=a.wandb_mode, name=a.run_name,
                     tags=["btc", "ppo", "experiments", "separability"],
                     config=dict(source=a.source, rows=a.rows, n_ep=a.n_ep, alphas=a.alphas))
    e1 = exp1_subspace(run, X, cat, skill, groups, outdir)
    e2 = exp2_offtype(run, X, S, groups, outdir)
    e3 = exp3_steering(run, b, a.checkpoint, X, S, groups, outdir, a.n_ep, a.alphas, a.device)
    (outdir / "experiments_summary.json").write_text(json.dumps({"E1": e1, "E2": e2, "E3": e3}, indent=2))
    run.finish()
    print("\nDONE")


if __name__ == "__main__":
    main()
