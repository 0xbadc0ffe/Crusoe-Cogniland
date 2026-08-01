#!/usr/bin/env python3
"""Fig-3 and Fig-4 analogues: representational geometry over training, and
whether that geometry predicts behaviour.

Fig 3 (theirs): hidden-state PCA at successive checkpoints with fixed points
overlaid. We keep the PCA-over-training panels and DROP the fixed-point markers
— on this task there are no slow points near the visited manifold (states move
~60% of their norm per step), so plotting them would be misleading.

Fig 4 (theirs): distance from the stable fixed point to the goal correlates with
reward-to-go. That construction needs a point-like goal in state space, which a
discrete-latent POMDP does not have. The analogous claim — *representational
geometry tracks task performance* — is testable with a quantity we do have:
how linearly separable the three map categories are in the recurrent state
(map-grouped probe accuracy, and cluster silhouette), against door accuracy.

One rollout pass per checkpoint feeds both figures.

    python scripts/mechinterp/forkwall_geometry_vs_performance.py --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "bridge_tunnel"))

from cogniland.bridge_tunnel.mapgen import generate_commit_map, CATEGORIES  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy  # noqa: E402
from eval_bridge_tunnel_forkwall_steered import batched_rollout_steered  # noqa: E402

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, silhouette_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

CAT_COLOR = {"balanced": "#5C6B57", "lakes": "#1E6FA6", "rocky": "#A3572A"}


def eval_checkpoint(ck, device, n_maps, n_traj, max_steps, seed_start):
    policy, cargs, view_size, env_size, env_width = _load_policy(ck, device)
    commit = False if cargs.get("no_commit", False) else None
    gh = cargs.get("goal_half", 0)
    gh = gh if (gh is not None and gh >= 0) else None
    X, y, groups, succ, doors = [], [], [], [], []
    for ci, cat in enumerate(CATEGORIES):
        for j in range(n_maps):
            s = seed_start + ci * 500 + j
            rec = generate_commit_map(size=env_size, width=env_width, seed=s, category=cat,
                                      tree_frac=cargs.get("tree_frac", 0.03), goal_half=gh,
                                      fork_wall=True,
                                      passage_half=cargs.get("passage_half", 1),
                                      wall_margin=cargs.get("wall_margin", 1))
            o = batched_rollout_steered(policy, rec, n_traj, view_size, max_steps, device,
                                        commit=commit, steer_fn=None, collect_hidden=True)
            if o["hiddens"] is not None and len(o["hiddens"]):
                h = o["hiddens"]
                if len(h) > 400:
                    h = h[np.random.default_rng(0).choice(len(h), 400, replace=False)]
                X.append(h); y.extend([ci] * len(h)); groups.extend([s] * len(h))
            succ.extend(o["success"].tolist())
            doors.extend(o["doors"])
    X = np.concatenate(X).astype(np.float64)
    y = np.asarray(y); groups = np.asarray(groups)

    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, y, groups))
    pipe = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000))])
    pipe.fit(X[tr], y[tr])
    bacc = float(balanced_accuracy_score(y[te], pipe.predict(X[te])))
    Z = PCA(n_components=2, random_state=0).fit_transform(X)
    sil = float(silhouette_score(Z, y)) if len(np.unique(y)) > 1 else float("nan")
    return dict(probe_bacc=bacc, silhouette=sil,
                success=float(np.mean(succ)),
                top_frac=float(np.mean([d == "top" for d in doors])),
                Z=Z, y=y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-prefix", default="ppo_gru_forkwall_noaux_dense_seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n-panels", type=int, default=10, help="checkpoints per seed")
    p.add_argument("--n-maps", type=int, default=4)
    p.add_argument("--n-traj", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed-start", type=int, default=95_000)
    p.add_argument("--dyn-json", type=Path,
                   default=REPO / "paper/figures/forkwall_dynamics_open.json")
    p.add_argument("--ckpt-root", type=Path, default=REPO / "outputs/ppo_checkpoints")
    p.add_argument("--out-prefix", type=Path, default=REPO / "paper/figures/forkwall_geometry")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    rows = {}
    pca_panels = {}
    for seed in args.seeds:
        d = args.ckpt_root / f"{args.run_prefix}{seed}"
        cks = sorted(d.glob("iter*.pt"),
                     key=lambda q: int(re.search(r"iter(\d+)", q.name).group(1)))
        cks.append(d / "final.pt")
        pick = sorted({int(round(v)) for v in np.geomspace(1, len(cks), num=args.n_panels)})
        cks = [cks[i - 1] for i in pick]
        rows[seed] = []
        print(f"\n=== seed {seed}: {len(cks)} checkpoints ===")
        for ck in cks:
            it = 0 if ck.name == "iter0.pt" else (
                10**9 if ck.name == "final.pt"
                else int(re.search(r"iter(\d+)", ck.name).group(1)))
            r = eval_checkpoint(ck, device, args.n_maps, args.n_traj,
                                args.max_steps, args.seed_start)
            if seed == args.seeds[0]:
                pca_panels[it] = (r["Z"], r["y"], r["success"])
            rows[seed].append({k: v for k, v in r.items() if k not in ("Z", "y")}
                              | {"iteration": it})
            print(f"  iter {it:>10}  probe {r['probe_bacc']:.3f}  sil {r['silhouette']:+.3f}  "
                  f"succ {r['success']:.3f}  top {r['top_frac']:.2f}", flush=True)

    # ---------- Fig 3 analogue: PCA over training ----------
    keys = sorted(pca_panels)[:8]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 3.0))
    axes = np.atleast_1d(axes)
    for ax, it in zip(axes, keys):
        Z, y, sc = pca_panels[it]
        for ci, cat in enumerate(CATEGORIES):
            m = y == ci
            ax.scatter(Z[m, 0], Z[m, 1], s=2.5, alpha=0.35, color=CAT_COLOR[cat],
                       linewidths=0, label=cat if ax is axes[0] else None)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{'final' if it > 1e8 else it} steps\nsucc {sc:.2f}", fontsize=8.5)
    axes[0].legend(fontsize=7, markerscale=3, loc="best")
    fig.suptitle(f"Recurrent-state geometry over training (seed {args.seeds[0]}) — "
                 "PCA of gru_h, coloured by map category. No fixed points are marked: "
                 "none exist near the visited manifold.", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    o1 = Path(str(args.out_prefix) + "_pca_over_training.png")
    o1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o1, dpi=140); print(f"\nsaved {o1}")

    # ---------- Fig 4 analogue: geometry vs performance ----------
    tau = {}
    if args.dyn_json.exists():
        dj = json.load(open(args.dyn_json))["results"]
        for s, rr in dj.items():
            tau[int(s)] = {r["iteration"]: r["taus_traj"][0] for r in rr}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    colors = {1: "#1E6FA6", 2: "#B4791E", 3: "#2F8F63"}
    ax = axes[0]
    for s, rr in rows.items():
        it = np.maximum([r["iteration"] for r in rr], 0.7)
        it = np.where(np.array(it) > 1e8, max(x for x in it if x < 1e8) * 1.3, it)
        ax.plot(it, [r["probe_bacc"] for r in rr], "o-", ms=3.5,
                color=colors.get(s, "k"), label=f"seed {s}")
    ax.axhline(1/3, color="gray", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("training iteration")
    ax.set_ylabel("belief probe balanced acc")
    ax.set_title("(A) category decodability over training", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.15)

    ax = axes[1]
    allx, ally = [], []
    for s, rr in rows.items():
        x = [r["probe_bacc"] for r in rr]; yv = [r["success"] for r in rr]
        ax.scatter(x, yv, s=28, color=colors.get(s, "k"), label=f"seed {s}", alpha=0.85)
        allx += x; ally += yv
    allx, ally = np.array(allx), np.array(ally)
    if len(allx) > 2:
        from scipy import stats
        lr = stats.linregress(allx, ally)
        xs = np.linspace(allx.min(), allx.max(), 50)
        ax.plot(xs, lr.intercept + lr.slope * xs, "k-", lw=1.3,
                label=f"r={lr.rvalue:+.2f}, p={lr.pvalue:.1e}")
    ax.set_xlabel("belief probe balanced acc"); ax.set_ylabel("success (correct door)")
    ax.set_title("(B) does decodability predict performance?", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.15)

    ax = axes[2]
    bx, by = [], []
    for s, rr in rows.items():
        if s not in tau:
            continue
        xs_, ys_ = [], []
        for r in rr:
            t = tau[s].get(r["iteration"])
            if t is None or not np.isfinite(t):
                continue
            xs_.append(t); ys_.append(r["success"])
        ax.scatter(xs_, ys_, s=28, color=colors.get(s, "k"), label=f"seed {s}", alpha=0.85)
        bx += xs_; by += ys_
    if len(bx) > 2:
        from scipy import stats
        lb = np.log10(np.clip(bx, 1e-6, None))
        lr2 = stats.linregress(lb, by)
        xs = np.linspace(lb.min(), lb.max(), 50)
        ax.plot(10 ** xs, lr2.intercept + lr2.slope * xs, "k-", lw=1.3,
                label=f"r={lr2.rvalue:+.2f}, p={lr2.pvalue:.1e}")
    ax.set_xscale("log"); ax.set_xlabel(r"integration time $\tau_1$ (steps)")
    ax.set_ylabel("success (correct door)")
    ax.set_title(r"(C) does memory timescale predict performance?", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.15, which="both")

    fig.suptitle("Representational geometry vs behaviour — the Fig-4 analogue "
                 "(no fixed-point-to-goal distance exists in this task)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    o2 = Path(str(args.out_prefix) + "_vs_performance.png")
    fig.savefig(o2, dpi=140); print(f"saved {o2}")

    jp = Path(str(args.out_prefix) + ".json")
    jp.write_text(json.dumps(rows, indent=2, default=float))
    print(f"saved {jp}")


if __name__ == "__main__":
    main()
