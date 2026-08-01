#!/usr/bin/env python3
"""Export the fork_wall belief manifold in 3-D PCA, plus skill directions.

Two questions, one rollout pass:

  (1) BELIEF MANIFOLD OVER TIME. Mean gru_h trajectory per map category,
      t = 0..T, projected into a 3-D PCA fit on all states. If the three
      category trajectories were a linear code the means would separate along a
      fixed axis and travel in parallel; curvature and time-varying separation
      instead say the manifold is non-linear, which we quantify by comparing a
      linear probe on gru_h against the same probe restricted to the 3 PCs, and
      by measuring how the lakes/rocky separation direction rotates over time.

  (2) SKILL DIRECTIONS. Hidden states are labelled by behavioural REGIME at the
      moment of action — ``build`` (executed a bridge placement), ``mine``
      (executed a rock removal), ``avoid`` (an obstacle is directly ahead and
      the agent moves laterally instead of using a skill). Skill axes are then
      extracted by several linear methods, because they do not agree and the
      disagreement matters:

        diff-of-means   mean(mine) - mean(build)          (what class_mean_direction uses)
        logistic        one-vs-rest probe weight row, un-standardised
        LDA             Fisher discriminant, whitens by within-class covariance
        ridge           regression onto a +/-1 skill indicator

      Each is compared by cosine against the BELIEF axes (probe rocky-lakes,
      and the empirical category class-mean axis) to test whether "steer the
      skill" and "steer the belief" are the same direction.

    python scripts/mechinterp/forkwall_pca3d_skills.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "bridge_tunnel"))
sys.path.insert(0, str(REPO / "scripts" / "mechinterp"))

from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv  # noqa: E402
from cogniland.bridge_tunnel.mapgen import generate_commit_map, CATEGORIES  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy  # noqa: E402
from cogniland.bridge_tunnel.steering import BELIEF2I, cosine  # noqa: E402
from train_belief_probe import load_belief_probe  # noqa: E402

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

FACE_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]
A_BUILD, A_MINE = 4, 5
SKILLS = ["build", "mine", "avoid"]


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


@torch.no_grad()
def rollout(policy, rec, n_traj, view_size, max_steps, device, commit, tmax):
    """gru_h per step with time index, plus skill-regime labels at action time."""
    Hh, Ww = rec.terrain.shape
    envs = [BridgeTunnelCommitEnv(map_record=rec, size=Hh, width=Ww, view_size=view_size,
                                  max_steps=max_steps, commit=commit) for _ in range(n_traj)]
    obs = [e.reset()[0] for e in envs]
    h = torch.zeros(1, n_traj, policy.gru_hidden, device=device)
    done = torch.zeros(n_traj, device=device)
    active = np.ones(n_traj, dtype=bool)
    traj = [[] for _ in range(n_traj)]          # (t, h) while alive
    skill_h = {k: [] for k in SKILLS}

    for t in range(max_steps):
        mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
        sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
        _, h = policy._gru_forward({"minimap": mm, "scalars": sc}, done[None], h)
        x = h.squeeze(0)
        logits, _ = policy._heads(x)
        xnp = x.cpu().numpy()
        acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            if t <= tmax:
                traj[i].append((t, xnp[i].copy()))
            a = int(acts[i])
            r0, c0 = e._pos
            # what is directly ahead of the agent right now?
            fr, fc = FACE_DELTA[a if a < 4 else e._facing]
            tr, tc = r0 + fr, c0 + fc
            ahead = int(rec.terrain[tr, tc]) if (0 <= tr < Hh and 0 <= tc < Ww) else T.OOB
            o, _, term, trunc, info = e.step(a)
            obs[i] = o
            if info["placed"]:
                skill_h["build"].append(xnp[i].copy())
            elif info["mined"]:
                skill_h["mine"].append(xnp[i].copy())
            elif a < 4 and ahead in (T.WATER, T.ROCK):
                # obstacle ahead, chose to move instead of using a skill
                skill_h["avoid"].append(xnp[i].copy())
            if term or trunc:
                active[i] = False
        done = torch.zeros(n_traj, device=device)
        if not active.any():
            break
    return traj, skill_h


def probe_bacc(X, y, groups, seed=0):
    if len(np.unique(y)) < 2:
        return float("nan")
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, groups))
    pipe = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=3000))])
    pipe.fit(X[tr], y[tr])
    return float(balanced_accuracy_score(y[te], pipe.predict(X[te])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=REPO / "outputs/ppo_checkpoints/ppo_gru_forkwall_noaux_seed1/final.pt")
    p.add_argument("--maps", type=int, default=14)
    p.add_argument("--traj", type=int, default=10)
    p.add_argument("--tmax", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed-start", type=int, default=120_000)
    p.add_argument("--cloud", type=int, default=2600, help="states kept for the 3-D cloud")
    p.add_argument("--out", type=Path,
                   default=REPO / "outputs/analysis/forkwall_pca3d_skills.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    policy, cargs, view_size, env_size, env_width = _load_policy(args.checkpoint, device)
    probe_lin, pmeta = load_belief_probe(args.checkpoint.parent / "belief_probe.pt", device)
    policy.belief = probe_lin
    commit = False if cargs.get("no_commit", False) else None
    gh = cargs.get("goal_half", 0)
    gh = gh if (gh is not None and gh >= 0) else None
    torch.manual_seed(0)

    per_cat_traj = {c: [] for c in CATEGORIES}
    skill_h = {k: [] for k in SKILLS}
    X, ycat, groups, tstep = [], [], [], []
    for ci, cat in enumerate(CATEGORIES):
        for j in range(args.maps):
            s = args.seed_start + ci * 500 + j
            rec = generate_commit_map(size=env_size, width=env_width, seed=s, category=cat,
                                      tree_frac=cargs.get("tree_frac", 0.03), goal_half=gh,
                                      fork_wall=True,
                                      passage_half=cargs.get("passage_half", 1),
                                      wall_margin=cargs.get("wall_margin", 1))
            tj, sk = rollout(policy, rec, args.traj, view_size, args.max_steps,
                             device, commit, args.tmax)
            per_cat_traj[cat].extend(tj)
            for k in SKILLS:
                skill_h[k].extend(sk[k])
            for ep in tj:
                for (t, hh) in ep:
                    X.append(hh); ycat.append(ci); groups.append(s); tstep.append(t)
        print(f"  {cat}: {sum(len(e) for e in per_cat_traj[cat])} states", flush=True)

    X = np.asarray(X, dtype=np.float64)
    ycat = np.asarray(ycat); groups = np.asarray(groups); tstep = np.asarray(tstep)
    print(f"pooled: {X.shape}   skills: " +
          ", ".join(f"{k}={len(v)}" for k, v in skill_h.items()))

    # ---------------- 3-D PCA ----------------
    pca = PCA(n_components=3, random_state=0).fit(X)
    Z = pca.transform(X)
    evr = pca.explained_variance_ratio_

    # mean trajectory per category, t = 0..tmax
    mean_traj = {}
    for cat in CATEGORIES:
        rows = []
        for t in range(args.tmax + 1):
            acc = [hh for ep in per_cat_traj[cat] for (tt, hh) in ep if tt == t]
            if len(acc) < 3:
                continue
            m = np.mean(acc, axis=0)
            rows.append({"t": t, "n": len(acc),
                         "xyz": [float(v) for v in pca.transform(m[None])[0]]})
        mean_traj[cat] = rows

    # ---------------- non-linearity evidence ----------------
    acc_full = probe_bacc(X, ycat, groups)
    acc_pc3 = probe_bacc(Z, ycat, groups)
    # how much the lakes->rocky separation direction ROTATES over time: if the code
    # were a fixed linear axis these would all be parallel (cos ~ 1)
    rot = []
    for t0 in range(0, args.tmax - 9, 10):
        m = (tstep >= t0) & (tstep < t0 + 10)
        if m.sum() < 30:
            continue
        a = X[m & (ycat == BELIEF2I["rocky"])]
        b = X[m & (ycat == BELIEF2I["lakes"])]
        if len(a) < 5 or len(b) < 5:
            continue
        rot.append({"t0": int(t0), "dir": unit(a.mean(0) - b.mean(0)).tolist()})
    rot_cos = [[float(np.dot(r1["dir"], r2["dir"])) for r2 in rot] for r1 in rot]

    # ---------------- skill directions ----------------
    have = [k for k in SKILLS if len(skill_h[k]) >= 30]
    Sx = np.concatenate([np.asarray(skill_h[k], dtype=np.float64) for k in have])
    Sy = np.concatenate([np.full(len(skill_h[k]), i) for i, k in enumerate(have)])
    mu = {k: np.asarray(skill_h[k], dtype=np.float64).mean(0) for k in have}

    dirs = {}
    if "mine" in have and "build" in have:
        dirs["diff_of_means: mine-build"] = unit(mu["mine"] - mu["build"])
    sc = StandardScaler().fit(Sx)
    Sz = sc.transform(Sx)
    # explicit one-vs-rest: sklearn dropped LogisticRegression(multi_class=...)
    for i, k in enumerate(have):
        lg = LogisticRegression(max_iter=4000).fit(Sz, (Sy == i).astype(int))
        dirs[f"logistic: {k} vs rest"] = unit(lg.coef_[0] / sc.scale_)
    try:
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(Sx, Sy)
        # at most n_classes-1 discriminants carry signal; `eigen` returns a full
        # (n_features, n_features) scaling matrix whose tail is numerical noise
        n_disc = min(len(have) - 1, lda.scalings_.shape[1])
        for i in range(n_disc):
            dirs[f"LDA: discriminant {i+1}"] = unit(lda.scalings_[:, i])
    except Exception as e:                                   # noqa: BLE001
        print(f"  (LDA skipped: {e})")
    if "mine" in have and "build" in have:
        m = np.isin(Sy, [have.index("mine"), have.index("build")])
        tgt = np.where(Sy[m] == have.index("mine"), 1.0, -1.0)
        rg = Ridge(alpha=1.0).fit(Sx[m], tgt)
        dirs["ridge: mine=+1 build=-1"] = unit(rg.coef_)

    # ---------------- belief axes for comparison ----------------
    W = probe_lin.weight.detach().cpu().numpy().astype(np.float64)
    belief_axes = {
        "belief probe: rocky-lakes": unit(W[BELIEF2I["rocky"]] - W[BELIEF2I["lakes"]]),
        "belief probe: rocky-balanced": unit(W[BELIEF2I["rocky"]] - W[BELIEF2I["balanced"]]),
        "category class-mean: rocky-lakes":
            unit(X[ycat == BELIEF2I["rocky"]].mean(0) - X[ycat == BELIEF2I["lakes"]].mean(0)),
    }

    cos_tbl = {dn: {bn: float(np.dot(dv, bv)) for bn, bv in belief_axes.items()}
               for dn, dv in dirs.items()}
    cos_skill = {a: {b: float(np.dot(dirs[a], dirs[b])) for b in dirs} for a in dirs}

    print("\ncos(skill direction, belief axis):")
    for dn, row in cos_tbl.items():
        print(f"  {dn:34s} " + "  ".join(f"{bn.split(':')[0][:9]}={v:+.3f}"
                                         for bn, v in row.items()))

    # ---------------- payload ----------------
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Z), size=min(args.cloud, len(Z)), replace=False)
    skill_xyz = {k: pca.transform(np.asarray(skill_h[k], dtype=np.float64)).tolist()
                 for k in have}
    skill_mean_xyz = {k: pca.transform(mu[k][None])[0].tolist() for k in have}

    out = {
        "checkpoint": args.checkpoint.parent.name,
        "probe_balanced_accuracy": pmeta["balanced_accuracy"],
        "explained_variance": [float(v) for v in evr],
        "tmax": args.tmax,
        "categories": list(CATEGORIES),
        "skills": have,
        "n_states": int(len(X)),
        "cloud": [{"xyz": [round(float(v), 3) for v in Z[i]],
                   "cat": int(ycat[i]), "t": int(tstep[i])} for i in idx],
        "mean_traj": mean_traj,
        "skill_points": {k: [[round(float(c), 3) for c in p] for p in v]
                         for k, v in skill_xyz.items()},
        "skill_means": {k: [float(c) for c in v] for k, v in skill_mean_xyz.items()},
        "skill_counts": {k: int(len(skill_h[k])) for k in have},
        "nonlinearity": {
            "probe_bacc_full_128d": acc_full,
            "probe_bacc_top3_pcs": acc_pc3,
            "sep_dir_rotation_cos": rot_cos,
            "sep_dir_windows": [r["t0"] for r in rot],
        },
        "cos_skill_vs_belief": cos_tbl,
        "cos_skill_vs_skill": cos_skill,
        # directions projected into PCA space, for drawing arrows
        "dir_xyz": {k: (pca.components_ @ v).tolist() for k, v in dirs.items()},
        "belief_dir_xyz": {k: (pca.components_ @ v).tolist() for k, v in belief_axes.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"\nPC variance: {evr.round(3)}  (total {evr.sum():.1%})")
    print(f"probe balanced-acc  full 128-D {acc_full:.3f}   top-3 PCs {acc_pc3:.3f}")
    print(f"saved {args.out}  ({args.out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
