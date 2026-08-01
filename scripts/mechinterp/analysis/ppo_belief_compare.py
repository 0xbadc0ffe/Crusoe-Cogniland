#!/usr/bin/env python3
"""Compare PPO+GRU with vs without the auxiliary belief loss, on gru_h:

  (A) BELIEF decodability — map-category (balanced/lakes/rocky) probe, map-grouped
      split; per-class accuracy + confusion ("belief->map") matrices.
  (B) SKILL-DECISION RAMP — one ramp PER SKILL (avoid/bridge/tunnel): a binary
      one-vs-rest probe at each offset tau in [-15,+15] around the decision onset,
      reaching ~1.0 at t0 (the skill is being executed) and ramping up before.
  (C) SKILL-PHASE PCA — mean gru_h per (skill x {beginning,during,end}) in a shared
      PCA space, showing how the skills separate and progress through execution.

Output: outputs/report/ppo_belief/REPORT.html + figures.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import recall_score, confusion_matrix, balanced_accuracy_score
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mechinterp.analysis.bundle import ActivationBundle

MODELS = [("btc_ppo", "no-aux (released)", "#888"),
          ("btc_ppo_belief", "aux belief loss", "#c0392b")]
SRC = "gru_h"
W = 15
SKILLS = ["avoid", "bridge", "tunnel"]
CATS = ["balanced", "lakes", "rocky"]
SKILL_COL = {"avoid": "#1f5fd0", "bridge": "#e6a800", "tunnel": "#a800e6"}
OUT = Path("outputs/report/ppo_belief")


def _fit(X, y, g, seed=0):
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(X, y, g))
    pipe = Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(C=1.0, max_iter=2000))])
    pipe.fit(X[tr], y[tr])
    return y[te], pipe.predict(X[te])


# ---------------------------------------------------------------- (A) belief
def belief(name, n=120_000, seed=0):
    b = ActivationBundle(f"activation_datasets/{name}")
    lab = b.labels.sample(min(n, len(b.labels)), random_state=seed)
    ids = np.sort(lab["row_id"].to_numpy()); lab = b.labels.set_index("row_id").loc[ids]
    X = b.load_activations(SRC, ids)
    yt, yp = _fit(X, lab["category"].to_numpy(), lab["map_id"].to_numpy(), seed)
    rec = recall_score(yt, yp, labels=CATS, average=None)
    cm = confusion_matrix(yt, yp, labels=CATS).astype(float)
    cm /= cm.sum(1, keepdims=True)
    return dict(per_class=dict(zip(CATS, rec)),
                balanced=balanced_accuracy_score(yt, yp), cm=cm)


# ------------------------------------------------------------ (B) per-skill ramp
def collect_ramp(name, per_class=900, seed=0):
    b = ActivationBundle(f"activation_datasets/{name}")
    dec = pd.read_parquet(b.path / "decisions.parquet")
    lab = b.labels[["row_id", "map_id", "traj_id", "t"]]
    key = lab.set_index(["map_id", "traj_id", "t"])["row_id"].to_dict()
    rng = np.random.default_rng(seed)
    keep = []
    for sk in SKILLS:
        d = dec[dec["choice"] == sk]
        if len(d) > per_class:
            d = d.iloc[rng.choice(len(d), per_class, replace=False)]
        keep.append(d)
    D = pd.concat(keep)
    rows, labels, groups, offs = [], [], [], []
    for _, d in D.iterrows():
        for tau in range(-W, W + 1):
            r = key.get((int(d.map_id), int(d.traj_id), int(d.decision_step) + tau))
            if r is not None:
                rows.append(int(r)); labels.append(d.choice)
                groups.append(int(d.map_id)); offs.append(tau)
    rows = np.array(rows); labels = np.array(labels)
    groups = np.array(groups); offs = np.array(offs)
    uniq, inv = np.unique(rows, return_inverse=True)
    X = b.load_activations(SRC, uniq)[inv]
    return X, labels, groups, offs


def ramp_per_skill(name):
    X, y, g, offs = collect_ramp(name)
    out = {sk: {} for sk in SKILLS}
    for tau in range(-W, W + 1):
        m = offs == tau
        if m.sum() < 80:
            for sk in SKILLS:
                out[sk][tau] = np.nan
            continue
        for sk in SKILLS:
            yb = (y[m] == sk).astype(int)
            if yb.sum() < 10 or (1 - yb).sum() < 10:
                out[sk][tau] = np.nan; continue
            yt, yp = _fit(X[m], yb, g[m])
            out[sk][tau] = balanced_accuracy_score(yt, yp)
    return out


# ---------------------------------------------------------- (C) skill-phase PCA
def skill_phase_means(name, n_pca=60_000, seed=0):
    b = ActivationBundle(f"activation_datasets/{name}")
    lab = b.labels.sort_values(["map_id", "traj_id", "t"]).reset_index(drop=True)
    # contiguous skill-segment occurrences -> split each into 3 phases
    seg = lab["segment"].to_numpy()
    is_sk = np.isin(seg, SKILLS)
    # run-id: new run when (segment changes) or (traj changes)
    tjr = (lab["map_id"].astype(str) + ":" + lab["traj_id"].astype(str)).to_numpy()
    newrun = np.ones(len(lab), bool)
    newrun[1:] = (seg[1:] != seg[:-1]) | (tjr[1:] != tjr[:-1])
    runid = np.cumsum(newrun)
    phase = np.full(len(lab), "", object)
    rid_col = np.where(is_sk, runid, -1)
    df = pd.DataFrame({"i": np.arange(len(lab)), "rid": rid_col, "seg": seg})
    for rid, grp in df[df.rid >= 0].groupby("rid"):
        idx = grp["i"].to_numpy(); L = len(idx)
        thirds = np.floor(np.arange(L) * 3 / max(L, 1)).astype(int).clip(0, 2)
        for k, ph in enumerate(["beginning", "during", "end"]):
            phase[idx[thirds == k]] = ph
    # PCA basis on a random gru_h sample
    rng = np.random.default_rng(seed)
    samp = np.sort(rng.choice(len(lab), min(n_pca, len(lab)), replace=False))
    Xs = b.load_activations(SRC, lab["row_id"].to_numpy()[samp])
    pca = PCA(3, random_state=0).fit(StandardScaler().fit_transform(Xs))
    scaler = StandardScaler().fit(Xs)
    means = {}
    for sk in SKILLS:
        for ph in ["beginning", "during", "end"]:
            sel = (seg == sk) & (phase == ph)
            if sel.sum() < 20:
                continue
            ids = np.sort(lab["row_id"].to_numpy()[sel][:8000])
            Xm = b.load_activations(SRC, ids).mean(0, keepdims=True)
            means[(sk, ph)] = pca.transform(scaler.transform(Xm))[0]
    return means, pca.explained_variance_ratio_[:2] * 100


# --------------------------------------------------------------------- report
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bel = {m[0]: belief(m[0]) for m in MODELS}
    for m in MODELS:
        print(f"[belief] {m[0]}: per-class", {k: round(v, 3) for k, v in bel[m[0]]['per_class'].items()},
              "balanced", round(bel[m[0]]['balanced'], 3), flush=True)
    ramp = {}
    for m in MODELS:
        print(f"[ramp] {m[0]} ...", flush=True)
        ramp[m[0]] = ramp_per_skill(m[0])
    pca = {}
    for m in MODELS:
        print(f"[pca] {m[0]} ...", flush=True)
        pca[m[0]] = skill_phase_means(m[0])

    # ---- FIG 1: belief per-class bars + confusion matrices
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    x = np.arange(3); w = 0.36
    for i, (name, lbl, c) in enumerate(MODELS):
        vals = [bel[name]["per_class"][k] for k in CATS]
        ax[0].bar(x + (i - .5) * w, vals, w, color=c, label=lbl)
        for j, v in enumerate(vals):
            ax[0].text(x[j] + (i - .5) * w, v + .02, f"{v:.2f}", ha="center", fontsize=9)
    ax[0].axhline(1/3, ls="--", color="k", alpha=.5, label="chance")
    ax[0].set_xticks(x); ax[0].set_xticklabels(CATS); ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("per-class accuracy (recall)")
    ax[0].set_title("(A) Belief decodability per map type\nmap-grouped 3-class probe")
    ax[0].legend(fontsize=8)
    for k, (name, lbl, c) in enumerate(MODELS):
        a = ax[k + 1]; cm = bel[name]["cm"]
        im = a.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for r in range(3):
            for cc in range(3):
                a.text(cc, r, f"{cm[r,cc]:.2f}", ha="center", va="center",
                       color="white" if cm[r, cc] > .5 else "black", fontsize=10)
        a.set_xticks(range(3)); a.set_xticklabels(CATS, rotation=30, fontsize=8)
        a.set_yticks(range(3)); a.set_yticklabels(CATS, fontsize=8)
        a.set_xlabel("predicted"); a.set_ylabel("true map type")
        a.set_title(f"belief→map confusion\n{lbl} (bal {bel[name]['balanced']:.2f})", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_belief.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- FIG 2: per-skill ramp (3 panels)
    taus = list(range(-W, W + 1))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for p, sk in enumerate(SKILLS):
        for name, lbl, c in MODELS:
            ax[p].plot(taus, [ramp[name][sk][t] for t in taus], color=c, lw=2,
                       marker="o", ms=3, label=lbl)
        ax[p].axvline(0, ls=":", color="k", alpha=.6)
        ax[p].axhline(0.5, ls="--", color="k", alpha=.4)
        ax[p].set_title(f"{sk}", color=SKILL_COL[sk], fontweight="bold")
        ax[p].set_xlabel("steps relative to decision (t0)"); ax[p].grid(alpha=.3)
        if p == 0:
            ax[p].set_ylabel("one-vs-rest balanced accuracy"); ax[p].legend(fontsize=8)
    fig.suptitle("(B) Per-skill decision ramp — how early each crossing is decodable in gru_h "
                 "(chance 0.5, ~1.0 at onset)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_ramp.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- FIG 3: skill-phase PCA
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    msz = {"beginning": 70, "during": 150, "end": 280}
    for k, (name, lbl, c) in enumerate(MODELS):
        means, evr = pca[name]; a = ax[k]
        for sk in SKILLS:
            pts = [means[(sk, ph)] for ph in ["beginning", "during", "end"] if (sk, ph) in means]
            if not pts:
                continue
            P = np.stack(pts)
            a.plot(P[:, 0], P[:, 1], "-", color=SKILL_COL[sk], lw=2, alpha=.6, zorder=1)
            for ph in ["beginning", "during", "end"]:
                if (sk, ph) in means:
                    p = means[(sk, ph)]
                    a.scatter(p[0], p[1], s=msz[ph], color=SKILL_COL[sk],
                              edgecolors="k", linewidths=1, zorder=3)
        a.set_title(f"{lbl}\nPC1 {evr[0]:.0f}% · PC2 {evr[1]:.0f}%")
        a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.set_facecolor("#eef3f8"); a.grid(True, color="white")
    handles = [plt.Line2D([], [], color=SKILL_COL[s], lw=3, label=s) for s in SKILLS]
    handles += [plt.scatter([], [], s=msz[p], color="#aaa", edgecolors="k", label=p)
                for p in ["beginning", "during", "end"]]
    ax[1].legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle("(C) PCA of mean gru_h per skill × execution phase (small→large = beginning→during→end)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_pca.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    pd.DataFrame({"tau": taus, **{f"{m[0]}_{sk}": [ramp[m[0]][sk][t] for t in taus]
                                  for m in MODELS for sk in SKILLS}}).to_csv(
        OUT / "ramp_per_skill.csv", index=False)
    np.save(OUT / "belief_results.npy", bel, allow_pickle=True)
    print("wrote figures fig_belief / fig_ramp / fig_pca")


if __name__ == "__main__":
    main()
