#!/usr/bin/env python3
"""Probe accuracy vs episode progress. For each 5% progress bucket we train a
SEPARATE linear probe (belief = category, skill = final_commit) on train-maps'
timesteps in that bucket and score it on held-out-maps' timesteps in that bucket.
Expectation: decodability rises as the agent gathers information.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechinterp.analysis.bundle import ActivationBundle

NB = 20                       # 20 buckets of 5%
NFOLD = 5                     # map-disjoint CV folds -> mean ± std
CENTERS = (np.arange(NB) + 0.5) * (100 / NB)
BELIEF_CL = ["rocky", "balanced", "lakes"]
SKILL_CL = ["none", "build", "mine"]


def probe_curve(X, y, bucket, groups, classes):
    """Per 5% bucket, per map-fold: train a probe on train-maps' rows, score on
    held-out-maps' rows in that bucket. Returns mean[NB], std[NB] over folds and
    cms[NB,C,C] (confusion counts summed over folds)."""
    C = len(classes)
    accs = np.full((NFOLD, NB), np.nan)
    cms = np.zeros((NB, C, C))
    for f, (tr_idx, te_idx) in enumerate(GroupKFold(NFOLD).split(X, y, groups)):
        trm = np.zeros(len(y), bool); trm[tr_idx] = True
        tem = np.zeros(len(y), bool); tem[te_idx] = True
        for b in range(NB):
            tr = np.where(trm & (bucket == b))[0]
            te = np.where(tem & (bucket == b))[0]
            if len(te) < 30 or len(np.unique(y[tr])) < 2:
                continue
            if len(tr) > 5000:
                tr = np.random.default_rng(f).choice(tr, 5000, replace=False)
            pipe = Pipeline([("s", StandardScaler()),
                             ("c", LogisticRegression(max_iter=1000))])
            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[te])
            accs[f, b] = (pred == y[te]).mean()
            cms[b] += confusion_matrix(y[te], pred, labels=classes)
    return np.nanmean(accs, 0), np.nanstd(accs, 0), cms


def run(name, src):
    b = ActivationBundle(f"activation_datasets/{name}")
    lab = b.labels
    S = lab.sample(min(150000, len(lab)), random_state=0).reset_index(drop=True)
    prog = (S["t"] / S["ep_len"].clip(lower=1)).to_numpy()
    bucket = np.clip((prog * NB).astype(int), 0, NB - 1)
    groups = S["map_id"].to_numpy()
    X = b.load_activations(src, S["row_id"])
    return {"belief": probe_curve(X, S["category"].to_numpy(), bucket, groups, BELIEF_CL),
            "skill": probe_curve(X, S["final_commit"].to_numpy(), bucket, groups, SKILL_CL)}


def conf_strip(name, factor, cms, classes, picks):
    fig, axes = plt.subplots(1, len(picks), figsize=(2.45 * len(picks), 2.8))
    for ax, b in zip(axes, picks):
        M = cms[b].astype(float)
        M = M / M.sum(1, keepdims=True).clip(min=1)
        ax.imshow(M, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes if b == picks[0] else [], fontsize=7)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] < 0.6 else "black")
        ax.set_title(f"{int(round(CENTERS[b]))}%", fontsize=10)
    fig.suptitle(f"{name} — {factor} confusion over progress (rows=true, cols=pred)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = Path(f"outputs/report/figs/confusion_progress_{name}_{factor}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print("wrote", p)


def main():
    specs = [("btc_ppo", "gru_h", "PPO  ·  gru_h"),
             ("btc_dreamer", "rssm_deter", "DreamerV3  ·  rssm_deter")]
    results = {lbl: run(name, src) for name, src, lbl in specs}
    picks = [0, 4, 8, 12, 16, 19]                # ~2.5,22.5,42.5,62.5,82.5,97.5 %
    for name, src, lbl in specs:
        r = results[lbl]
        conf_strip(name, "belief", r["belief"][2], BELIEF_CL, picks)
        conf_strip(name, "skill", r["skill"][2], SKILL_CL, picks)
    fig, axes = plt.subplots(1, len(specs), figsize=(12.5, 4.6), sharey=True)
    for ax, (name, src, lbl) in zip(axes, specs):
        r = results[lbl]
        for key, c, nm in [("belief", "#2e86c1", "belief (map type)"),
                           ("skill", "#d1495b", "skill (final commit)")]:
            m, s, _ = r[key]
            ax.errorbar(CENTERS, m, yerr=s, fmt="-o", color=c, lw=2.3, ms=5,
                        capsize=3, elinewidth=1.3, label=nm)
            ax.fill_between(CENTERS, m - s, m + s, color=c, alpha=0.15)
        ax.axhline(1 / 3, ls="--", color="#888", lw=1.2, label="chance (1/3)")
        ax.set_xlabel("episode progress (%)"); ax.set_title(lbl, fontsize=12)
        ax.set_ylim(0.2, 1.02); ax.set_xlim(0, 100)
        ax.set_facecolor("#eef3f8"); ax.grid(True, color="white"); ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel("held-out probe accuracy")
    fig.suptitle("Probe accuracy vs episode progress — separate probe per 5% bucket, "
                 f"mean ± std over {NFOLD} map-folds", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path("outputs/report/figs/probe_over_progress.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
    for lbl, r in results.items():
        bm = r["belief"][0]; km = r["skill"][0]
        print(f"{lbl}: belief {np.nanmin(bm):.2f}->{np.nanmax(bm):.2f}  "
              f"skill {np.nanmin(km):.2f}->{np.nanmax(km):.2f}")


if __name__ == "__main__":
    main()
