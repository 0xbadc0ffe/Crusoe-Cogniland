#!/usr/bin/env python3
"""Belief↔skill separability on ANY bundle/source (PPO gru_h or Dreamer rssm_deter).

Runs three confound/separability tests and logs one W&B run:
  C  within-category control — is the apparent cos(belief, skill) a LABEL CONFOUND?
     (global cos vs cos measured at fixed map category; Cramér's V of the labels)
  E1 subspace — does skill survive removing the belief subspace (and vice versa)?
  E2 off-type — does a build-vs-mine probe generalise to against-type commits,
     on full vs belief-removed features?

    python -m scripts.mechinterp.analysis.separability_report \
        --dataset activation_datasets/btc_dreamer --source rssm_deter --wandb-mode online
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from .bundle import ActivationBundle
from . import geometry as G, style
from .run_steering_experiments import exp1_subspace, exp2_offtype

CAT = ["rocky", "balanced", "lakes"]


def within_category_control(run, X, cat, skill, df, outdir, source):
    print("\n=== C: within-category control (confound check) ===")
    ct = pd.crosstab(df["category"], df["final_commit"])
    chi2 = chi2_contingency(ct.values)[0]
    n = ct.values.sum()
    V = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
    cen = lambda lab, c: X[lab == c].mean(0)
    b = cen(cat, "lakes") - cen(cat, "rocky")
    out = {"cramers_v": V,
           "cos_global": G.cosine(b, cen(skill, "build") - cen(skill, "mine"))}
    for c in CAT:
        m = cat == c
        nb = int(((skill == "build") & m).sum())
        nm = int(((skill == "mine") & m).sum())
        sc = X[m & (skill == "build")].mean(0) - X[m & (skill == "mine")].mean(0)
        out[f"cos_within_{c}"] = G.cosine(b, sc)
        out[f"n_within_{c}"] = [nb, nm]
    print(json.dumps(out, indent=2))
    run.summary.update({f"C/{k}": v for k, v in out.items() if np.isscalar(v)})

    labels = ["global"] + [f"within\n{c}" for c in CAT]
    vals = [out["cos_global"]] + [out[f"cos_within_{c}"] for c in CAT]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    cols = ["#888"] + [style.CATEGORY_COLORS[c] for c in CAT]
    ax.bar(labels, vals, color=cols, edgecolor="white")
    ax.axhline(0, color="#333", lw=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:+.2f}", ha="center", fontsize=9)
    ax.set_ylim(-1, 1); ax.set_ylabel("cos(belief lakes−rocky, build−mine)")
    ax.set_title(f"{source}: belief↔skill cosine — global vs at fixed belief\n"
                 f"(collapse ⇒ confound)   Cramér's V={V:.2f}")
    fig.tight_layout()
    p = outdir / f"{source}__within_category_control.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    import wandb
    run.log({f"C/{source}_within_category_control": wandb.Image(str(p))})
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--source", default=None, help="default = bundle's primary belief carrier")
    ap.add_argument("--rows", type=int, default=80000)
    ap.add_argument("--out-dir", default="outputs/analysis_sep")
    ap.add_argument("--wandb-project", default="bridge_tunnel_geometry")
    ap.add_argument("--wandb-mode", default="online")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    import wandb
    style.apply_theme()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    b = ActivationBundle(args.dataset)
    src = args.source or ("rssm_deter" if "rssm_deter" in b.sources else b.sources[0])
    print(f"bundle={b.name} source={src}")

    rng = np.random.default_rng(0)
    lab = b.labels
    frac = args.rows / len(lab); take = []
    for idx in lab.groupby(["category", "final_commit"], observed=True).indices.values():
        k = min(len(idx), max(1, int(round(len(idx) * frac))))
        take.append(rng.choice(idx, k, replace=False))
    S = lab.iloc[np.sort(np.concatenate(take))].reset_index(drop=True)
    X = b.load_activations(src, S["row_id"])
    cat = S["category"].to_numpy(); skill = S["final_commit"].to_numpy()
    groups = S["map_id"].to_numpy()
    print(f"rows={len(S)}  source={src}({X.shape[1]}d)")

    run = wandb.init(project=args.wandb_project, mode=args.wandb_mode,
                     name=args.run_name or f"separability-{b.name}-{src}",
                     tags=[b.name, src, "separability", "confound"],
                     config=dict(source=src, rows=args.rows, dim=int(X.shape[1])))
    c = within_category_control(run, X, cat, skill, S, out, src)
    e1 = exp1_subspace(run, X, cat, skill, groups, out)
    e2 = exp2_offtype(run, X, S, groups, out)
    (out / f"{b.name}_{src}_separability.json").write_text(
        json.dumps({"C": c, "E1": e1, "E2": e2}, indent=2))
    run.finish()
    print("DONE")


if __name__ == "__main__":
    main()
