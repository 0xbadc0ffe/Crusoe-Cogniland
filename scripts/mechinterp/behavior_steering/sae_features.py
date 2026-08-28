#!/usr/bin/env python3
"""Browse + label the SAE features; add the top behaviour features to the kit.

For every alive feature (activation frequency >= 1e-4 on TEST rows): frequency,
Pearson correlation with named covariates (action one-hots, water/rock in
view, phase one-hots, col_rel_wall, category one-hots, seen-surplus, route
label), and the 8 highest-activating (map_id, t) exemplars. Features are ranked
by |corr| with MINE, BUILD and the route label; the top decoder columns are
mapped back to RAW h space (multiply by the scaler sd, renormalise) and
appended to behavior_axes.npz as sae_mine_i / sae_build_i / sae_route_i.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=scripts/mechinterp/belief_report \
      python scripts/mechinterp/behavior_steering/sae_features.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/mechinterp/belief_report"))
sys.path.insert(0, str(REPO / "scripts/mechinterp/behavior_steering"))
from data import load, split_maps  # noqa: E402
from sae import SAE  # noqa: E402

OUT = REPO / "outputs/behavior_steering"
FIG = REPO / "paper/figures/behavior_steering"
A_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "BUILD", "MINE"]
TOP_PER_TARGET = 3          # decoder columns per behaviour added to the kit


def main():
    X, df = load("ppo")
    tr, te = split_maps(df)
    m_te = df["map_id"].isin(te).to_numpy()
    rows = np.flatnonzero(m_te)
    Z = np.asarray(X[rows], np.float32)
    d = torch.load(OUT / "sae_ppo.pt", weights_only=False)
    mu, sd = d["mu"], d["sd"]
    sae = SAE(d["d"], d["f"]); sae.load_state_dict(d["state_dict"]); sae.eval()
    with torch.no_grad():
        Fmat = sae.encode(torch.tensor((Z - mu) / sd)).numpy()
    sub = df.iloc[rows]

    # covariates
    cov = {}
    act = sub["action"].to_numpy()
    for i, n in enumerate(A_NAMES):
        cov[f"act_{n}"] = (act == i).astype(np.float32)
    cov["water_now"] = sub["water_now"].to_numpy(np.float32)
    cov["rock_now"] = sub["rock_now"].to_numpy(np.float32)
    for ph in ("evidence", "corridor", "past_wall"):
        cov[f"phase_{ph}"] = (sub["phase"] == ph).to_numpy(np.float32)
    cov["col_rel_wall"] = sub["col_rel_wall"].to_numpy(np.float32)
    for c in ("lakes", "balanced", "rocky"):
        cov[f"cat_{c}"] = (sub["category"] == c).to_numpy(np.float32)
    rs = sub["rock_seen"].to_numpy(np.float32)
    ws = sub["water_seen"].to_numpy(np.float32)
    cov["seen_surplus"] = (rs - ws) / np.maximum(rs + ws, 1.0)
    tools = df.assign(is_tool=df["action"].isin([4, 5]))
    ep_tool = tools.groupby("map_id")["is_tool"].sum()
    through = set(ep_tool[ep_tool > 0].index)
    cov["route_through"] = sub["map_id"].isin(through).to_numpy(np.float32)
    names = list(cov)
    C = np.stack([cov[n] for n in names], 1)

    # correlations (guard zero-variance)
    Fc = Fmat - Fmat.mean(0)
    Cc = C - C.mean(0)
    fs = Fmat.std(0); cs = C.std(0)
    R = (Fc.T @ Cc) / len(Fmat)
    R = R / np.outer(np.maximum(fs, 1e-8), np.maximum(cs, 1e-8))
    freq = (Fmat > 1e-6).mean(0)
    alive = np.flatnonzero(freq >= 1e-4)
    print(f"alive features: {len(alive)}/{Fmat.shape[1]}")

    # per-feature table
    table = {}
    mids = sub["map_id"].to_numpy(); ts = sub["t"].to_numpy()
    for j in alive:
        order = np.argsort(-np.abs(R[j]))[:5]
        top = np.argsort(-Fmat[:, j])[:8]
        table[int(j)] = dict(
            freq=round(float(freq[j]), 4),
            top_cov=[[names[k], round(float(R[j, k]), 3)] for k in order],
            exemplars=[[int(mids[i]), int(ts[i]), round(float(Fmat[i, j]), 2)]
                       for i in top])

    # rank by behaviour targets and map decoder cols to RAW h space
    Wd = sae.Wd.detach().numpy()               # (128, F), standardised space
    kit = dict(np.load(OUT / "behavior_axes.npz"))
    vb = kit["v_belief"]
    idx = {n: names.index(n) for n in ("act_MINE", "act_BUILD", "route_through")}
    sae_meta = {}
    for tgt, key in (("mine", "act_MINE"), ("build", "act_BUILD"),
                     ("route", "route_through")):
        rank = sorted(alive, key=lambda j: -abs(R[j, idx[key]]))
        for i, j in enumerate(rank[:TOP_PER_TARGET]):
            raw = Wd[:, j] * sd                # std-space direction -> raw h
            raw = raw / (np.linalg.norm(raw) + 1e-9)
            kname = f"sae_{tgt}_{i}"
            kit[kname] = raw.astype(np.float32)
            sae_meta[kname] = dict(
                feature=int(j), corr=round(float(R[j, idx[key]]), 3),
                freq=round(float(freq[j]), 4),
                cos_v_mine=round(float(raw @ kit["v_mine"]), 3),
                cos_v_build=round(float(raw @ kit["v_build"]), 3),
                cos_belief=round(float(raw @ vb), 3),
                top_cov=table[int(j)]["top_cov"])
    np.savez(OUT / "behavior_axes.npz", **kit)
    (OUT / "sae_feature_table.json").write_text(json.dumps(
        dict(info=d["info"], alive=len(alive), features=table,
             kit_additions=sae_meta), indent=1))

    # figure: top-30 features (by max |corr| over behaviour targets) x covariates
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    score = np.abs(R[:, [idx["act_MINE"], idx["act_BUILD"], idx["route_through"]]]).max(1)
    top30 = sorted(alive, key=lambda j: -score[j])[:30]
    Hm = R[np.array(top30)][:, :]
    with plt.rc_context({"figure.dpi": 150, "font.size": 8}):
        fig, ax = plt.subplots(figsize=(10.5, 7.2))
        im = ax.imshow(Hm, cmap="RdBu_r", vmin=-.8, vmax=.8, aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(top30)))
        ax.set_yticklabels([f"f{j}" for j in top30], fontsize=7)
        for yy in range(Hm.shape[0]):
            for xx in range(Hm.shape[1]):
                if abs(Hm[yy, xx]) > .25:
                    ax.text(xx, yy, f"{Hm[yy, xx]:+.2f}", ha="center", va="center",
                            fontsize=5.2,
                            color="white" if abs(Hm[yy, xx]) > .5 else "#111827")
        cb = fig.colorbar(im, ax=ax, shrink=.7)
        cb.set_label("Pearson r (feature activation vs covariate), test maps")
        inf = d["info"]
        ax.set_title("The SAE finds clean MINE and BUILD feature families "
                     "(action + its tile context); route intent is not a sparse feature\n"
                     f"SAE 128→1024, λ={inf['lam']}, held-out R²={inf['val_r2']:.3f}, "
                     f"L0={inf['l0']:.0f}, PCA R² at matched k={inf['pca_r2_at_L0']:.3f}",
                     loc="left", fontsize=9)
        fig.tight_layout()
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG / "fig_sae_features.png", bbox_inches="tight")
    print("kit additions:", json.dumps(sae_meta, indent=1)[:1200])
    print("wrote", OUT / "sae_feature_table.json", "and fig_sae_features.png")


if __name__ == "__main__":
    main()
