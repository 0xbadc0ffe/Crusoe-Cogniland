#!/usr/bin/env python3
"""Paired trajectory figure: for 2–3 qualitatively distinct routes on ONE map,
show the path on the map (left) next to its latent trajectory in PCA space (right),
coloured the same way — by commitment (BTC: blue=none → yellow=build / purple=mine)
or by segment (BT). One PCA is fit per (model×dataset) so all panels share PC axes.

    python scripts/figures/paired_trajectories.py --dataset activation_datasets/bt_dreamer \
        --source rssm_deter --map-id 4
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # scripts/
from mechinterp.analysis.bundle import ActivationBundle

# crossing colours shared across BT/BTC: build/bridge = yellow, mine/tunnel = purple
COMMIT_COL = {"none": "#1f5fd0", "build": "#ffd000", "mine": "#a800e6"}
SEG_COL = {"free": "#c2ccd6", "approach": "#7aa8d6", "avoid": "#3a6ea5",
           "bridge": "#ffd000", "tunnel": "#a800e6"}


def traj_rows(lab, map_id, traj_id):
    s = lab[(lab.map_id == map_id) & (lab.traj_id == traj_id)].sort_values("t")
    return s


def pick_trajs(lab, dec, map_id, is_commit, n=3):
    sub = lab[lab.map_id == map_id]
    eps = sub.groupby("traj_id").agg(reached=("reached", "first"),
                                     **({"fc": ("final_commit", "first")} if is_commit else {}))
    eps = eps[eps.reached]
    chosen, desc = [], []
    if is_commit:
        for fc in ["build", "mine", "none"]:
            cand = eps[eps.fc == fc]
            if len(cand):
                chosen.append(int(cand.index[0])); desc.append(f"commit: {fc}")
    else:
        # rank by crossing usage: pure-avoid, bridge, tunnel
        d = dec[dec.map_id == map_id]
        cross = d.groupby("traj_id")["choice"].apply(lambda s: set(s))
        want = [("detour (avoid only)", lambda c: c <= {"avoid"}),
                ("crosses — bridge", lambda c: "bridge" in c),
                ("crosses — tunnel", lambda c: "tunnel" in c)]
        for label, f in want:
            for tid in eps.index:
                if tid in cross.index and f(cross[tid]) and tid not in chosen:
                    chosen.append(int(tid)); desc.append(label); break
    return chosen[:n], desc[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--map-id", type=int, default=None)
    ap.add_argument("--trajs", type=int, nargs="*", default=None)
    ap.add_argument("--pca-rows", type=int, default=15000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    b = ActivationBundle(a.dataset)
    lab = b.labels
    import pandas as pd
    dec = pd.read_parquet(Path(a.dataset) / "decisions.parquet") if not b.is_commit else None
    is_commit = b.is_commit

    # ---- one PCA per model×dataset (shared axes) ----
    samp = lab.sample(min(a.pca_rows, len(lab)), random_state=0)
    Xs = b.load_activations(a.source, samp["row_id"])
    pca = PCA(n_components=2, random_state=0).fit(Xs)

    # ---- choose map + trajectories ----
    map_id = a.map_id
    if map_id is None:
        if is_commit:
            g = lab.groupby("map_id")["final_commit"].nunique()
            map_id = int(g.idxmax())
        else:
            map_id = 4
    if a.trajs:
        trajs = a.trajs; desc = [f"traj {t}" for t in trajs]
    else:
        trajs, desc = pick_trajs(lab, dec, map_id, is_commit)
    print(f"{b.name} · {a.source} · map {map_id} · trajs {trajs} ({desc})")

    terr = b.maps["terrain"][map_id]; palette = b.palette
    cat = b.maps["category"][map_id] if "category" in b.maps else None

    # project all chosen trajs first to set shared PC limits
    projs, keys, paths = [], [], []
    for tid in trajs:
        s = traj_rows(lab, map_id, tid)
        X = b.load_activations(a.source, s["row_id"])
        projs.append(pca.transform(X))
        keys.append(s["commit_state"].to_numpy() if is_commit else s["segment"].to_numpy())
        paths.append(s[["pos_r", "pos_c"]].to_numpy())
    allp = np.concatenate(projs)
    xpad = (allp[:, 0].max() - allp[:, 0].min()) * 0.08 + 1e-6
    ypad = (allp[:, 1].max() - allp[:, 1].min()) * 0.08 + 1e-6
    xlim = (allp[:, 0].min() - xpad, allp[:, 0].max() + xpad)
    ylim = (allp[:, 1].min() - ypad, allp[:, 1].max() + ypad)
    COL = COMMIT_COL if is_commit else SEG_COL

    n = len(trajs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.1 * n))
    axes = np.atleast_2d(axes)
    for i, (tid, d) in enumerate(zip(trajs, desc)):
        col_seq = [COL.get(k, "#888") for k in keys[i]]
        # --- left: map + path ---
        axm = axes[i, 0]
        axm.imshow(palette[terr], interpolation="nearest")
        xy = np.stack([paths[i][:, 1], paths[i][:, 0]], 1).astype(float)
        segs = np.stack([xy[:-1], xy[1:]], 1)
        axm.add_collection(LineCollection(segs, colors=col_seq[1:], linewidths=2.2))
        axm.scatter(*xy[0], c="white", s=55, edgecolors="k", marker="o", zorder=5)
        axm.scatter(*xy[-1], c="k", s=90, marker="*", zorder=5)
        axm.set_xticks([]); axm.set_yticks([])
        axm.set_title(f"map {map_id}{f' [{cat}]' if cat else ''} · traj {tid}\n{d}", fontsize=10)
        # --- right: latent PCA trajectory ---
        axp = axes[i, 1]
        p = projs[i]
        psegs = np.stack([p[:-1], p[1:]], 1)
        axp.add_collection(LineCollection(psegs, colors=col_seq[1:], linewidths=2.0, alpha=0.9))
        axp.scatter(p[0, 0], p[0, 1], c="white", s=55, edgecolors="k", marker="o", zorder=5)
        axp.scatter(p[-1, 0], p[-1, 1], c="k", s=90, marker="*", zorder=5)
        axp.set_xlim(*xlim); axp.set_ylim(*ylim)
        axp.set_xlabel("PC1"); axp.set_ylabel("PC2")
        axp.set_facecolor("#eef3f8"); axp.grid(True, color="white")
        axp.set_title(f"latent trajectory in {a.source} PCA", fontsize=10)
    # legend
    handles = [plt.Line2D([0], [0], color=c, lw=3, label=k) for k, c in COL.items()]
    handles += [plt.Line2D([0], [0], marker="o", mfc="white", mec="k", ls="", label="start"),
                plt.Line2D([0], [0], marker="*", mfc="k", mec="k", ls="", label="end")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    sch = "commitment (none→build/mine)" if is_commit else "segment"
    fig.suptitle(f"{b.name} · {a.source} — paths (left) and their latent trajectories (right), "
                 f"coloured by {sch}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = a.out or f"outputs/report/figs/paired_{b.name}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
