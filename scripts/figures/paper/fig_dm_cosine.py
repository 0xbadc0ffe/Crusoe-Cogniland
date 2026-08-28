#!/usr/bin/env python3
"""Pairwise cosine between the per-bin difference-of-means axes, in FULL space.

One heat map per agent. Cell (i, j) is the cosine between the unit rocky-lakes
axis fitted at bin i and the one fitted at bin j, both on training maps, in the
agent's own state space (128-d, 3072-d, 512-d). No PCA is involved: this is the
rotation the projection figure can only suggest.

The scale is diverging and centred on zero, so orthogonality reads as white and
a sign flip reads as blue.

  PYTHONPATH=src:scripts/mechinterp/belief_report python scripts/figures/paper/fig_dm_cosine.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "mechinterp" / "belief_report"))

import data as D  # noqa: E402

OUT = REPO / "paper/figures/forkwall_paper"
RC = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
      "axes.spines.top": False, "axes.spines.right": False}


def axes_for(agent):
    X, df = D.load(agent)
    tr, _ = D.split_maps(df)
    bins = D.bin_states(X, df)
    nb = len(D.BIN_EDGES) - 1
    V = []
    for b in range(nb):
        ids, cats, M = bins[b]
        m = np.isin(ids, tr)
        v, _ = D.fit_dm(M[m], cats[m])
        V.append(v)
    return np.stack(V), X.shape[1]


def main():
    names = [D.BIN_LABELS[b].split("\n")[0] for b in range(len(D.BIN_EDGES) - 1)]
    nb = len(names)
    stats = {}

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9))
        for ax, agent in zip(axes, D.AGENTS):
            V, dim = axes_for(agent)
            Cm = V @ V.T
            im = ax.imshow(Cm, cmap="RdBu_r", vmin=-1, vmax=1)
            for i in range(nb):
                for j in range(nb):
                    val = Cm[i, j]
                    ax.text(j, i, f"{val:.2f}".lstrip("0").replace("-0.", "-."),
                            ha="center", va="center", fontsize=6.4,
                            color="white" if abs(val) > .62 else "#111827")
            # mark the phase boundaries: evidence | corridor | wall
            for k in (4.5, 6.5):
                ax.axhline(k, color="#111827", lw=1.3)
                ax.axvline(k, color="#111827", lw=1.3)
            ax.set_xticks(range(nb)); ax.set_yticks(range(nb))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.4)
            ax.set_yticklabels(names, fontsize=7.4)
            off = Cm[~np.eye(nb, dtype=bool)]
            ax.set_title(f"{D.LBL[agent]}  ({dim}-d)\n"
                         f"off-diagonal mean {off.mean():+.2f}, min {off.min():+.2f}",
                         loc="left", fontsize=9.5)
            stats[agent] = dict(dim=int(dim), cos=Cm.round(4).tolist(),
                                off_mean=float(off.mean()), off_min=float(off.min()),
                                adjacent_mean=float(np.mean([Cm[i, i + 1]
                                                             for i in range(nb - 1)])),
                                evid1_corr2=float(Cm[1, 6]), evid1_wall=float(Cm[1, 7]))
            for sp in ax.spines.values():
                sp.set_visible(False)

        cb = fig.colorbar(im, ax=axes, fraction=.018, pad=.015)
        cb.set_label("cosine between the two bins' belief axes", fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)
        fig.suptitle("The belief axis rotates over the episode. PPO stays positively "
                     "aligned throughout; the world models reach orthogonality "
                     "between the evidence phase and the corridor.",
                     y=1.02, fontsize=10.5)
        fig.savefig(OUT / "fig_dm_cosine.png", bbox_inches="tight")
        print("wrote fig_dm_cosine.png")

    (OUT / "dm_axis_cosine.json").write_text(json.dumps(stats, indent=1))
    for a, s in stats.items():
        print(f"  {a:8s} dim={s['dim']:5d} off-diag mean {s['off_mean']:+.3f} "
              f"min {s['off_min']:+.3f}  adjacent {s['adjacent_mean']:+.3f}  "
              f"evid1->corr2 {s['evid1_corr2']:+.3f}")


if __name__ == "__main__":
    main()
