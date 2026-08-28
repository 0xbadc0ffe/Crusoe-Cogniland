#!/usr/bin/env python3
"""The difference-of-means belief axis, drawn in PCA space.

One point per (episode, bin): the mean carried state over the rows of that
episode inside that position bin -- exactly the aggregation the probes use.
Points are coloured by map type, with hue for the type and lightness for the
bin. For every bin a segment joins the lakes class mean to the rocky class
mean: that segment IS the difference-of-means axis of that bin, shown in place.

The axis is computed in the FULL state space and PCA is used only to look at
it, so a segment is only as honest as the share of its axis that lies in the
plane being drawn. That share is printed, written on each faithful segment, and
used to dot and fade the segments that mostly point out of the plane. Without
it a short segment reads as a small separation, which for the world models in
PC1-PC3 would be badly wrong.

  PYTHONPATH=src:scripts/mechinterp/belief_report python scripts/figures/paper/fig_dm_pca.py
  ... --agents dreamer,storm --pcs 1,3     # one figure, one panel per agent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "mechinterp" / "belief_report"))

import data as D  # noqa: E402

OUT = REPO / "paper/figures/forkwall_paper"
RC = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
      "axes.spines.top": False, "axes.spines.right": False}
# Two encodings at once: HUE says which map type, LIGHTNESS says which bin.
RAMP = {"lakes": "GnBu", "rocky": "YlOrRd"}
FAITHFUL = 0.25          # below this share of the axis, a segment is not drawn solid


def bin_shade(cat, b, nb):
    return matplotlib.colormaps[RAMP[cat]](0.34 + 0.60 * b / max(nb - 1, 1))


def prepare(agent, px, py):
    """-> PC scores, categories, bin ids, per-bin class means in PC space, the
    per-bin share of the axis inside the drawn plane, and explained variance."""
    X, df = D.load(agent)
    tr, _ = D.split_maps(df)
    bins = D.bin_states(X, df)
    nb = len(D.BIN_EDGES) - 1

    P, C, B = [], [], []
    for b in range(nb):
        ids, cats, M = bins[b]
        P.append(M); C.append(cats); B.append(np.full(len(ids), b))
    P = np.concatenate(P); C = np.concatenate(C); B = np.concatenate(B)

    pca = PCA(n_components=max(px, py) + 1).fit(P)
    Z = pca.transform(P)

    seg, frac = [], []
    for b in range(nb):
        ids, cats, M = bins[b]
        m = np.isin(ids, tr)                       # axis fit on train maps only
        mu_l = M[m][cats[m] == "lakes"].mean(0)
        mu_r = M[m][cats[m] == "rocky"].mean(0)
        v = mu_r - mu_l
        v = v / (np.linalg.norm(v) + 1e-12)
        seg.append((pca.transform(mu_l[None])[0], pca.transform(mu_r[None])[0]))
        frac.append(float(np.linalg.norm(pca.components_[[px, py]] @ v)))
    return Z, C, B, seg, frac, pca.explained_variance_ratio_, P.shape[1], nb


def draw(ax, agent, px, py, show_key):
    Z, C, B, seg, frac, ev, dim, nb = prepare(agent, px, py)
    for cat in ("lakes", "rocky"):
        for b in range(nb):
            m = (C == cat) & (B == b)
            ax.scatter(Z[m, px], Z[m, py], s=4.6, alpha=.45,
                       color=bin_shade(cat, b, nb), lw=0, zorder=2)

    unfaithful = []
    for b, (p_l, p_r) in enumerate(seg):
        f = frac[b]
        ok = f >= FAITHFUL
        ax.plot([p_l[px], p_r[px]], [p_l[py], p_r[py]], color="#111827",
                lw=2.2 if ok else 1.2, ls="-" if ok else ":",
                alpha=1.0 if ok else .40, solid_capstyle="round", zorder=6)
        ax.scatter([p_l[px]], [p_l[py]], s=44 if ok else 20,
                   color=bin_shade("lakes", b, nb), ec="black", lw=.8,
                   alpha=1.0 if ok else .5, zorder=7)
        ax.scatter([p_r[px]], [p_r[py]], s=44 if ok else 20,
                   color=bin_shade("rocky", b, nb), ec="black", lw=.8,
                   alpha=1.0 if ok else .5, zorder=7)
        name = D.BIN_LABELS[b].split("\n")[0]
        if ok:                                   # only label what is drawn honestly
            ax.annotate(f"{name} ({f:.0%})",
                        ((p_l[px] + p_r[px]) / 2, (p_l[py] + p_r[py]) / 2),
                        xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=7.4, color="#111827", zorder=8)
        else:
            unfaithful.append(f"{name} {f:.0%}")

    ax.set_xlabel(f"PC{px+1} ({ev[px]*100:.0f}% of variance)")
    ax.set_ylabel(f"PC{py+1} ({ev[py]*100:.0f}% of variance)")
    ax.set_title(f"{D.LBL[agent]}  ({dim}-d)", loc="left", fontsize=9.8)
    ax.grid(alpha=.18, lw=.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * (0.26 if show_key else 0.12))

    if unfaithful:                               # crowded bins: listed, not labelled
        ax.text(.985, .015,
                "axis mostly outside this plane:\n" + ",  ".join(unfaithful),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
                color="#9ca3af", zorder=9)
    if show_key:
        x0, y0, dx, dy = .013, .975, .020, .048
        for i, cat in enumerate(("lakes", "rocky")):
            ax.text(x0, y0 - i * dy + .012, cat, transform=ax.transAxes,
                    fontsize=8.2, va="bottom", ha="left")
            for b in range(nb):
                ax.add_patch(plt.Rectangle(
                    (x0 + .085 + b * dx, y0 - i * dy - .004), dx * .88, .024,
                    transform=ax.transAxes, facecolor=bin_shade(cat, b, nb),
                    edgecolor="none", clip_on=False, zorder=9))
        ax.text(x0 + .085, y0 - 2 * dy + .006, "early", transform=ax.transAxes,
                fontsize=6.8, color="#6b7280")
        ax.text(x0 + .085 + (nb - 1) * dx, y0 - 2 * dy + .006, "wall",
                transform=ax.transAxes, fontsize=6.8, color="#6b7280", ha="right")
    return dict(agent=agent, dim=int(dim), pcs=[px + 1, py + 1],
                explained_variance=[float(x) for x in ev],
                axis_fraction_in_shown_pcs={D.BIN_LABELS[b].split("\n")[0]: frac[b]
                                            for b in range(nb)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="ppo")
    ap.add_argument("--pcs", default="1,2", help="which two PCs to draw, 1-indexed")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    px, py = (int(t) - 1 for t in a.pcs.split(","))
    agents = a.agents.split(",")

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(agents),
                                 figsize=(8.6 * len(agents), 6.2), squeeze=False)
        stats = [draw(axes[0][i], ag, px, py, show_key=(i == 0))
                 for i, ag in enumerate(agents)]
        fig.suptitle(
            "The difference-of-means belief axis, one segment per position bin. "
            "Hue is the map type, lightness is the bin.\n"
            "PCA is for viewing only: the axis is fitted in the full state space, "
            "and the percentage on each segment is how much of it this plane shows.",
            y=1.0, fontsize=10.2)
        fig.tight_layout()
        name = a.out or f"fig_dm_pca_{'_'.join(agents)}_pc{px+1}{py+1}.png"
        fig.savefig(OUT / name, bbox_inches="tight")
        print("wrote", name)

    (OUT / name.replace(".png", ".json")).write_text(json.dumps(stats, indent=1))
    for s in stats:
        sh = {k: round(v, 3) for k, v in s["axis_fraction_in_shown_pcs"].items()}
        print(f"  {s['agent']:8s} dim={s['dim']:5d} "
              f"PC{s['pcs'][0]}-PC{s['pcs'][1]} share {sh}")


if __name__ == "__main__":
    main()
