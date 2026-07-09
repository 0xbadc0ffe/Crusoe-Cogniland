#!/usr/bin/env python
"""Belief-plane vs behavior-manifold figure: disentangle the held cue MEMORY
(belief) from the motor POLICY (behavior) inside the PPO+GRU hidden state.

Belief lives in a 2-D linear subspace = the shape-probe direction (+) the
colour-probe direction; project the hidden onto it and each cue is a fixed point
(stable all episode). Behavior lives in the residual (belief removed) — its top
PCA is the maze trajectory that flows over the episode. The two subspaces are
near-orthogonal: belief encodes the cue and is invariant to maze phase; behavior
encodes phase and is invariant to the cue.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression, Ridge

CUE_NAMES = ["green_up", "blue_up", "green_down", "blue_down"]
CUE_COL = ["#1b9e77", "#3b6fb6", "#7fd4b8", "#9ec9ec"]   # green_up, blue_up, green_down, blue_down
PHASE_NAMES = ["pre-cue", "cue-room", "pre-branch", "branch", "post", "door"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#bbbbbb", "axes.linewidth": 1.0,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def clean_ax(ax, xl, yl, title, subtitle=None):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(xl, fontsize=10, color="#555"); ax.set_ylabel(yl, fontsize=10, color="#555")
    ax.set_title(title, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="#888")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/ppo_runs/ppo_4cue_vs4/activations.npz")
    ap.add_argument("--out", default="outputs/belief_vs_behavior.png")
    ap.add_argument("--sub", type=int, default=4500)
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    feat = d["feat"].astype(np.float64)
    shape, colour, cue, phase, t = d["shape"], d["colour"], d["cue_type"], d["phase"], d["t"]
    cue_name = str(d["cue"])

    # standardize
    mu, sd = feat.mean(0), feat.std(0) + 1e-6
    Z = (feat - mu) / sd

    # ---- belief directions: linear probes for shape & colour ----
    w_shape = unit(LogisticRegression(max_iter=2000, C=1.0).fit(Z, shape).coef_[0])
    w_colour = unit(LogisticRegression(max_iter=2000, C=1.0).fit(Z, colour).coef_[0])
    # orthonormal belief basis (Gram-Schmidt) for residualising
    b1 = w_shape
    b2 = unit(w_colour - (w_colour @ b1) * b1)
    Qb = np.stack([b1, b2], 1)                       # (D,2)

    belief = Z @ Qb                                    # project onto orthonormal belief basis

    # ---- behavior manifold: PCA of belief-removed residual ----
    resid = Z - Z @ Qb @ Qb.T
    rc = resid - resid.mean(0)
    U, S, Vt = np.linalg.svd(rc, full_matrices=False)
    behav = rc @ Vt[:2].T                            # (N,2)

    # ---- disentanglement metrics (measured, not by construction) ----
    w_pos = unit(Ridge(alpha=10.0).fit(Z, d["agent_x"].astype(np.float64)).coef_)  # behavior/position dir
    cos_sc = abs(float(w_shape @ w_colour))                # shape vs colour
    cos_bp = max(abs(float(w_shape @ w_pos)), abs(float(w_colour @ w_pos)))  # belief vs position(behavior)

    rng = np.random.default_rng(0)
    idx = rng.choice(feat.shape[0], min(a.sub, feat.shape[0]), replace=False)

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 11))
    fig.suptitle(f"Disentangling BELIEF from BEHAVIOR in the {cue_name} PPO+GRU memory",
                 fontsize=16, fontweight="bold", y=0.985)

    # (0,0) BELIEF plane coloured by CUE  -> 4 fixed clusters
    axx = ax[0, 0]
    for c in range(4):
        m = cue[idx] == c
        axx.scatter(belief[idx][m, 0], belief[idx][m, 1], s=7, c=CUE_COL[c], alpha=0.55,
                    edgecolors="none", label=CUE_NAMES[c])
    for c in range(4):
        m = cue == c
        axx.annotate(CUE_NAMES[c], (belief[m, 0].mean(), belief[m, 1].mean()),
                     fontsize=10, fontweight="bold", ha="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=CUE_COL[c], alpha=0.9))
    axx.axhline(0, color="#ddd", lw=0.8, zorder=0); axx.axvline(0, color="#ddd", lw=0.8, zorder=0)
    clean_ax(axx, "shape-probe direction →", "colour-probe direction →",
             "BELIEF  —  held cue memory", "coloured by CUE  →  4 fixed clusters")

    # (0,1) BEHAVIOR manifold coloured by PHASE -> flowing trajectory
    axx = ax[0, 1]
    sc = axx.scatter(behav[idx, 0], behav[idx, 1], s=7, c=phase[idx], cmap="viridis",
                     alpha=0.6, edgecolors="none")
    # mean trajectory over timestep
    tmax = int(t.max()); xs, ys = [], []
    for tt in range(tmax + 1):
        m = t == tt
        if m.sum() >= 20:
            xs.append(behav[m, 0].mean()); ys.append(behav[m, 1].mean())
    axx.plot(xs, ys, "-", color="#d1495b", lw=2.0, alpha=0.9, zorder=5)
    axx.scatter([xs[0]], [ys[0]], c="#d1495b", s=60, marker="o", zorder=6)
    axx.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-3], ys[-3]),
                 arrowprops=dict(arrowstyle="-|>", color="#d1495b", lw=2), zorder=6)
    cb = fig.colorbar(sc, ax=axx, ticks=range(6), fraction=0.046, pad=0.02)
    cb.ax.set_yticklabels(PHASE_NAMES, fontsize=8)
    clean_ax(axx, "behavior PC1 →", "behavior PC2 →",
             "BEHAVIOR  —  motor policy (belief removed)", "coloured by PHASE  →  one flowing path")

    # (1,0) BELIEF plane coloured by PHASE -> clusters DON'T move (invariant)
    axx = ax[1, 0]
    axx.scatter(belief[idx, 0], belief[idx, 1], s=7, c=phase[idx], cmap="viridis",
                alpha=0.5, edgecolors="none")
    axx.axhline(0, color="#ddd", lw=0.8, zorder=0); axx.axvline(0, color="#ddd", lw=0.8, zorder=0)
    clean_ax(axx, "shape-probe direction →", "colour-probe direction →",
             "BELIEF, coloured by PHASE", "no phase structure → memory is invariant to what it's doing")

    # (1,1) BEHAVIOR manifold coloured by CUE -> no cue structure (invariant)
    axx = ax[1, 1]
    for c in range(4):
        m = cue[idx] == c
        axx.scatter(behav[idx][m, 0], behav[idx][m, 1], s=7, c=CUE_COL[c], alpha=0.5, edgecolors="none")
    clean_ax(axx, "behavior PC1 →", "behavior PC2 →",
             "BEHAVIOR, coloured by CUE", "cues overlap → motor space is invariant to the memory")

    ax[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.9, markerscale=1.5)

    msg = (f"belief axes near-orthogonal:  cos(shape, colour) = {cos_sc:.2f}       "
           f"belief vs behavior:  max cos(belief, position) = {cos_bp:.2f}       "
           f"the memory (WHAT) and the policy (WHERE) live in separate, orthogonal subspaces")
    fig.text(0.5, 0.005, msg, ha="center", fontsize=11, color="#333",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f3f8f5", ec="#41ae76"))
    fig.tight_layout(rect=[0, 0.03, 1, 0.965])
    outp = pathlib.Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=145)
    print(f"[belief-behavior] cos(shape,colour)={cos_sc:.3f}  cos(belief,position)={cos_bp:.3f}")
    print(f"[belief-behavior] wrote {outp}")


if __name__ == "__main__":
    main()
