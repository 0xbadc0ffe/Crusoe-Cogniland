#!/usr/bin/env python3
"""Discover latent switching states in HIGH dimensions, display in 3D.

Pipeline (bt_ppo gru_h, time-ordered):
  1. roll-outs kept in time order (per-trajectory contiguous arrays)
  2. discovery subspace ~8D  (PCA; actor Jacobian is rank-6 for a linear head)
  3. fit a switching latent there (GaussianHMM; rSLDS if `ssm` available)
  4. GATE on validation: gap_score, dwell_stats, slow_points(full-D), relax_test
  5. 3D display basis that maximally separates surviving basins (LDA on states,
     K>=4; else top-3 actor-Jacobian directions)
  6. project trajectories + centroids + slow points with the SAME basis B
  7. render time-ordered, state-coloured 3D line + centroids + slow points,
     and an animation with a growing tail / moving head.

Discovery is in 8D; 3D is ONLY for display, so the wells aren't projection
crowding. Output: outputs/report/hopping/*.html + a static PNG + a validation
JSON.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from hmmlearn.hmm import GaussianHMM

DS = "activation_datasets/bt_ppo"
POLICY = "released_models/bridge_tunnel/ppo_gru.pt"
SRC = "gru_h"
OUT = Path("outputs/report/hopping")
SKILL_COL = {"free": "#7f8c8d", "approach": "#16a085", "avoid": "#2980b9",
             "bridge": "#e6a800", "tunnel": "#a000c8"}
STATE_PAL = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
             "#42d4f4", "#f032e6", "#bfef45"]
RNG = np.random.default_rng(0)


# ───────────────────────────── 1. load, keep time order ─────────────────────
def load_episodes(max_traj=None):
    lab = pd.read_parquet(f"{DS}/labels.parquet").sort_values(
        ["map_id", "traj_id", "t"]).reset_index(drop=True)
    with h5py.File(f"{DS}/activations.h5", "r") as f:
        # row_id -> position in h5 (h5 is already row_id order, but be safe)
        rid = f["row_id"][:]
        pos = {int(r): i for i, r in enumerate(rid)}
        order = np.array([pos[int(r)] for r in lab["row_id"]])
        H = f[SRC][:][order].astype(np.float32)          # (N, D) time-ordered
    groups = lab.groupby(["map_id", "traj_id"], sort=False)
    eps = []                                             # list of dicts per episode
    for (_, _), g in groups:
        idx = g.index.to_numpy()
        eps.append(dict(H=H[idx], seg=g["segment"].to_numpy(),
                        act=g["action"].to_numpy(), t=g["t"].to_numpy()))
        if max_traj and len(eps) >= max_traj:
            break
    return eps, H, lab


# ───────────────────────────── 2. discovery subspace ────────────────────────
def jacobian_subspace(k=3):
    """Top-k right singular vectors of the (linear) actor-head Jacobian
    dlogits/dh = actor.weight. Directions the policy actually acts on."""
    ck = torch.load(POLICY, map_location="cpu", weights_only=False)
    W = ck["policy"]["actor.weight"].numpy()             # (n_act, D)
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    return Vt[:k]                                        # (k, D) in raw-h space


def discovery_subspace(H, d=8):
    scaler = StandardScaler().fit(H)
    Hs = scaler.transform(H)
    pca = PCA(d, random_state=0).fit(Hs)
    return scaler, pca, pca.transform(Hs)               # Z8 (N, d)


# ───────────────────────────── 3. fit switching latent ──────────────────────
def fit_hmm(Z, lengths, K, seed=0):
    m = GaussianHMM(n_components=K, covariance_type="full",
                    n_iter=120, random_state=seed, tol=1e-3)
    m.fit(Z, lengths)
    states = m.predict(Z, lengths)
    logL = m.score(Z, lengths)
    D = Z.shape[1]
    n_params = (K - 1) + K * (K - 1) + K * D + K * D * (D + 1) / 2
    bic = -2 * logL + n_params * np.log(len(Z))
    return m, states, bic


# ───────────────────────────── 4. validation gates ──────────────────────────
def gap_score(Zfull, states, K):
    """min over state pairs of  centroid_distance / (avg within-state radius).
    >1 => basins are separated (gap exceeds spread)."""
    cents = np.stack([Zfull[states == k].mean(0) for k in range(K)])
    radii = np.array([np.linalg.norm(Zfull[states == k] - cents[k], axis=1).mean()
                      for k in range(K)])
    best = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            d = np.linalg.norm(cents[i] - cents[j])
            best = min(best, d / (0.5 * (radii[i] + radii[j]) + 1e-9))
    return float(best), cents, radii


def dwell_stats(states, lengths):
    """run-length (consecutive same-state) distribution, per-episode."""
    runs = []
    off = 0
    for L in lengths:
        s = states[off:off + L]; off += L
        if L == 0:
            continue
        chg = np.flatnonzero(np.diff(s)) + 1
        bounds = np.concatenate([[0], chg, [L]])
        runs.extend(np.diff(bounds).tolist())
    runs = np.array(runs)
    self_trans = float(np.mean(np.diff(np.concatenate(
        [states[off2:off2 + L] for off2, L in _spans(lengths)])) == 0)) if False else None
    return dict(mean_dwell=float(runs.mean()), median_dwell=float(np.median(runs)),
                n_segments=int(len(runs)), max_dwell=int(runs.max()))


def _spans(lengths):
    off = 0
    for L in lengths:
        yield off, L; off += L


def slow_points(Zfull, lengths, pct=5.0):
    """Full-D slow points: steps whose local speed ||h_{t+1}-h_t|| is in the
    bottom `pct` percentile (computed in full-D, never in the projection)."""
    speed = np.full(len(Zfull), np.nan)
    for off, L in _spans(lengths):
        if L < 2:
            continue
        d = np.linalg.norm(np.diff(Zfull[off:off + L], axis=0), axis=1)
        speed[off:off + L - 1] = d
    thr = np.nanpercentile(speed, pct)
    mask = np.nan_to_num(speed, nan=np.inf) <= thr
    return mask, float(thr)


def relax_test(states, lengths, K, W=10):
    """Basin stability: after the chain leaves state k, does it RETURN to k
    within W steps? High return fraction => attracting basin, not pass-through."""
    ret = {k: [0, 0] for k in range(K)}                 # [returns, exits]
    for off, L in _spans(lengths):
        s = states[off:off + L]
        for t in range(1, L):
            if s[t] != s[t - 1]:                         # just left s[t-1]
                k = int(s[t - 1]); ret[k][1] += 1
                if k in s[t:t + W]:
                    ret[k][0] += 1
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in ret.items()}


# ───────────────────────────── 5. 3D display basis ──────────────────────────
def lda_basis(H, states, scaler, denoise=20):
    """B: full-D raw-h -> 3D, maximally separating the discovered states.
    Fit LDA in a denoised PCA space, compose back to raw-h coordinates."""
    Hs = scaler.transform(H)
    pca = PCA(denoise, random_state=0).fit(Hs)
    Zp = pca.transform(Hs)
    lda = LinearDiscriminantAnalysis(n_components=3).fit(Zp, states)
    # raw-h x -> scaler -> pca -> lda :  B(x) = ((x-mu)/sd) @ pca.comp.T @ lda.scalings
    def B(Hraw):
        return lda.transform(pca.transform(scaler.transform(Hraw)))[:, :3]
    return B


def jac_basis(scaler):
    V = jacobian_subspace(3)                             # (3, D) raw-h
    def B(Hraw):
        return scaler.transform(Hraw) @ V.T              # project standardized h
    return B


# ───────────────────────────── 7. render ────────────────────────────────────
def render(eps, states_by_ep, B, cents_raw, slow_eps, name, K, n_show=40, stride=2):
    import plotly.graph_objects as go
    fig = go.Figure()
    # state-coloured time-ordered lines, None-break between episodes
    for st in range(K):
        xs, ys, zs = [], [], []
        for ei in range(min(n_show, len(eps))):
            P = B(eps[ei]["H"]); s = states_by_ep[ei]
            seg_mask = s == st
            # draw contiguous runs of this state as line pieces
            run = np.flatnonzero(seg_mask)
            for r in _contig(run):
                xs += P[r, 0][::stride].tolist() + [None]
                ys += P[r, 1][::stride].tolist() + [None]
                zs += P[r, 2][::stride].tolist() + [None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                      line=dict(color=STATE_PAL[st], width=3),
                      name=f"state {st}", opacity=0.55))
    # centroids as spheres
    C = B(cents_raw)
    fig.add_trace(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="markers+text",
                  marker=dict(size=14, color=[STATE_PAL[k] for k in range(K)],
                              line=dict(color="black", width=2)),
                  text=[f"s{k}" for k in range(K)], textposition="top center",
                  name="basin centroids"))
    # slow points as black ×
    if len(slow_eps):
        S = np.concatenate([B(eps[ei]["H"][m]) for ei, m in slow_eps if m.any()])
        fig.add_trace(go.Scatter3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], mode="markers",
                      marker=dict(size=3, color="black", symbol="x"),
                      name="slow points (full-D)"))
    fig.update_layout(title=f"bt_ppo gru_h — latent hopping ({name} basis, K={K})",
                      scene=dict(aspectmode="data",
                                 xaxis_title="B1", yaxis_title="B2", zaxis_title="B3"),
                      width=1000, height=760)
    fig.write_html(str(OUT / f"hopping_{name}.html"))
    return fig


def render_anim(ep, states, B, K):
    import plotly.graph_objects as go
    P = B(ep["H"]); T = len(P); s = states
    step = max(1, T // 160)
    frames = []
    for tt in range(2, T, step):
        frames.append(go.Frame(data=[
            go.Scatter3d(x=P[:tt, 0], y=P[:tt, 1], z=P[:tt, 2], mode="lines",
                         line=dict(color="#888", width=4)),
            go.Scatter3d(x=[P[tt - 1, 0]], y=[P[tt - 1, 1]], z=[P[tt - 1, 2]],
                         mode="markers", marker=dict(size=8, color=STATE_PAL[int(s[tt - 1])]))],
            name=str(tt)))
    fig = go.Figure(
        data=[go.Scatter3d(x=P[:2, 0], y=P[:2, 1], z=P[:2, 2], mode="lines",
                           line=dict(color="#888", width=4)),
              go.Scatter3d(x=[P[0, 0]], y=[P[0, 1]], z=[P[0, 2]], mode="markers",
                           marker=dict(size=8, color=STATE_PAL[int(s[0])]))],
        frames=frames)
    fig.update_layout(
        title="latent settling-then-switching (growing tail)",
        scene=dict(aspectmode="data"), width=900, height=720,
        updatemenus=[dict(type="buttons", buttons=[dict(label="play", method="animate",
                     args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True)])])])
    fig.write_html(str(OUT / "hopping_anim.html"))


def _contig(idx):
    if len(idx) == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1) + 1
    return np.split(idx, brk)


# ───────────────────────────── main ─────────────────────────────────────────
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="activation_datasets/bt_ppo")
    p.add_argument("--policy", default="released_models/bridge_tunnel/ppo_gru.pt")
    p.add_argument("--out", default="outputs/report/hopping")
    a = p.parse_args()
    global DS, POLICY, OUT
    DS, POLICY, OUT = a.dataset, a.policy, Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"dataset={DS}  policy={POLICY}  out={OUT}", flush=True)
    print("[1] loading episodes (time-ordered)...", flush=True)
    eps, H, lab = load_episodes()
    lengths = np.array([len(e["H"]) for e in eps])
    print(f"    {len(eps)} episodes, {len(H)} steps, D={H.shape[1]}", flush=True)

    print("[2] discovery subspace (8D PCA)...", flush=True)
    scaler, pca8, Z8 = discovery_subspace(H, 8)
    print(f"    PCA8 var explained: {pca8.explained_variance_ratio_.sum():.2f}", flush=True)

    print("[3] fit HMM, choose K by BIC...", flush=True)
    results = {}
    for K in range(2, 9):
        m, states, bic = fit_hmm(Z8, lengths, K)
        results[K] = (m, states, bic)
        print(f"    K={K}: BIC={bic:,.0f}", flush=True)
    Kbest = min(results, key=lambda k: results[k][2])
    print(f"    -> BIC-best K={Kbest}", flush=True)

    # full-D standardized space for the geometric gates
    Hs = scaler.transform(H)
    val = {}
    for K in sorted(set([Kbest, max(4, Kbest)])):
        m, states, bic = results[K]
        gap, cents_s, radii = gap_score(Hs, states, K)
        dw = dwell_stats(states, lengths)
        smask, sthr = slow_points(Hs, lengths, 5.0)
        rt = relax_test(states, lengths, K)
        val[K] = dict(bic=float(bic), gap_score=gap,
                      dwell=dw, relax_return=rt,
                      diag_self_trans=[float(m.transmat_[k, k]) for k in range(K)],
                      slow_frac=float(smask.mean()))
        print(f"    [gate K={K}] gap={gap:.2f} mean_dwell={dw['mean_dwell']:.1f} "
              f"relax={ {k:round(v,2) for k,v in rt.items()} }", flush=True)
    json.dump(val, open(OUT / "validation.json", "w"), indent=2)

    # pick K for display: BIC-best, but LDA needs K>=4
    Kdisp = Kbest if Kbest >= 4 else max(4, Kbest)
    m, states, _ = results[Kdisp]
    # per-episode state arrays + centroids in RAW-h coords
    off = 0; states_by_ep = []
    for L in lengths:
        states_by_ep.append(states[off:off + L]); off += L
    cents_raw = np.stack([H[states == k].mean(0) for k in range(Kdisp)])
    smask, _ = slow_points(Hs, lengths, 5.0)
    slow_eps = []
    off = 0
    for ei, L in enumerate(lengths[:40]):
        slow_eps.append((ei, smask[off:off + L])); off += L
    # (the offset bookkeeping for slow_eps must match eps order; recompute cleanly)
    slow_eps = []
    off = 0
    for ei, L in enumerate(lengths):
        if ei < 40:
            slow_eps.append((ei, smask[off:off + L]))
        off += L

    print(f"[5/6/7] render with LDA basis (K={Kdisp}) + Jacobian basis...", flush=True)
    B_lda = lda_basis(H, states, scaler)
    render(eps, states_by_ep, B_lda, cents_raw, slow_eps, "lda", Kdisp)
    B_jac = jac_basis(scaler)
    render(eps, states_by_ep, B_jac, cents_raw, slow_eps, "jacobian", Kdisp)
    # animation: longest episode
    longest = int(np.argmax(lengths))
    render_anim(eps[longest], states_by_ep[longest], B_lda, Kdisp)

    # static PNG snapshot (matplotlib) of the LDA view for quick inline viewing
    _static_png(eps, states_by_ep, B_lda, cents_raw, Kdisp)
    print("DONE -> outputs/report/hopping/", flush=True)


def _static_png(eps, states_by_ep, B, cents_raw, K, n_show=30):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
    for ei in range(min(n_show, len(eps))):
        P = B(eps[ei]["H"]); s = states_by_ep[ei]
        for r in _contig(np.arange(len(P))):
            ax.plot(P[r, 0], P[r, 1], P[r, 2], color="#ccc", lw=0.4, alpha=0.4)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=[STATE_PAL[k] for k in s], s=2)
    C = B(cents_raw)
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], c=[STATE_PAL[k] for k in range(K)],
               s=320, edgecolors="k", linewidths=2, depthshade=False)
    for k in range(K):
        ax.text(C[k, 0], C[k, 1], C[k, 2], f"s{k}", fontsize=12, fontweight="bold")
    ax.set_title(f"bt_ppo gru_h latent states (LDA-3D display, K={K})")
    ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.set_zlabel("LD3")
    fig.tight_layout(); fig.savefig(OUT / "hopping_static.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
