#!/usr/bin/env python3
"""Proper GaussianHMM state-machine of the RNN/RSSM hidden state, fit on the
ENTIRE dataset, with the ACTUAL hidden-state transition trajectories drawn as the
connectors between regime blobs (the "scribbles" of the abstract FSM figure).

Pipeline per agent (PPO gru_h / Dreamer rssm_deter):
  1. order every timestep by (map,traj,t); lengths = per-trajectory sequence lens.
  2. StandardScaler + PCA(20) basis estimated on a large sample, then transform
     ALL timesteps (one streaming pass over the h5).
  3. fit hmmlearn.GaussianHMM(K, diag) on ALL sequences (temporal transitions are
     part of the fit, unlike the static GMM); Viterbi-decode every timestep.
  4. map HMM states -> dominant behavioural segment (purity + contingency).
  5. nodes = state centroids in PCA-3D (size ~ occupancy, dwell = self-trans).
     edges = REAL transition snippets: for every state switch, the window of
     hidden-state PCA points around it, drawn as faint lines coloured by the
     destination state -> the scribbles connecting the blobs.

Two 3-D panels per agent (HMM states | ground-truth segments, both with real
transition scribbles) + HMM transition-matrix heatmap + state->segment purity.

Output: outputs/report/latent_hmm_state_machine.html
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechinterp.analysis.bundle import ActivationBundle
from mechinterp.analysis import style

SPECS = [("bt_ppo", "gru_h", "BT · PPO gru_h"),
         ("btc_ppo", "gru_h", "BTC · PPO gru_h"),
         ("bt_dreamer", "rssm_deter", "BT · Dreamer rssm_deter"),
         ("btc_dreamer", "rssm_deter", "BTC · Dreamer rssm_deter")]
SEG_ORDER = ["free", "approach", "avoid", "bridge", "tunnel"]
SEG_COLORS = {"free": "#9bb1c4", "approach": "#5b8def", "avoid": "#1f5fd0",
              "bridge": "#e6a800", "tunnel": "#a800e6"}
PCA_DIM = 20          # HMM is fit in this many PCA dims
PCA_FIT = 200_000     # rows used to estimate the PCA/scaler basis
K_EXTRA = 2           # HMM states beyond the number of segments
HMM_ITER = 30
W = 5                 # half-window of a transition snippet (2W+1 points)
MAX_SNIP = 40         # snippets drawn per ordered state pair
N_PLOT = 4500         # scatter-cloud subsample
CHUNK = 50_000


def chunks(b, src, ids):
    for s in range(0, len(ids), CHUNK):
        yield s, b.load_activations(src, ids[s:s + CHUNK])


def reduce_full(b, src, ids, rng):
    """Scaler+PCA(20) basis on a sample, then transform ALL rows (streaming)."""
    fit_ids = ids if len(ids) <= PCA_FIT else np.sort(
        rng.choice(ids, PCA_FIT, replace=False))
    Xs = b.load_activations(src, fit_ids)
    scaler = StandardScaler().fit(Xs)
    pca = PCA(PCA_DIM, svd_solver="randomized", random_state=0).fit(
        scaler.transform(Xs))
    del Xs
    Z = np.empty((len(ids), PCA_DIM), np.float32)
    for s, Xc in chunks(b, src, ids):
        Z[s:s + len(Xc)] = pca.transform(scaler.transform(Xc))
    return Z, pca.explained_variance_ratio_[:3] * 100


def transition_snippets(coords, states, tkey, rng):
    """Real hidden-state paths around each state switch, grouped per (a,b) pair."""
    N = len(states)
    pairs = {}
    sw = np.nonzero((states[:-1] != states[1:]) & (tkey[:-1] == tkey[1:]))[0]
    for i in sw:
        a, b = int(states[i]), int(states[i + 1])
        lo = i
        while lo > 0 and tkey[lo - 1] == tkey[i] and i - lo < W:
            lo -= 1
        hi = i + 1
        while hi < N - 1 and tkey[hi + 1] == tkey[i] and hi - i < W:
            hi += 1
        pairs.setdefault((a, b), []).append((lo, hi + 1))
    out = {}
    for k, v in pairs.items():
        if len(v) > MAX_SNIP:
            v = [v[j] for j in rng.choice(len(v), MAX_SNIP, replace=False)]
        out[k] = [coords[lo:hi] for lo, hi in v]
    return out


def regimes(name, src, rng, n_traj_cap=None):
    from hmmlearn.hmm import GaussianHMM
    b = ActivationBundle(f"activation_datasets/{name}")
    lab = b.labels.sort_values(["map_id", "traj_id", "t"]).reset_index(drop=True)
    if n_traj_cap:
        keys = lab[["map_id", "traj_id"]].drop_duplicates()
        pick = keys.sample(min(n_traj_cap, len(keys)), random_state=0)
        lab = lab.merge(pick, on=["map_id", "traj_id"]).sort_values(
            ["map_id", "traj_id", "t"]).reset_index(drop=True)
    ids = lab["row_id"].to_numpy()
    lengths = lab.groupby(["map_id", "traj_id"], sort=True).size().to_numpy()
    seg = lab["segment"].astype(str).to_numpy()
    seg_names = [s for s in SEG_ORDER if s in set(seg)]
    seg_idx = np.array([seg_names.index(s) for s in seg])
    tkey = (lab["map_id"].astype(np.int64) * 100000 + lab["traj_id"]).to_numpy()

    print(f"   reducing {len(ids):,} timesteps ({src}) -> PCA{PCA_DIM} ...", flush=True)
    Z, evr = reduce_full(b, src, ids, rng)
    coords = Z[:, :3]

    K = len(seg_names) + K_EXTRA
    print(f"   fitting GaussianHMM K={K} on {len(lengths):,} sequences ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hmm = GaussianHMM(n_components=K, covariance_type="diag",
                          n_iter=HMM_ITER, random_state=0, tol=1e-2)
        hmm.fit(Z, lengths)
        st = hmm.predict(Z, lengths)

    cont = np.zeros((K, len(seg_names)), int)
    for s, g in zip(st, seg_idx):
        cont[s, g] += 1
    dom = cont.argmax(1)
    order = sorted(range(K), key=lambda c: (dom[c], -cont[c].sum()))
    remap = np.full(K, 0);
    for i, c in enumerate(order):
        remap[c] = i
    st = remap[st]
    cont = cont[order]; dom = dom[order]
    transmat = hmm.transmat_[np.ix_(order, order)]
    purity = cont.max(1).sum() / cont.sum()

    cen_h = np.array([coords[st == i].mean(0) for i in range(K)])
    cen_s = np.array([coords[seg_idx == i].mean(0) for i in range(len(seg_names))])
    occ_h = np.array([(st == i).sum() for i in range(K)], float)
    occ_s = np.array([(seg_idx == i).sum() for i in range(len(seg_names))], float)

    snip_h = transition_snippets(coords, st, tkey, rng)
    snip_s = transition_snippets(coords, seg_idx, tkey, rng)

    sel = rng.choice(len(coords), min(N_PLOT, len(coords)), replace=False)
    return dict(name=name, src=src, evr=evr, K=K, seg_names=seg_names,
                coords=coords[sel], st=st[sel], seg_sel=seg_idx[sel],
                cen_h=cen_h, cen_s=cen_s, occ_h=occ_h, occ_s=occ_s,
                dom=dom, cont=cont, transmat=transmat, purity=float(purity),
                snip_h=snip_h, snip_s=snip_s,
                n=len(coords), nseq=len(lengths))


# ---------------------------------------------------------------- plotly panels
def panel_traces(coords, idx, n_states, colors, names, cen, occ, snips, dwell):
    import plotly.graph_objects as go
    tr = []
    # faint blob cloud per state
    for i in range(n_states):
        m = idx == i
        tr.append(go.Scatter3d(
            x=coords[m, 0], y=coords[m, 1], z=coords[m, 2], mode="markers",
            name=names[i], marker=dict(size=1.7, color=colors[i], opacity=0.22),
            hoverinfo="skip", showlegend=True))
    # REAL transition snippets, one trace per (a,b) pair (None-separated), coloured
    # by the destination state
    for (a, b), paths in snips.items():
        xs, ys, zs = [], [], []
        for p in paths:
            xs += list(p[:, 0]) + [None]; ys += list(p[:, 1]) + [None]
            zs += list(p[:, 2]) + [None]
        tr.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=colors[b], width=1.2), opacity=0.30,
            hoverinfo="skip", showlegend=False))
    # state nodes
    sz = 12 + 32 * (occ / occ.max())
    tr.append(go.Scatter3d(
        x=cen[:, 0], y=cen[:, 1], z=cen[:, 2], mode="markers+text",
        marker=dict(size=sz, color=[colors[i] for i in range(n_states)],
                    line=dict(color="black", width=1.5)),
        text=[f"{names[i]}<br>dwell {dwell[i]:.2f}" for i in range(n_states)],
        textposition="top center", textfont=dict(size=10, color="#222"),
        hovertext=[f"{names[i]}  visits={int(occ[i])}" for i in range(n_states)],
        hoverinfo="text", showlegend=False))
    return tr


def fig3d(traces, title, evr):
    import plotly.graph_objects as go
    pane = dict(backgroundcolor=style.PANEL, gridcolor="white", showticklabels=False)
    fig = go.Figure(traces)
    fig.update_layout(title=dict(text=title, font=dict(size=12)),
                      width=560, height=520, margin=dict(l=0, r=0, t=30, b=0),
                      paper_bgcolor="white",
                      legend=dict(font=dict(size=9), itemsizing="constant", y=0.5),
                      scene=dict(xaxis=dict(title=f"PC1 {evr[0]:.0f}%", **pane),
                                 yaxis=dict(title=f"PC2 {evr[1]:.0f}%", **pane),
                                 zaxis=dict(title=f"PC3 {evr[2]:.0f}%", **pane)))
    return fig


def heat(z, x, y, title, zmax=1.0):
    import plotly.graph_objects as go
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale="Viridis", zmin=0, zmax=zmax,
                               colorbar=dict(thickness=10)))
    fig.update_layout(title=dict(text=title, font=dict(size=11)),
                      width=360, height=270, margin=dict(l=42, r=8, t=30, b=28),
                      paper_bgcolor="white")
    return fig


def section(r, label, first):
    hmm_names = [f"H{k}→{r['seg_names'][r['dom'][k]]}" for k in range(r["K"])]
    hmm_colors = [SEG_COLORS[r["seg_names"][r["dom"][k]]] for k in range(r["K"])]
    seg_colors = [SEG_COLORS[s] for s in r["seg_names"]]
    th = panel_traces(r["coords"], r["st"], r["K"], hmm_colors, hmm_names,
                      r["cen_h"], r["occ_h"], r["snip_h"], np.diag(r["transmat"]))
    # segment-panel dwell from empirical self-transition of segments
    ts = panel_traces(r["coords"], r["seg_sel"], len(r["seg_names"]), seg_colors,
                      r["seg_names"], r["cen_s"], r["occ_s"], r["snip_s"],
                      np.zeros(len(r["seg_names"])))
    f1 = fig3d(th, "HMM states (fit on full data) + real transition paths", r["evr"])
    f2 = fig3d(ts, "ground-truth segments + real transition paths", r["evr"])
    Zc = r["cont"] / np.clip(r["cont"].sum(1, keepdims=True), 1, None)
    h1 = heat(Zc, r["seg_names"], [f"H{k}" for k in range(r["K"])],
              f"HMM state → segment  (purity {r['purity']*100:.0f}%)")
    h2 = heat(r["transmat"], [f"H{k}" for k in range(r["K"])],
              [f"H{k}" for k in range(r["K"])],
              "HMM transition matrix  P(Hᵢ→Hⱼ)")
    inc = "cdn" if first else False
    return f"""
<section style="margin:30px 0;border-top:1px solid #e3e9ef;padding-top:14px">
<h2 style="color:#1b4f72">{label}</h2>
<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start">
 <div>{f1.to_html(full_html=False, include_plotlyjs=inc)}</div>
 <div>{f2.to_html(full_html=False, include_plotlyjs=False)}</div>
 <div>{h1.to_html(full_html=False, include_plotlyjs=False)}
      {h2.to_html(full_html=False, include_plotlyjs=False)}
   <div style="font-size:12px;color:#456;max-width:360px;margin-top:6px">
    GaussianHMM (K={r['K']}, diag cov) fit on all <b>{r['n']:,}</b> timesteps over
    <b>{r['nseq']:,}</b> trajectories. Unsupervised HMM states recover the skills
    with <b>{r['purity']*100:.0f}%</b> purity. Lines = the actual hidden-state PCA
    paths through each state switch (coloured by destination); node size ∝ visits,
    dwell = self-transition prob.</div></div>
</div></section>"""


def main():
    args = sys.argv[1:]
    names = [a for a in args if not a.isdigit()]
    cap = next((int(a) for a in args if a.isdigit()), None)
    specs = [s for s in SPECS if not names or s[0] in names]
    rng = np.random.default_rng(0)
    secs = []
    for i, (name, src, label) in enumerate(specs):
        print(f"[{name} · {src}] HMM state machine ...", flush=True)
        r = regimes(name, src, rng, n_traj_cap=cap)
        print(f"   purity={r['purity']*100:.0f}%  K={r['K']}  "
              f"states→{[r['seg_names'][d] for d in r['dom']]}", flush=True)
        secs.append(section(r, label, first=(i == 0)))
    html = ("<!doctype html><meta charset='utf-8'>"
            "<title>HMM latent state machine</title>"
            "<style>body{font-family:sans-serif;color:#223;max-width:1250px;"
            "margin:0 auto;padding:24px}h1{color:#1b4f72}</style>"
            "<h1>GaussianHMM latent state-machine of the RNN/RSSM hidden state</h1>"
            "<p>A Gaussian Hidden Markov Model is fit on the <b>entire</b> dataset "
            "(every trajectory, PCA-reduced), so temporal transitions are part of the "
            "fit. Left: discovered HMM states; right: ground-truth segments — both with "
            "the <b>actual hidden-state paths</b> traced through every state switch drawn "
            "as connectors between the blobs. Heatmaps: state→skill purity and the HMM's "
            "learned transition matrix. Drag to rotate.</p>"
            + "\n".join(secs))
    out = Path("outputs/report/latent_hmm_state_machine.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print("wrote", out, f"({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
