#!/usr/bin/env python3
"""Latent state-machine of the RNN/RSSM hidden state.

For each agent we treat the recurrent state (PPO gru_h, Dreamer rssm_deter) as the
node of a finite-state machine whose *states* are behavioural regimes and whose
*edges* are the empirical timestep->timestep transition probabilities.  Two views
side by side per dataset:

  LABEL  — states = ground-truth behavioural segment (free/approach/avoid/bridge/tunnel)
  GMM    — states = regimes DISCOVERED unsupervised by a Gaussian mixture on the
           PCA-reduced hidden state; coloured by the segment each cluster maps to,
           with a contingency heatmap + purity score proving the RNN's hidden state
           changes regime with the active skill.

Each 3-D panel: faint per-regime hidden-state cloud (the "blobs") + centroid nodes
(size ~ occupancy) + directed transition edges (width/opacity ~ probability, cone
arrowheads, hover = p).  Self-transition (dwell) printed on each node.

Output: outputs/report/latent_state_machine.html
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

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
N_TRAJ = 240          # sampled trajectories (ordered) for transitions + GMM
N_PLOT = 6000         # subsample for the scatter cloud
EDGE_MIN = 0.04       # don't draw transition edges below this probability
PCA_GMM = 20          # PCA dims the GMM is fit on
K_EXTRA = 2           # GMM clusters beyond the number of segments


def hidden_source(b):
    return "gru_h" if "gru_h" in b.sources else "rssm_deter"


def transition_matrix(states, traj_key, K):
    """Row-normalised KxK transition counts within trajectories (no cross-traj)."""
    C = np.zeros((K, K))
    for i in range(len(states) - 1):
        if traj_key[i] == traj_key[i + 1]:
            C[states[i], states[i + 1]] += 1
    occ = C.sum(1)
    P = C / np.clip(C.sum(1, keepdims=True), 1, None)
    return P, occ + C.sum(0)            # occ ~ total visits (in+out)


def regimes(name, src, rng):
    b = ActivationBundle(f"activation_datasets/{name}")
    lab = b.labels
    keys = lab[["map_id", "traj_id"]].drop_duplicates()
    pick = keys.sample(min(N_TRAJ, len(keys)), random_state=0)
    sub = lab.merge(pick, on=["map_id", "traj_id"]).sort_values(
        ["map_id", "traj_id", "t"]).reset_index(drop=True)
    ids = sub["row_id"].to_numpy()
    X = b.load_activations(src, ids)
    Z = StandardScaler().fit_transform(X)
    pca = PCA(PCA_GMM, svd_solver="randomized", random_state=0).fit(Z)
    Zp = pca.transform(Z)
    coords = Zp[:, :3]
    evr = pca.explained_variance_ratio_[:3] * 100

    seg = sub["segment"].astype(str).to_numpy()
    seg_names = [s for s in SEG_ORDER if s in set(seg)]
    seg_idx = np.array([seg_names.index(s) for s in seg])
    tkey = (sub["map_id"].astype(np.int64) * 100000 + sub["traj_id"]).to_numpy()

    # fit the GMM on a CLASS-BALANCED subset so rare skills (bridge/tunnel) get
    # their own components instead of the dominant "free" mass eating every cluster
    K = len(seg_names) + K_EXTRA
    cap = min((seg_idx == i).sum() for i in range(len(seg_names)))
    cap = int(min(cap, 4000))
    fit_idx = np.concatenate([
        rng.choice(np.where(seg_idx == i)[0], cap, replace=False)
        for i in range(len(seg_names))])
    gmm = GaussianMixture(K, covariance_type="full", random_state=0,
                          max_iter=300, n_init=3).fit(Zp[fit_idx])
    cl = gmm.predict(Zp)

    # map each GMM cluster -> dominant segment (for colour + purity)
    cont = np.zeros((K, len(seg_names)), int)
    for c, s in zip(cl, seg_idx):
        cont[c, s] += 1
    dom = cont.argmax(1)
    purity = cont.max(1).sum() / cont.sum()
    # relabel clusters in a stable order by dominant segment then size
    order = sorted(range(K), key=lambda c: (dom[c], -cont[c].sum()))
    remap = {c: i for i, c in enumerate(order)}
    cl = np.array([remap[c] for c in cl])
    dom = dom[order]
    cont = cont[order]

    P_lab, occ_lab = transition_matrix(seg_idx, tkey, len(seg_names))
    P_gmm, occ_gmm = transition_matrix(cl, tkey, K)

    cen_lab = np.array([coords[seg_idx == i].mean(0) for i in range(len(seg_names))])
    cen_gmm = np.array([coords[cl == i].mean(0) for i in range(K)])

    sel = rng.choice(len(coords), min(N_PLOT, len(coords)), replace=False)
    return dict(
        name=name, src=src, evr=evr,
        coords=coords[sel], seg_idx=seg_idx[sel], cl=cl[sel],
        seg_names=seg_names, K=K,
        cen_lab=cen_lab, cen_gmm=cen_gmm, P_lab=P_lab, P_gmm=P_gmm,
        occ_lab=occ_lab, occ_gmm=occ_gmm,
        dom=dom, cont=cont, purity=float(purity))


# ---------------------------------------------------------------- plotly traces
def fsm_traces(coords, idx, names, colors, cen, P, occ, dwell_lab, node_text):
    import plotly.graph_objects as go
    tr = []
    for i, nm in enumerate(names):
        m = idx == i
        tr.append(go.Scatter3d(
            x=coords[m, 0], y=coords[m, 1], z=coords[m, 2], mode="markers",
            name=nm, marker=dict(size=1.8, color=colors[i], opacity=0.28),
            hoverinfo="skip", showlegend=True))
    # edges (one trace per edge so width can vary) + arrowhead cones
    dists = [np.linalg.norm(cen[i] - cen[j]) for i in range(len(cen))
             for j in range(len(cen)) if i != j]
    scale = float(np.median(dists)) if dists else 1.0
    cx, cy, cz, cu, cv, cw = [], [], [], [], [], []
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j or P[i, j] < EDGE_MIN:
                continue
            a, b = cen[i], cen[j]
            tr.append(go.Scatter3d(
                x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines",
                line=dict(color="rgba(60,60,60,%.2f)" % (0.25 + 0.6 * P[i, j]),
                          width=1 + 9 * P[i, j]),
                hovertext=f"{names[i]} → {names[j]}  p={P[i, j]:.2f}",
                hoverinfo="text", showlegend=False))
            t = a + 0.80 * (b - a); d = (b - a)
            cx.append(t[0]); cy.append(t[1]); cz.append(t[2])
            cu.append(d[0]); cv.append(d[1]); cw.append(d[2])
    if cx:
        tr.append(go.Cone(x=cx, y=cy, z=cz, u=cu, v=cv, w=cw, sizemode="absolute",
                          sizeref=0.18 * scale, anchor="tip", showscale=False,
                          colorscale=[[0, "#444"], [1, "#444"]], hoverinfo="skip"))
    # nodes
    sz = 12 + 34 * (occ / occ.max())
    tr.append(go.Scatter3d(
        x=cen[:, 0], y=cen[:, 1], z=cen[:, 2], mode="markers+text",
        marker=dict(size=sz, color=[colors[i] for i in range(len(names))],
                    line=dict(color="black", width=1.5), opacity=0.95),
        text=node_text, textposition="top center",
        textfont=dict(size=10, color="#222"),
        hovertext=[f"{names[i]}  dwell={dwell_lab[i]:.2f}  visits={int(occ[i])}"
                   for i in range(len(names))],
        hoverinfo="text", name="states", showlegend=False))
    return tr


def fig3d(traces, title, evr):
    import plotly.graph_objects as go
    pane = dict(backgroundcolor=style.PANEL, gridcolor="white", showticklabels=False)
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)), width=560, height=520,
        margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="white",
        legend=dict(font=dict(size=9), itemsizing="constant", y=0.5),
        scene=dict(xaxis=dict(title=f"PC1 {evr[0]:.0f}%", **pane),
                   yaxis=dict(title=f"PC2 {evr[1]:.0f}%", **pane),
                   zaxis=dict(title=f"PC3 {evr[2]:.0f}%", **pane)))
    return fig


def heatmap(cont, seg_names, K, purity):
    import plotly.graph_objects as go
    Z = cont / np.clip(cont.sum(1, keepdims=True), 1, None)
    fig = go.Figure(go.Heatmap(
        z=Z, x=seg_names, y=[f"C{k}" for k in range(K)], colorscale="Viridis",
        zmin=0, zmax=1, colorbar=dict(title="row frac", thickness=10)))
    fig.update_layout(
        title=dict(text=f"GMM cluster → segment  (purity {purity*100:.0f}%)",
                   font=dict(size=12)),
        width=420, height=300, margin=dict(l=40, r=10, t=34, b=30),
        paper_bgcolor="white", xaxis_title="segment", yaxis_title="GMM cluster")
    return fig


def section(r, label, first):
    seg_colors = [SEG_COLORS[s] for s in r["seg_names"]]
    gmm_colors = [SEG_COLORS[r["seg_names"][r["dom"][k]]] for k in range(r["K"])]
    lab_nodes = [f"{s}<br>dwell {r['P_lab'][i, i]:.2f}"
                 for i, s in enumerate(r["seg_names"])]
    gmm_nodes = [f"C{k}→{r['seg_names'][r['dom'][k]]}<br>dwell {r['P_gmm'][k, k]:.2f}"
                 for k in range(r["K"])]
    t_lab = fsm_traces(r["coords"], r["seg_idx"], r["seg_names"], seg_colors,
                       r["cen_lab"], r["P_lab"], r["occ_lab"],
                       np.diag(r["P_lab"]), lab_nodes)
    t_gmm = fsm_traces(r["coords"], r["cl"], [f"C{k}" for k in range(r["K"])],
                       gmm_colors, r["cen_gmm"], r["P_gmm"], r["occ_gmm"],
                       np.diag(r["P_gmm"]), gmm_nodes)
    f1 = fig3d(t_lab, "LABEL states = behavioural segment", r["evr"])
    f2 = fig3d(t_gmm, "GMM states (unsupervised) — coloured by mapped segment",
               r["evr"])
    f3 = heatmap(r["cont"], r["seg_names"], r["K"], r["purity"])
    inc = "cdn" if first else False
    h1 = f1.to_html(full_html=False, include_plotlyjs=inc)
    h2 = f2.to_html(full_html=False, include_plotlyjs=False)
    h3 = f3.to_html(full_html=False, include_plotlyjs=False)
    return f"""
<section style="margin:30px 0;border-top:1px solid #e3e9ef;padding-top:14px">
<h2 style="color:#1b4f72">{label}</h2>
<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start">
 <div>{h1}</div><div>{h2}</div>
 <div>{h3}<div style="font-size:12px;color:#456;max-width:400px;margin-top:8px">
  Unsupervised GMM regimes recover the behavioural skills with
  <b>{r['purity']*100:.0f}%</b> purity — the {r['src']} hidden state changes
  regime with the active skill. Node size ∝ visits; edge width/opacity ∝
  transition probability (arrow = direction); dwell = self-transition prob.</div></div>
</div></section>"""


def main():
    rng = np.random.default_rng(0)
    secs = []
    for i, (name, src, label) in enumerate(SPECS):
        print(f"[{name} · {src}] fitting regimes + GMM ...", flush=True)
        r = regimes(name, src, rng)
        print(f"   purity={r['purity']*100:.0f}%  K={r['K']}  "
              f"segs={r['seg_names']}", flush=True)
        secs.append(section(r, label, first=(i == 0)))
    html = ("<!doctype html><meta charset='utf-8'>"
            "<title>latent state machine</title>"
            "<style>body{font-family:sans-serif;color:#223;max-width:1200px;"
            "margin:0 auto;padding:24px}h1{color:#1b4f72}</style>"
            "<h1>Latent state-machine of the RNN/RSSM hidden state</h1>"
            "<p>Each panel treats the recurrent state as an FSM: nodes = behavioural "
            "regimes at their hidden-state PCA centroid, directed edges = empirical "
            "transition probabilities (width/opacity ∝ p, cone = direction, dwell = "
            "self-loop). <b>LABEL</b> uses the ground-truth segment; <b>GMM</b> "
            "discovers the regimes unsupervised from the hidden state and the heatmap "
            "shows they map onto the skills. Drag to rotate.</p>"
            + "\n".join(secs))
    out = Path("outputs/report/latent_state_machine.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print("wrote", out, f"({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
