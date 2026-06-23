#!/usr/bin/env python3
"""Average skill-execution trajectory overlaid on the per-action-mean PCA space.

Per model×dataset: fit the SAME PCA as pca_action_means (same 60k sample, seed 0),
project the 6 per-action mean anchors, then overlay the average activation
trajectory during each skill execution — snippets aligned at the ONSET of a
bridge / tunnel / avoid segment, averaged per offset over events still inside
that segment (NO baseline subtraction: same absolute coordinates as the anchors).
Each average step is annotated with its modal action, so an alternation like
mine→right→mine→right is directly visible as a zigzag between the 'mine' anchor
and the move anchors.

Outputs: outputs/report/figs/skill_traj_on_action_pca.png (2×2, PC1×PC2)
         outputs/report/skill_traj_on_action_pca_3d.html (rotatable, PC1–3)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechinterp.analysis.bundle import ActivationBundle

SPECS = [("bt_ppo", "gru_h", "BT · PPO gru_h"),
         ("btc_ppo", "gru_h", "BTC · PPO gru_h"),
         ("bt_dreamer", "rssm_deter", "BT · Dreamer rssm_deter"),
         ("btc_dreamer", "rssm_deter", "BTC · Dreamer rssm_deter")]
W = 12                       # offsets 0..W-1 from segment onset
MIN_EVENTS = 30              # per-offset support needed to keep the point
PER_TYPE = 2000
SEG_COL = {"bridge": "#e6a800", "tunnel": "#a800e6", "avoid": "#1f5fd0"}
ANCH_COL = {0: "#5b8def", 1: "#5b8def", 2: "#5b8def", 3: "#5b8def",
            4: "#ffd000", 5: "#a800e6"}          # moves muted blue; build/mine loud


def collect(lab, W, per_type=PER_TYPE):
    """Onset-aligned snippets per segment type: row_ids[W] + actions[W] with a
    validity mask (event still inside the same segment at that offset)."""
    ev = {k: [] for k in SEG_COL}
    for _, g in lab.groupby(["map_id", "traj_id"], sort=False):
        g = g.sort_values("t")
        seg = g["segment"].to_numpy(); rid = g["row_id"].to_numpy()
        act = g["action"].to_numpy(); L = len(g)
        for k in ev:
            for i in range(L):
                if seg[i] == k and (i == 0 or seg[i - 1] != k):
                    n = min(W, L - i)
                    rows = np.full(W, -1, np.int64); acts = np.full(W, -1, np.int64)
                    m = np.zeros(W, bool)
                    for j in range(n):
                        if seg[i + j] != k:
                            break
                        rows[j] = rid[i + j]; acts[j] = act[i + j]; m[j] = True
                    if m.sum() >= 2:
                        ev[k].append((rows, acts, m))
    rng = np.random.default_rng(0)
    for k in ev:
        if len(ev[k]) > per_type:
            ev[k] = [ev[k][j] for j in rng.choice(len(ev[k]), per_type, replace=False)]
    return ev


def panel_data(name, src):
    b = ActivationBundle(f"activation_datasets/{name}")
    S = b.labels.sample(min(60000, len(b.labels)), random_state=0)
    ids = np.sort(S["row_id"].to_numpy())
    lab_s = b.labels.set_index("row_id").loc[ids]
    X = b.load_activations(src, ids)
    pca = PCA(3, svd_solver="randomized", random_state=0).fit(X)
    evr = pca.explained_variance_ratio_ * 100
    act = lab_s["action"].to_numpy()
    anames = dict(b.labels[["action", "action_name"]].drop_duplicates().values)
    anchors = {a: (pca.transform(X[act == a].mean(0, keepdims=True))[0], int((act == a).sum()))
               for a in range(6) if (act == a).any()}

    ev = collect(b.labels, W)
    need = np.unique(np.concatenate([r for k in ev for (r, _, _) in ev[k]]))
    need = need[need >= 0]
    Xe = b.load_activations(src, need)
    pos = {int(r): i for i, r in enumerate(need)}
    trajs = {}
    for k, events in ev.items():
        if not events:
            continue
        mean_pts, modal, ns = [], [], []
        for j in range(W):
            rows = [e[0][j] for e in events if e[2][j]]
            acts = [e[1][j] for e in events if e[2][j]]
            if len(rows) < MIN_EVENTS:
                break
            mean_pts.append(Xe[[pos[int(r)] for r in rows]].mean(0))
            modal.append(int(np.bincount(acts, minlength=6).argmax()))
            ns.append(len(rows))
        if mean_pts:
            trajs[k] = (pca.transform(np.stack(mean_pts)), modal, ns)
    return anchors, trajs, evr, anames


def main():
    res = {lbl: panel_data(name, src) for name, src, lbl in SPECS}
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 11.6))
    for ax, (name, src, lbl) in zip(axes.ravel(), SPECS):
        anchors, trajs, evr, anames = res[lbl]
        for a, (p, n) in anchors.items():
            ax.scatter(p[0], p[1], s=300, marker="*", color=ANCH_COL[a],
                       edgecolors="k", linewidths=1.2, zorder=4)
            ax.annotate(anames[a], (p[0], p[1]), xytext=(8, 6), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="#333")
        for k, (P, modal, ns) in trajs.items():
            c = SEG_COL[k]
            ax.plot(P[:, 0], P[:, 1], "-", color=c, lw=2.4, zorder=5,
                    label=f"{k} (n0={ns[0]})")
            ax.scatter(P[:, 0], P[:, 1], s=26, color=c, zorder=6)
            ax.scatter(P[0, 0], P[0, 1], s=110, facecolors="white", edgecolors=c,
                       linewidths=2, zorder=7)
            for j, m in enumerate(modal):
                ax.annotate(anames[m][0], (P[j, 0], P[j, 1]), xytext=(0, -11),
                            textcoords="offset points", fontsize=7.5, ha="center",
                            color=c, fontweight="bold")
        ax.set_title(lbl, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"PC1 ({evr[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]:.1f}%)")
        ax.set_facecolor("#eef3f8"); ax.grid(True, color="white")
        ax.legend(fontsize=8.5, loc="best")
    fig.suptitle("Average skill-execution trajectory on the per-action-mean PCA space\n"
                 "★ = per-action mean anchors · lines = onset-aligned segment averages "
                 f"(○ = onset, ≤{W} steps, letters = modal action at that step)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path("outputs/report/figs/skill_traj_on_action_pca.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)

    # rotatable 3-D
    import plotly.graph_objects as go
    frags = ["<!doctype html><meta charset='utf-8'><title>skill traj on action PCA</title>",
             "<style>body{font-family:sans-serif;max-width:980px;margin:0 auto;padding:24px;color:#223}"
             "h2{color:#1b4f72}</style>",
             "<h1>Average skill-execution trajectory on the per-action-mean PCA space (3-D)</h1>",
             "<p>★/◆ anchors = per-action means; lines = onset-aligned average trajectories "
             "(hover shows step + modal action). Drag to rotate.</p>"]
    for i, (name, src, lbl) in enumerate(SPECS):
        anchors, trajs, evr, anames = res[lbl]
        fig = go.Figure()
        for a, (p, n) in anchors.items():
            fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text",
                marker=dict(size=9 if a >= 4 else 7, color=ANCH_COL[a],
                            symbol="diamond" if a >= 4 else "circle",
                            line=dict(color="black", width=1.5)),
                text=[anames[a]], textposition="top center", textfont=dict(size=10),
                name=f"anchor {anames[a]}", hovertext=[f"anchor {anames[a]} (n={n})"],
                hoverinfo="text"))
        for k, (P, modal, ns) in trajs.items():
            c = SEG_COL[k]
            fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2],
                mode="lines+markers", line=dict(color=c, width=7),
                marker=dict(size=4, color=c), name=f"{k} avg (n0={ns[0]})",
                hovertext=[f"{k} t=+{j} modal={anames[m]} (n={n})"
                           for j, (m, n) in enumerate(zip(modal, ns))], hoverinfo="text"))
            fig.add_trace(go.Scatter3d(x=[P[0, 0]], y=[P[0, 1]], z=[P[0, 2]], mode="markers",
                marker=dict(size=7, color=c, symbol="circle-open"), showlegend=False,
                hovertext=[f"{k} onset"], hoverinfo="text"))
        pane = dict(backgroundcolor="#eef3f8", gridcolor="white", showbackground=True)
        fig.update_layout(title=f"{lbl} (PC1 {evr[0]:.1f}% · PC2 {evr[1]:.1f}% · PC3 {evr[2]:.1f}%)",
                          width=900, height=620, paper_bgcolor="white",
                          scene=dict(xaxis=dict(title="PC1", **pane),
                                     yaxis=dict(title="PC2", **pane),
                                     zaxis=dict(title="PC3", **pane)))
        frags.append(f"<h2>{name} · {src}</h2>")
        frags.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
    out3 = Path("outputs/report/skill_traj_on_action_pca_3d.html")
    out3.write_text("\n".join(frags))
    print("wrote", out3, f"({out3.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
