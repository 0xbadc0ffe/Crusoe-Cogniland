#!/usr/bin/env python3
"""Mean activation trajectory over episode progress for the BT dataset (NO classes).

BT has no belief/map-category label (unlike BTC), so instead of fanning class
lines we trace a single aggregate trajectory: for each 5% episode-progress bucket
average the activations of every timestep in that bucket and project into a shared
PCA space.  The result is one line from 0% (start) to 100% (end), coloured by
progress — the overall arc the hidden state sweeps through an episode.

Mirrors class_mean_trajectories.py (the BTC version) but single-line / no class.
Outputs a rotatable 3-D plotly (PC1×PC2×PC3) + a static 2-D PNG per dataset.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechinterp.analysis.bundle import ActivationBundle

NB = 20
PCT = (np.arange(NB) + 0.5) * (100 / NB)            # bucket-centre progress %
SPECS = [("bt_ppo", "gru_h", "PPO · gru_h"),
         ("bt_dreamer", "rssm_deter", "DreamerV3 · rssm_deter")]


def mean_traj(name, src, rows=140000):
    b = ActivationBundle(f"activation_datasets/{name}")
    S = b.labels.sample(min(rows, len(b.labels)), random_state=0).reset_index(drop=True)
    prog = (S["t"] / S["ep_len"].clip(lower=1)).to_numpy()
    bucket = np.clip((prog * NB).astype(int), 0, NB - 1)
    X = b.load_activations(src, S["row_id"])
    pca = PCA(3, random_state=0).fit(X)
    means = np.stack([X[bucket == k].mean(0) for k in range(NB)])
    return pca.transform(means), pca.explained_variance_ratio_[:3] * 100   # [NB,3]


def static_fig(name, lbl, p, evr):
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    seg = np.stack([p[:-1, :2], p[1:, :2]], 1)
    lc = LineCollection(seg, cmap="viridis", linewidths=3,
                        array=PCT[:-1], norm=plt.Normalize(0, 100))
    ax.add_collection(lc)
    sc = ax.scatter(p[:, 0], p[:, 1], c=PCT, cmap="viridis", s=34, zorder=3,
                    vmin=0, vmax=100, edgecolors="k", linewidths=0.4)
    ax.scatter(*p[0, :2], facecolors="white", edgecolors="#3b528b", s=120,
               linewidths=2, zorder=5)                                  # 0%
    ax.scatter(*p[-1, :2], color="#fde725", marker="*", s=240,
               edgecolors="k", zorder=6)                               # 100%
    ax.set_xlabel(f"PC1 ({evr[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({evr[1]:.0f}%)")
    ax.set_facecolor("#eef3f8"); ax.grid(True, color="white")
    fig.colorbar(sc, ax=ax, label="episode progress %")
    ax.set_title(f"{lbl}: mean activation trajectory over progress\n"
                 "(○ = 0%, ★ = 100%; markers every 5%)")
    fig.tight_layout()
    out = Path(f"outputs/report/figs/meantraj_bt_{name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


def plotly_fig(lbl, p, evr):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines+markers",
        line=dict(color="#888", width=4),
        marker=dict(size=5, color=PCT, colorscale="Viridis", cmin=0, cmax=100,
                    colorbar=dict(title="progress %", thickness=12)),
        hovertext=[f"{int(round(t))}%" for t in PCT], hoverinfo="text",
        name="mean trajectory"))
    fig.add_trace(go.Scatter3d(x=[p[0, 0]], y=[p[0, 1]], z=[p[0, 2]], mode="markers",
        marker=dict(size=7, color="#3b528b", symbol="circle-open"),
        hovertext=["0% (start)"], hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[p[-1, 0]], y=[p[-1, 1]], z=[p[-1, 2]], mode="markers",
        marker=dict(size=9, color="#fde725", symbol="diamond",
                    line=dict(color="black", width=1)),
        hovertext=["100% (end)"], hoverinfo="text", showlegend=False))
    pane = dict(backgroundcolor="#eef3f8", gridcolor="white", showbackground=True)
    fig.update_layout(title=f"{lbl} — mean activation trajectory over progress (0→100%)",
                      width=900, height=640, paper_bgcolor="white",
                      scene=dict(xaxis=dict(title=f"PC1 {evr[0]:.0f}%", **pane),
                                 yaxis=dict(title=f"PC2 {evr[1]:.0f}%", **pane),
                                 zaxis=dict(title=f"PC3 {evr[2]:.0f}%", **pane)))
    return fig


def main():
    frags = ["<!doctype html><meta charset='utf-8'><title>BT mean trajectory</title>",
             "<style>body{font-family:sans-serif;max-width:980px;margin:0 auto;"
             "padding:24px;color:#223}h2{color:#1b4f72}</style>",
             "<h1>Mean activation trajectory over episode progress (BT — no classes)</h1>",
             "<p>BT has no belief/map-category label, so this is the single aggregate "
             "mean activation per 5% progress bucket, traced 0→100% in a shared PCA "
             "space and coloured by progress. It shows the overall arc the hidden state "
             "sweeps through an episode (no class fanning, unlike the BTC figure). "
             "○ = 0% (start), ◆ = 100% (end). Drag to rotate.</p>"]
    for i, (name, src, lbl) in enumerate(SPECS):
        p, evr = mean_traj(name, src)
        sp = static_fig(name, lbl, p, evr)
        span = np.linalg.norm(p[-1, :2] - p[0, :2])
        print(f"{name}: PC1-3 evr={evr.round(1)}  start→end PC1×PC2 span={span:.2f}  ({sp})")
        fig = plotly_fig(lbl, p, evr)
        frags.append(f"<h2>{name} · {src}</h2>")
        frags.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
    out = Path("outputs/report/mean_trajectory_progress_bt.html")
    out.write_text("\n".join(frags))
    print("wrote", out, f"({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
