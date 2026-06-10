#!/usr/bin/env python3
"""Class-mean activation trajectory over episode progress, in a shared PCA space.

For each BTC dataset+source: fit one PCA; for each map category (lakes/balanced/
rocky) and each 5% progress bucket, average the activations of all timesteps in
that (category, bucket) cell and project. Result: 3 lines in PCA space, each a
trajectory from 0% (start) to 100% (end). If they fan apart over progress, the
belief (map type) becomes increasingly represented as the agent gathers info.
Outputs rotatable 3-D plotly (PC1×PC2×PC3) + a static 2-D PNG per dataset.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechinterp.analysis.bundle import ActivationBundle
from mechinterp.analysis import style

NB = 20
PCT = (np.arange(NB) + 0.5) * (100 / NB)            # bucket-centre progress %
CATS = ["rocky", "balanced", "lakes"]
COL = style.CATEGORY_COLORS
SPECS = [("btc_ppo", "gru_h", "PPO · gru_h"), ("btc_dreamer", "rssm_deter", "DreamerV3 · rssm_deter")]


def class_means(name, src, rows=140000):
    b = ActivationBundle(f"activation_datasets/{name}")
    S = b.labels.sample(min(rows, len(b.labels)), random_state=0).reset_index(drop=True)
    prog = (S["t"] / S["ep_len"].clip(lower=1)).to_numpy()
    bucket = np.clip((prog * NB).astype(int), 0, NB - 1)
    cat = S["category"].to_numpy()
    X = b.load_activations(src, S["row_id"])
    pca = PCA(3, random_state=0).fit(X)
    out = {}
    for c in CATS:
        means = np.stack([X[(cat == c) & (bucket == k)].mean(0) for k in range(NB)])
        out[c] = pca.transform(means)              # [NB, 3]
    return out


def static_fig(name, lbl, traj):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for c in CATS:
        p = traj[c]
        seg = np.stack([p[:-1, :2], p[1:, :2]], 1)
        lc = LineCollection(seg, colors=COL[c], linewidths=2.6)
        ax.add_collection(lc)
        ax.scatter(p[:, 0], p[:, 1], c=COL[c], s=18, zorder=3)
        ax.scatter(*p[0, :2], facecolors="white", edgecolors=COL[c], s=70, zorder=5)   # 0%
        ax.scatter(*p[-1, :2], color=COL[c], marker="*", s=190, edgecolors="k", zorder=6, label=c)  # 100%
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_facecolor("#eef3f8"); ax.grid(True, color="white")
    ax.legend(title="map category", fontsize=9)
    ax.set_title(f"{lbl}: class-mean trajectory over progress\n(○ = 0%, ★ = 100%; markers every 5%)")
    fig.tight_layout()
    p = Path(f"outputs/report/figs/classmean_{name}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); return p


def plotly_fig(lbl, traj):
    import plotly.graph_objects as go
    fig = go.Figure()
    for c in CATS:
        p = traj[c]
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines+markers",
            line=dict(color=COL[c], width=6), marker=dict(size=4, color=COL[c]),
            name=c, hovertext=[f"{c}  {int(round(t))}%" for t in PCT], hoverinfo="text"))
        fig.add_trace(go.Scatter3d(x=[p[0, 0]], y=[p[0, 1]], z=[p[0, 2]], mode="markers",
            marker=dict(size=6, color=COL[c], symbol="circle-open"), showlegend=False,
            hovertext=[f"{c} 0% (start)"], hoverinfo="text"))
        fig.add_trace(go.Scatter3d(x=[p[-1, 0]], y=[p[-1, 1]], z=[p[-1, 2]], mode="markers",
            marker=dict(size=8, color=COL[c], symbol="diamond"), showlegend=False,
            hovertext=[f"{c} 100% (end)"], hoverinfo="text"))
    pane = dict(backgroundcolor="#eef3f8", gridcolor="white", showbackground=True)
    fig.update_layout(title=f"{lbl} — class-mean trajectory over episode progress (0→100%)",
                      width=900, height=640, paper_bgcolor="white",
                      scene=dict(xaxis=dict(title="PC1", **pane), yaxis=dict(title="PC2", **pane),
                                 zaxis=dict(title="PC3", **pane)))
    return fig


def main():
    frags = ["<!doctype html><meta charset='utf-8'><title>class-mean trajectories</title>",
             "<style>body{font-family:sans-serif;max-width:980px;margin:0 auto;padding:24px;color:#223}"
             "h2{color:#1b4f72}p.cap{font-size:13.5px;color:#445;background:#f4f7fa;border-left:3px solid #2e86c1;padding:8px 12px}</style>",
             "<h1>Class-mean activation trajectories over episode progress (BTC)</h1>",
             "<p>Each line = the mean activation of one map category, traced from 0% to 100% of the "
             "episode in a shared PCA space. Drag to rotate. ○ = 0% (start), ◆ = 100% (end), markers every "
             "5%. Fanning-apart over progress = the belief (map type) becoming represented as the agent "
             "explores.</p>"]
    for i, (name, src, lbl) in enumerate(SPECS):
        traj = class_means(name, src)
        sp = static_fig(name, lbl, traj)
        sep = np.linalg.norm(traj["lakes"][-1, :2] - traj["rocky"][-1, :2]) / \
              (np.linalg.norm(traj["lakes"][0, :2] - traj["rocky"][0, :2]) + 1e-9)
        print(f"{name}: lakes–rocky centroid separation grew {sep:.1f}x from 0% to 100%")
        fig = plotly_fig(lbl, traj)
        frags.append(f"<h2>{name} · {src}</h2>")
        frags.append(fig.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)))
    out = Path("outputs/report/class_mean_trajectories_3d.html")
    out.write_text("\n".join(frags))
    print("wrote", out, f"({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
