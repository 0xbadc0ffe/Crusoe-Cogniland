#!/usr/bin/env python3
"""Event-aligned average latent trajectories (peri-event averages).

BT : align snippets on the ONSET of each bridge / tunnel / avoid segment,
     window [-W, +W] steps; average across many events.
BTC: align on the COMMITMENT step (committed_now), window [-W, +W]; split build vs mine.

Each snippet is projected with the per-(model×dataset) PCA and baseline-subtracted
(its own pre-event mean) so the plot shows the event-LOCKED excursion, not the
per-episode offset. Reveals whether segments / commitments have consistent PCA-space
structure. Writes outputs/report/event_aligned.html (figures embedded).
"""
from __future__ import annotations
import base64, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechinterp.analysis.bundle import ActivationBundle

BT_COL = {"bridge": "#ffd000", "tunnel": "#a800e6", "avoid": "#1f5fd0"}
BTC_COL = {"build": "#ffd000", "mine": "#a800e6"}
OUT = Path("outputs/report/figs"); OUT.mkdir(parents=True, exist_ok=True)
parts = ["<!doctype html><meta charset='utf-8'><title>event-aligned latents</title>",
         "<style>body{font-family:sans-serif;max-width:1050px;margin:0 auto;padding:24px;color:#223}"
         "h2{color:#1b4f72;border-bottom:1px solid #cdd8e3;margin-top:40px}img{width:100%;border:1px solid #e1e7ee;border-radius:6px}"
         "figcaption{font-size:13.5px;color:#445;background:#f4f7fa;border-left:3px solid #2e86c1;padding:8px 12px;margin:6px 0 22px}</style>",
         "<h1>Event-aligned average latent trajectories</h1>",
         "<p>Snippets are aligned on the event (BT: onset of a bridge/tunnel/avoid segment; "
         "BTC: the commitment step), baseline-subtracted (each snippet minus its own pre-event "
         "mean), and <b>averaged across many events</b>. So we see the event-locked excursion in "
         "PCA space, not single-trajectory noise. ●=event time (t=0), ○=window start, arrow=time.</p>"]


def collect_bt(lab, W, per_type=1500):
    ev = {k: [] for k in BT_COL}
    for _, g in lab.groupby(["map_id", "traj_id"], sort=False):
        g = g.sort_values("t"); seg = g["segment"].to_numpy(); rid = g["row_id"].to_numpy(); L = len(g)
        for k in ev:
            for i in range(L):
                if seg[i] == k and (i == 0 or seg[i - 1] != k) and i - W >= 0 and i + W < L:
                    ev[k].append(rid[i - W:i + W + 1])
    rng = np.random.default_rng(0)
    for k in ev:
        a = ev[k]
        if len(a) > per_type:
            a = [a[j] for j in rng.choice(len(a), per_type, replace=False)]
        ev[k] = np.array(a, dtype=np.int64) if a else np.zeros((0, 2 * W + 1), np.int64)
    return ev


def collect_btc(lab, W, per_type=1500):
    ev = {k: [] for k in BTC_COL}
    for _, g in lab.groupby(["map_id", "traj_id"], sort=False):
        g = g.sort_values("t"); fc = g["final_commit"].iloc[0]
        if fc not in BTC_COL:
            continue
        cn = g["committed_now"].to_numpy(); rid = g["row_id"].to_numpy(); L = len(g)
        idx = np.where(cn)[0]
        if len(idx) and idx[0] - W >= 0 and idx[0] + W < L:
            ev[fc].append(rid[idx[0] - W:idx[0] + W + 1])
    rng = np.random.default_rng(0)
    for k in ev:
        a = ev[k]
        if len(a) > per_type:
            a = [a[j] for j in rng.choice(len(a), per_type, replace=False)]
        ev[k] = np.array(a, dtype=np.int64) if a else np.zeros((0, 2 * W + 1), np.int64)
    return ev


def averages(bundle, source, pca, events, W):
    """For each event type return mean[L,3] and sem[L,3] of baseline-subtracted PCA coords."""
    allids = np.unique(np.concatenate([v.ravel() for v in events.values() if len(v)]))
    X = bundle.load_activations(source, allids)
    coords = pca.transform(X)
    pos = {int(r): i for i, r in enumerate(allids)}
    out = {}
    for k, wins in events.items():
        if not len(wins):
            continue
        snips = np.stack([coords[[pos[int(r)] for r in w]] for w in wins])  # [n, L, 3]
        snips = snips - snips[:, :W, :].mean(1, keepdims=True)              # baseline subtract
        out[k] = (snips.mean(0), snips.std(0) / np.sqrt(len(snips)), len(snips))
    return out


def figure(name, source, avg, W, colmap, title):
    rel = np.arange(-W, W + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # 2-D mean path
    for k, (m, s, n) in avg.items():
        c = colmap[k]
        seg = np.stack([m[:-1, :2], m[1:, :2]], 1)
        ax[0].add_collection(LineCollection(seg, colors=c, linewidths=2.6))
        ax[0].scatter(*m[0, :2], facecolors="white", edgecolors=c, s=40, zorder=4)
        ax[0].scatter(*m[W, :2], color=c, s=110, zorder=5, label=f"{k} (n={n})")
        ax[0].annotate("", xy=m[-1, :2], xytext=m[-2, :2],
                       arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].set_facecolor("#eef3f8")
    ax[0].grid(True, color="white"); ax[0].legend(fontsize=8, loc="best")
    ax[0].set_title("mean latent path (baseline-subtracted)\n● = event (t=0)")
    # PC vs relative time
    for pc, axi in [(0, ax[1]), (1, ax[2])]:
        for k, (m, s, n) in avg.items():
            c = colmap[k]
            axi.plot(rel, m[:, pc], color=c, lw=2.2, label=k)
            axi.fill_between(rel, m[:, pc] - s[:, pc], m[:, pc] + s[:, pc], color=c, alpha=0.18)
        axi.axvline(0, color="#888", ls="--", lw=1)
        axi.set_xlabel("steps relative to event"); axi.set_ylabel(f"Δ PC{pc+1}")
        axi.set_facecolor("#eef3f8"); axi.grid(True, color="white"); axi.set_title(f"Δ PC{pc+1} vs event time")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT / f"event_{name}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


def plotly3d(avg, W, colmap, title):
    """Rotatable 3-D plotly of the event-aligned mean paths (PC1×PC2×PC3)."""
    import plotly.graph_objects as go
    rel = np.arange(-W, W + 1)
    fig = go.Figure()
    for k, (m, s, n) in avg.items():
        c = colmap[k]
        fig.add_trace(go.Scatter3d(
            x=m[:, 0], y=m[:, 1], z=m[:, 2], mode="lines", line=dict(color=c, width=7),
            name=f"{k} (n={n})", hovertext=[f"{k}  t={r:+d}" for r in rel], hoverinfo="text"))
        fig.add_trace(go.Scatter3d(x=[m[0, 0]], y=[m[0, 1]], z=[m[0, 2]], mode="markers",
            marker=dict(color=c, size=4, symbol="circle-open"), showlegend=False,
            hovertext=[f"{k} start (t=-{W})"], hoverinfo="text"))
        fig.add_trace(go.Scatter3d(x=[m[W, 0]], y=[m[W, 1]], z=[m[W, 2]], mode="markers",
            marker=dict(color=c, size=9, symbol="circle", line=dict(color="black", width=1)),
            showlegend=False, hovertext=[f"{k} EVENT (t=0)"], hoverinfo="text"))
        fig.add_trace(go.Scatter3d(x=[m[-1, 0]], y=[m[-1, 1]], z=[m[-1, 2]], mode="markers",
            marker=dict(color=c, size=6, symbol="diamond"), showlegend=False,
            hovertext=[f"{k} end (t=+{W})"], hoverinfo="text"))
    pane = dict(backgroundcolor="#eef3f8", gridcolor="white", showbackground=True)
    fig.update_layout(title=title, width=900, height=640, paper_bgcolor="white",
                      legend=dict(title="event (●=t0, ○=start, ◆=end)"),
                      scene=dict(xaxis=dict(title="ΔPC1", **pane), yaxis=dict(title="ΔPC2", **pane),
                                 zaxis=dict(title="ΔPC3", **pane)))
    return fig


def embed(p, cap):
    b = base64.b64encode(Path(p).read_bytes()).decode()
    parts.append(f"<figure><img src='data:image/png;base64,{b}'><figcaption>{cap}</figcaption></figure>")


SPECS = [
    ("bt_ppo", "gru_h", "bt", 8), ("bt_dreamer", "rssm_deter", "bt", 8),
    ("btc_ppo", "gru_h", "btc", 12), ("btc_dreamer", "rssm_deter", "btc", 12),
]


def main():
    frags3d = ["<!doctype html><meta charset='utf-8'><title>event-aligned latents (3D)</title>",
               "<style>body{font-family:sans-serif;max-width:980px;margin:0 auto;padding:24px;color:#223}"
               "h2{color:#1b4f72}p.cap{font-size:13.5px;color:#445;background:#f4f7fa;"
               "border-left:3px solid #2e86c1;padding:8px 12px}</style>",
               "<h1>Event-aligned average latent trajectories — rotatable 3-D</h1>",
               "<p>Mean PCA path (PC1×PC2×PC3) of baseline-subtracted snippets aligned on the event. "
               "Drag to rotate. ●=event (t=0), ○=window start, ◆=end.</p>"]
    for name, src, variant, W in SPECS:
        b = ActivationBundle(f"activation_datasets/{name}")
        samp = b.labels.sample(min(15000, len(b.labels)), random_state=0)
        pca = PCA(3, random_state=0).fit(b.load_activations(src, samp["row_id"]))
        if variant == "bt":
            ev = collect_bt(b.labels, W); colmap = BT_COL
            ttl = f"{name} · {src} — bridge/tunnel/avoid segment onsets (±{W} steps)"
            cap = (f"<b>{name}</b>: average latent path around the onset of each segment type "
                   f"(n per type in legend). If the colored mean paths separate / trace consistent "
                   f"excursions, the segment is encoded in {src}; if they overlap near the origin, it is not.")
        else:
            ev = collect_btc(b.labels, W); colmap = BTC_COL
            ttl = f"{name} · {src} — commitment-aligned (±{W} steps; build vs mine)"
            cap = (f"<b>{name}</b>: average latent path from {W} steps before to {W} after the "
                   f"commitment. Separation of build (yellow) vs mine (purple) around t=0 indicates "
                   f"commitment structure in {src}.")
        avg = averages(b, src, pca, ev, W)
        p = figure(name, src, avg, W, colmap, ttl)
        ns = ", ".join(f"{k}:{v[2]}" for k, v in avg.items())
        print(f"{name}: events {ns}")
        parts.append(f"<h2>{name} · {src}</h2>")
        embed(p, cap)
        # rotatable 3-D plotly (self-contained: inline plotly.js once)
        fig3d = plotly3d(avg, W, colmap, ttl)
        inc = (len(frags3d) <= 4)
        frags3d.append(f"<h2>{name} · {src}</h2><p class='cap'>{cap}</p>")
        frags3d.append(fig3d.to_html(full_html=False, include_plotlyjs=(True if inc else False)))
    out = Path("outputs/report/event_aligned.html")
    out.write_text("\n".join(parts))
    out3 = Path("outputs/report/event_aligned_3d.html")
    out3.write_text("\n".join(frags3d))
    print("wrote", out, f"({out.stat().st_size/1e6:.1f} MB)")
    print("wrote", out3, f"({out3.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
