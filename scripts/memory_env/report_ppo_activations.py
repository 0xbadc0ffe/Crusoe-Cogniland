#!/usr/bin/env python
"""HTML report of PPO+GRU activations on the MemoryEnv.

For each solved model (activations.npz from build_ppo_activations.py):
  - PCA (2D) of the GRU hidden state, coloured by cue shape / cue colour / maze
    phase / timestep.
  - UMAP (2D) with the same colourings.
  - Average activation over an episode: per-cue mean trajectory in PCA space,
    a mean-activation heatmap (timestep x top-variance dims), and per-timestep
    linear-probe accuracy for shape & colour (what the memory carries, when).
All figures are embedded (base64) into one self-contained HTML file.
"""
from __future__ import annotations

import argparse
import base64
import io
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

PHASE_NAMES = ["pre-cue", "cue-room", "pre-branch", "branch", "post", "door"]


def _b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def dual_ridge_acc(X, y, rng, lam=10.0, frac=0.7):
    n = X.shape[0]
    if n < 40 or y.min() == y.max():
        return np.nan
    perm = rng.permutation(n); X, y = X[perm], y[perm].astype(np.float64)
    ntr = int(n * frac); Xtr, Xte, ytr, yte = X[:ntr], X[ntr:], y[:ntr], y[ntr:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    yb = ytr.mean(); K = Xtr @ Xtr.T
    alpha = np.linalg.solve(K + lam * np.eye(ntr), ytr - yb)
    pred = Xte @ (Xtr.T @ alpha) + yb
    return float(np.mean((pred > 0.5) == (yte > 0.5)))


def scatter_panel(emb, D, title):
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    # shape
    for v, c, lab in [(0, "#1f77b4", "up"), (1, "#d62728", "down")]:
        m = D["shape"] == v
        ax[0].scatter(emb[m, 0], emb[m, 1], s=3, c=c, alpha=0.4, label=lab)
    ax[0].set_title("cue SHAPE"); ax[0].legend(markerscale=3, fontsize=8)
    # colour
    for v, c, lab in [(0, "#2ca02c", "green"), (1, "#1f77b4", "blue")]:
        m = D["colour"] == v
        ax[1].scatter(emb[m, 0], emb[m, 1], s=3, c=c, alpha=0.4, label=lab)
    ax[1].set_title("cue COLOUR"); ax[1].legend(markerscale=3, fontsize=8)
    # phase
    sc = ax[2].scatter(emb[:, 0], emb[:, 1], s=3, c=D["phase"], cmap="viridis", alpha=0.5)
    ax[2].set_title("maze PHASE");
    cb = fig.colorbar(sc, ax=ax[2], ticks=range(6)); cb.ax.set_yticklabels(PHASE_NAMES, fontsize=7)
    # timestep
    sc = ax[3].scatter(emb[:, 0], emb[:, 1], s=3, c=D["t"], cmap="plasma", alpha=0.5)
    ax[3].set_title("TIMESTEP"); fig.colorbar(sc, ax=ax[3])
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=13); fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def avg_panel(feat, D, pca, title):
    fig = plt.figure(figsize=(18, 4.6))
    # (1) per-cue mean trajectory in PCA space
    ax1 = fig.add_subplot(1, 3, 1)
    proj = pca.transform((feat - feat.mean(0)) / (feat.std(0) + 1e-6))
    cue_names = ["green_up", "blue_up", "green_down", "blue_down"]
    cols = ["#2ca02c", "#1f77b4", "#98df8a", "#aec7e8"]
    tmax = int(D["t"].max())
    for ct in range(4):
        xs, ys = [], []
        for t in range(tmax + 1):
            m = (D["cue_type"] == ct) & (D["t"] == t)
            if m.sum() >= 3:
                xs.append(proj[m, 0].mean()); ys.append(proj[m, 1].mean())
        if xs:
            ax1.plot(xs, ys, "-", color=cols[ct], lw=1.5, label=cue_names[ct])
            ax1.scatter(xs[0], ys[0], color=cols[ct], marker="o", s=30, zorder=5)
            ax1.scatter(xs[-1], ys[-1], color=cols[ct], marker="s", s=30, zorder=5)
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.legend(fontsize=7)
    ax1.set_title("mean trajectory / cue (o=start, □=end)")
    # (2) mean-activation heatmap: timestep x top-variance dims
    ax2 = fig.add_subplot(1, 3, 2)
    topd = np.argsort(feat.var(0))[::-1][:40]
    H = np.stack([feat[D["t"] == t][:, topd].mean(0) if (D["t"] == t).sum() else np.zeros(len(topd))
                  for t in range(tmax + 1)])
    im = ax2.imshow(H.T, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                    vmin=-np.abs(H).max(), vmax=np.abs(H).max())
    ax2.set_xlabel("timestep"); ax2.set_ylabel("top-40 variance units")
    ax2.set_title("mean activation over episode"); fig.colorbar(im, ax=ax2)
    # (3) per-timestep probe accuracy (shape / colour)
    ax3 = fig.add_subplot(1, 3, 3)
    rng = np.random.default_rng(0)
    ts, sh, co, vis = [], [], [], []
    for t in range(tmax + 1):
        m = D["t"] == t
        if m.sum() < 40:
            continue
        ts.append(t)
        sh.append(dual_ridge_acc(feat[m], D["shape"][m], rng))
        co.append(dual_ridge_acc(feat[m], D["colour"][m], rng))
        vis.append(D["cue_vis"][m].mean())
    ax3.fill_between(ts, 0, vis, color="0.85", step="mid", label="cue in view")
    ax3.plot(ts, sh, "-o", ms=3, color="#1f77b4", label="shape probe")
    ax3.plot(ts, co, "-o", ms=3, color="#d62728", label="colour probe")
    ax3.axhline(0.5, ls="--", c="k", lw=0.8)
    ax3.set_ylim(0.3, 1.03); ax3.set_xlabel("timestep"); ax3.set_ylabel("probe acc")
    ax3.set_title("what the memory carries, when"); ax3.legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=13); fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _embed_figs(feat, D, name, tag, umap_n=8000):
    Z = (feat - feat.mean(0)) / (feat.std(0) + 1e-6)
    pca = PCA(n_components=2).fit(Z)
    pca_emb = pca.transform(Z)
    rng = np.random.default_rng(0)
    sub = rng.choice(feat.shape[0], min(umap_n, feat.shape[0]), replace=False)
    um = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.1, random_state=0)
    um_emb = um.fit_transform(Z[sub])
    Dsub = {k: v[sub] for k, v in D.items()}
    return {
        f"{tag}_pca": _b64(scatter_panel(pca_emb, D, f"{name} — PCA "
                           f"({pca.explained_variance_ratio_.sum():.0%} var)")),
        f"{tag}_umap": _b64(scatter_panel(um_emb, Dsub, f"{name} — UMAP")),
        f"{tag}_avg": _b64(avg_panel(feat, D, pca, f"{name} — average over an episode")),
    }


def section(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    D = {k: d[k] for k in ("shape", "colour", "phase", "t", "cue_type", "cue_vis", "ep_success")}
    cue = str(d["cue"])
    obs = d["obs_embed"].astype(np.float32)
    feat = d["feat"].astype(np.float32)
    imgs = {}
    imgs.update(_embed_figs(obs, D, f"{cue} · OBSERVATION embedding (encoder, pre-GRU)", "obs"))
    imgs.update(_embed_figs(feat, D, f"{cue} · GRU HIDDEN (recurrent memory)", "mem"))
    meta = dict(cue=cue, n_steps=feat.shape[0], dim_mem=feat.shape[1], dim_obs=obs.shape[1],
                success=float(D["ep_success"].mean()))
    return meta, imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True, help="activations.npz files")
    ap.add_argument("--out", default="outputs/ppo_activation_report.html")
    a = ap.parse_args()
    parts = []
    for path in a.npz:
        print(f"[report] processing {path}", flush=True)
        meta, imgs = section(path)
        parts.append((meta, imgs))

    rows = ""
    for meta, imgs in parts:
        rows += f"""
        <h2>{meta['cue']} model &nbsp;<small>({meta['n_steps']:,} live steps ·
             obs-embed {meta['dim_obs']}d · GRU hidden {meta['dim_mem']}d ·
             door success {meta['success']:.2f})</small></h2>
        <h3>&#9312; Observation embedding &mdash; encoder output, pre-GRU (what the agent sees <i>now</i>; no memory)</h3>
        <img src="data:image/png;base64,{imgs['obs_pca']}"/>
        <img src="data:image/png;base64,{imgs['obs_umap']}"/>
        <img src="data:image/png;base64,{imgs['obs_avg']}"/>
        <h3>&#9313; GRU hidden state &mdash; the recurrent memory (carries the cue after it leaves view)</h3>
        <img src="data:image/png;base64,{imgs['mem_pca']}"/>
        <img src="data:image/png;base64,{imgs['mem_umap']}"/>
        <img src="data:image/png;base64,{imgs['mem_avg']}"/>
        <hr/>"""

    html = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>PPO+GRU activation report — MemoryEnv</title>
    <style>
      body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1200px;
            margin:24px auto;color:#222;padding:0 16px}}
      h1{{border-bottom:2px solid #41ae76;padding-bottom:6px}}
      h2{{margin-top:28px;color:#2b6}} small{{color:#888;font-weight:normal;font-size:0.6em}}
      img{{max-width:100%;border:1px solid #eee;border-radius:6px;margin:6px 0}}
      hr{{border:none;border-top:1px solid #eee;margin:28px 0}}
      .intro{{background:#f6fbf8;border-left:4px solid #41ae76;padding:10px 16px;border-radius:4px}}
    </style></head><body>
    <h1>PPO+GRU activation report — MemoryEnv (random 50/50 doors)</h1>
    <div class="intro">
      Two representations of the solved recurrent-PPO agents, collected over greedy episodes with
      mixed cues and random door layouts: <b>&#9312; the observation embedding</b> (the encoder output
      <i>before</i> the GRU &mdash; what the agent sees at the current step, with no memory) and
      <b>&#9313; the GRU hidden state</b> (the recurrent memory the actor reads). Contrast them: the
      obs embedding should encode the cue <i>only while it is in view</i>, whereas the GRU hidden
      <i>carries</i> it to the door. These models solve the full task (shape&rarr;branch AND
      colour&rarr;door), so both features are represented <i>and</i> used &mdash; the substrate for
      steering. For each: PCA &amp; UMAP coloured by cue shape/colour, maze phase and timestep; plus
      per-cue mean trajectories, a mean-activation heatmap, and per-timestep shape/colour probe accuracy.
    </div>
    {rows}
    </body></html>"""
    outp = pathlib.Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(html)
    print(f"[report] wrote {outp}  ({outp.stat().st_size//1024} KB)", flush=True)


if __name__ == "__main__":
    main()
