"""Matplotlib figures (Goodfire-inspired). Every function returns a Figure so the
pipeline can both save a PNG and hand it to wandb.Image. Reused for BT and BTC —
functions that need belief/skill are simply not called when those labels are
absent.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from . import style
from .geometry import unit


def _panel(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(True, color=style.GRIDC, linewidth=1.0)


def _legend_handles(colors: dict, order):
    return [plt.Line2D([0], [0], marker="o", ls="", mfc=colors[c], mec="none",
                       ms=8, label=c) for c in order if c in colors]


# ----------------------------------------------------------- categorical scatter
def categorical_scatter(coords, labels, color_kind, *, title="", centroid_path=True,
                        s=6, alpha=0.55):
    """2-D scatter coloured by a discrete label ('category' or 'skill'),
    optionally overlaying the centroid-to-centroid path + smoothing spline."""
    colors = style.colors_for(color_kind)
    order = style.order_for(color_kind)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    labels = np.asarray(labels)
    for c in order:
        m = labels == c
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], s=s, alpha=alpha,
                       c=colors[c], edgecolors="none", rasterized=True, label=c)
    if centroid_path:
        cents = [coords[labels == c].mean(0) for c in order if (labels == c).any()]
        if len(cents) >= 2:
            cp = np.stack(cents)
            _spline(ax, cp)
            ax.plot(cp[:, 0], cp[:, 1], "-", color=style.ACCENT_PATH, lw=2.4, zorder=5)
            ax.scatter(cp[:, 0], cp[:, 1], s=90, facecolors="white",
                       edgecolors=style.ACCENT_PATH, linewidths=2.2, zorder=6)
    _panel(ax)
    ax.set_title(title or f"coloured by {color_kind}")
    ax.legend(handles=_legend_handles(colors, order), loc="best")
    fig.tight_layout()
    return fig


def _spline(ax, pts):
    if len(pts) < 3:
        return
    try:
        from scipy.interpolate import splprep, splev
        tck, _ = splprep([pts[:, 0], pts[:, 1]], k=min(2, len(pts) - 1), s=0)
        u = np.linspace(0, 1, 100)
        xs, ys = splev(u, tck)
        ax.plot(xs, ys, "-", color=style.ACCENT_SPLINE, lw=3.0, alpha=0.9, zorder=4)
    except Exception:
        pass


# ----------------------------------------------------------- continuous scatter
def continuous_scatter(coords, values, *, title="", cmap=style.BELIEF_CMAP,
                       vmin=-1, vmax=1, label="", s=6, alpha=0.6):
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, vmin=vmin,
                    vmax=vmax, s=s, alpha=alpha, edgecolors="none", rasterized=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label or "value")
    _panel(ax); ax.set_title(title)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------- centroids only
def centroid_plot(coords, labels, color_kind, *, title=""):
    colors = style.colors_for(color_kind)
    order = style.order_for(color_kind)
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    for c in order:
        m = labels == c
        if not m.any():
            continue
        ctr = coords[m].mean(0)
        cov = np.cov(coords[m].T)
        ax.scatter(coords[m, 0], coords[m, 1], s=5, alpha=0.18, c=colors[c],
                   edgecolors="none", rasterized=True)
        ax.scatter(*ctr, s=220, c=colors[c], edgecolors="white", linewidths=2, zorder=5)
        _cov_ellipse(ax, ctr, cov, colors[c])
        ax.annotate(c, ctr, color=style.INKC, fontweight="bold", fontsize=11,
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    _panel(ax); ax.set_title(title or f"{color_kind} centroids (1σ)")
    fig.tight_layout()
    return fig


def _cov_ellipse(ax, ctr, cov, color, nstd=1.0):
    from matplotlib.patches import Ellipse
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(np.maximum(vals, 1e-9))
    ax.add_patch(Ellipse(ctr, w, h, angle=theta, facecolor="none",
                         edgecolor=color, lw=1.8, alpha=0.8, zorder=4))


# ----------------------------------------------------------- trajectory paths
def trajectory_paths(coords, df, *, title="PCA trajectories", color_kind=None):
    """Draw each episode's path through the 2-D space, fading by timestep.
    Episodes are taken from df['_traj_key'] (must be aligned to coords)."""
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    keys = df["_traj_key"].to_numpy()
    for k in np.unique(keys):
        m = np.where(keys == k)[0]
        m = m[np.argsort(df["t"].to_numpy()[m])]
        xy = coords[m]
        segs = np.stack([xy[:-1], xy[1:]], 1)
        t = np.linspace(0.15, 1.0, len(segs))
        if color_kind:
            colors = style.colors_for(color_kind)
            lab = df[color_kind].to_numpy()[m]
            cseg = [colors.get(lab[i + 1], "#999999") for i in range(len(segs))]
            ax.add_collection(LineCollection(segs, colors=cseg, linewidths=1.6, alpha=0.8))
        else:
            ax.add_collection(LineCollection(segs, array=t, cmap="viridis",
                                             linewidths=1.6, alpha=0.85))
        ax.scatter(*xy[0], s=30, c="white", edgecolors="k", zorder=5, marker="s")
        ax.scatter(*xy[-1], s=36, c="k", zorder=5, marker="*")
    _panel(ax); ax.set_title(title + "  (□ start, ★ end)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------- entanglement plane
def entanglement_plane(X, belief_dir, skill_dir, df, *, source=""):
    """Project activations onto (belief_axis, skill_axis) and show the SAME cloud
    twice — coloured by category then by skill. If the build/mine split runs along
    the belief axis, belief and skill are entangled."""
    bx, sx = unit(belief_dir), unit(skill_dir)
    u = X @ bx
    v = X @ sx
    cos = float(bx @ sx)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3))
    for ax, kind, ttl in [(axes[0], "category", "belief label"),
                          (axes[1], "skill", "committed skill")]:
        if kind not in df:
            ax.set_visible(False); continue
        colors = style.colors_for(kind); order = style.order_for(kind)
        lab = df[kind].to_numpy()
        for c in order:
            m = lab == c
            if m.any():
                ax.scatter(u[m], v[m], s=7, alpha=0.5, c=colors[c],
                           edgecolors="none", rasterized=True, label=c)
        ax.axhline(0, color="#b9c6d4", lw=0.8); ax.axvline(0, color="#b9c6d4", lw=0.8)
        ax.set_xlabel("belief axis  (lakes − rocky)")
        ax.set_ylabel("skill axis  (build − mine)")
        ax.grid(True, color=style.GRIDC, lw=1.0)
        ax.set_title(f"{source}: coloured by {ttl}")
        ax.legend(handles=_legend_handles(colors, order), loc="best")
    fig.suptitle(f"belief↔skill plane   cos(belief, skill) = {cos:+.3f}",
                 fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------- heatmaps / matrices
def cosine_heatmap(M, rows, cols, *, title="cosine similarity"):
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(cols), 1.6 + 0.7 * len(rows)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    color="white" if abs(M[i, j]) > 0.5 else style.INKC, fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("cosine")
    ax.set_title(title); fig.tight_layout()
    return fig


def confusion(cm, classes, *, title="confusion", normalize=True):
    M = cm.astype(float)
    if normalize:
        M = M / np.clip(M.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(1.4 + 0.9 * len(classes), 1.4 + 0.9 * len(classes)))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "k", fontsize=9)
    ax.grid(False); ax.set_title(title); fig.tight_layout()
    return fig


def probe_bars(metrics: dict, *, title="probe performance"):
    """metrics: {label: accuracy}. Adds a chance line per group if provided."""
    names = list(metrics); vals = [metrics[n] for n in names]
    fig, ax = plt.subplots(figsize=(1.6 + 0.8 * len(names), 4.2))
    bars = ax.bar(names, vals, color="#4c78a8", edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=9, color=style.INKC)
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy / score")
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_title(title); fig.tight_layout()
    return fig
