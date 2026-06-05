"""Matplotlib figures (Goodfire-inspired). Every function returns a Figure so the
pipeline can both save a PNG and hand it to wandb.Image. Reused for BT and BTC —
functions that need belief/skill are simply not called when those labels are
absent.

Scatter / centroid / trajectory plots render in 3-D by default (set ``dims=2``
for the flat version); the interactive plotly versions in wandb_io are rotatable.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from . import style
from .geometry import unit

_ELEV, _AZIM = 18, -60


# --------------------------------------------------------------------- axes
def _new_axes(d, figsize=(6.6, 5.6)):
    fig = plt.figure(figsize=figsize)
    if d >= 3:
        ax = fig.add_subplot(111, projection="3d")
        _style3d(ax)
    else:
        ax = fig.add_subplot(111)
        _panel(ax)
    return fig, ax


def _panel(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(True, color=style.GRIDC, linewidth=1.0)


def _style3d(ax):
    ax.view_init(elev=_ELEV, azim=_AZIM)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(style.PANEL)
        axis.pane.set_alpha(1.0)
        axis.pane.set_edgecolor("#c6d2de")
        axis._axinfo["grid"]["color"] = (1, 1, 1, 1)
        axis._axinfo["grid"]["linewidth"] = 1.0
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel("c1", labelpad=-12); ax.set_ylabel("c2", labelpad=-12)
    ax.set_zlabel("c3", labelpad=-12)


def _scatter(ax, pts, d, **kw):
    """ax.scatter with d (2 or 3) positional columns; drops 3-D-only kwargs in 2-D."""
    cols = [pts[:, i] for i in range(d)]
    if d >= 3:
        kw.setdefault("depthshade", False)
    else:
        kw.pop("depthshade", None)
    return ax.scatter(*cols, **kw)


def _plot_line(ax, pts, d, **kw):
    ax.plot(*[pts[:, i] for i in range(d)], **kw)


def _legend_handles(colors: dict, order):
    return [plt.Line2D([0], [0], marker="o", ls="", mfc=colors[c], mec="none",
                       ms=8, label=c) for c in order if c in colors]


# ----------------------------------------------------------- categorical scatter
def categorical_scatter(coords, labels, color_kind, *, title="", centroid_path=True,
                        s=6, alpha=0.55, dims=3):
    colors = style.colors_for(color_kind)
    order = style.order_for(color_kind)
    d = min(dims, coords.shape[1])
    fig, ax = _new_axes(d)
    labels = np.asarray(labels)
    for c in order:
        m = labels == c
        if m.any():
            _scatter(ax, coords[m], d, s=s, alpha=alpha, c=colors[c],
                     edgecolors="none", rasterized=True, label=c)
    if centroid_path:
        cents = [coords[labels == c, :d].mean(0) for c in order if (labels == c).any()]
        if len(cents) >= 2:
            cp = np.stack(cents)
            _spline(ax, cp, d)
            _plot_line(ax, cp, d, color=style.ACCENT_PATH, lw=2.4, zorder=5)
            _scatter(ax, cp, d, s=90, facecolors="white",
                     edgecolors=style.ACCENT_PATH, linewidths=2.2, zorder=6)
    ax.set_title(title or f"coloured by {color_kind}")
    ax.legend(handles=_legend_handles(colors, order), loc="best")
    fig.tight_layout()
    return fig


def _spline(ax, pts, d):
    if len(pts) < 3:
        return
    try:
        from scipy.interpolate import splprep, splev
        tck, _ = splprep([pts[:, i] for i in range(d)], k=min(2, len(pts) - 1), s=0)
        u = np.linspace(0, 1, 100)
        out = splev(u, tck)
        ax.plot(*out, color=style.ACCENT_SPLINE, lw=3.0, alpha=0.9, zorder=4)
    except Exception:
        pass


# ----------------------------------------------------------- continuous scatter
def continuous_scatter(coords, values, *, title="", cmap=style.BELIEF_CMAP,
                       vmin=-1, vmax=1, label="", s=6, alpha=0.6, dims=3):
    d = min(dims, coords.shape[1])
    fig, ax = _new_axes(d, figsize=(7.0, 5.6))
    sc = _scatter(ax, coords, d, c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=s,
                  alpha=alpha, edgecolors="none", rasterized=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.08)
    cb.set_label(label or "value")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------- centroids
def centroid_plot(coords, labels, color_kind, *, title="", dims=3):
    colors = style.colors_for(color_kind)
    order = style.order_for(color_kind)
    labels = np.asarray(labels)
    d = min(dims, coords.shape[1])
    fig, ax = _new_axes(d, figsize=(6.4, 5.6))
    for c in order:
        m = labels == c
        if not m.any():
            continue
        _scatter(ax, coords[m], d, s=5, alpha=0.16, c=colors[c],
                 edgecolors="none", rasterized=True)
        ctr = coords[m, :d].mean(0)
        _scatter(ax, ctr[None], d, s=240, c=colors[c], edgecolors="white",
                 linewidths=2, zorder=6)
        if d == 2:
            _cov_ellipse(ax, ctr, np.cov(coords[m, :2].T), colors[c])
            ax.annotate(c, ctr, color=style.INKC, fontweight="bold", fontsize=11,
                        ha="center", va="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        else:
            ax.text(*ctr, f"  {c}", color=style.INKC, fontweight="bold", fontsize=10, zorder=7)
    ax.set_title(title or f"{color_kind} centroids")
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
def trajectory_paths(coords, df, *, title="trajectories", color_kind=None, dims=3):
    """Draw each episode's path through the space, fading by timestep.
    Episodes come from df['_traj_key'] (row-aligned with coords)."""
    d = min(dims, coords.shape[1])
    fig, ax = _new_axes(d, figsize=(6.8, 5.8))
    keys = df["_traj_key"].to_numpy()
    use3d = d >= 3
    if use3d:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection as LC
    else:
        LC = LineCollection
    for k in np.unique(keys):
        m = np.where(keys == k)[0]
        m = m[np.argsort(df["t"].to_numpy()[m])]
        xy = coords[m, :d]
        if len(xy) < 2:
            continue
        segs = np.stack([xy[:-1], xy[1:]], 1)
        if color_kind:
            colors = style.colors_for(color_kind)
            lab = df[color_kind].to_numpy()[m]
            cseg = [colors.get(lab[i + 1], "#999999") for i in range(len(segs))]
            ax.add_collection(LC(segs, colors=cseg, linewidths=1.6, alpha=0.8))
        else:
            ax.add_collection(LC(segs, array=np.linspace(0.15, 1, len(segs)),
                                 cmap="viridis", linewidths=1.6, alpha=0.85))
        _scatter(ax, xy[:1], d, s=30, c="white", edgecolors="k", zorder=5, marker="s")
        _scatter(ax, xy[-1:], d, s=40, c="k", zorder=5, marker="*")
    if use3d:    # autoscale doesn't track add_collection in 3-D
        allc = coords[:, :d]
        ax.set_xlim(allc[:, 0].min(), allc[:, 0].max())
        ax.set_ylim(allc[:, 1].min(), allc[:, 1].max())
        ax.set_zlim(allc[:, 2].min(), allc[:, 2].max())
    ax.set_title(title + "  (□ start, ★ end)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------- entanglement plane
def entanglement_plane(X, belief_dir, skill_dir, df, *, source=""):
    """Project activations onto (belief_axis, skill_axis) and show the SAME cloud
    twice — by category then by skill. If build/mine runs along the belief axis,
    belief and skill are entangled. (Inherently a 2-direction projection -> 2-D.)"""
    bx, sx = unit(belief_dir), unit(skill_dir)
    u, v = X @ bx, X @ sx
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
    fig.suptitle(f"belief↔skill plane   cos(belief, skill) = {cos:+.3f}", fontweight="bold")
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
