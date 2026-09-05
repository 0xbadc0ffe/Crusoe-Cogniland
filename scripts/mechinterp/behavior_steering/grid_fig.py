#!/usr/bin/env python3
"""Shared trajectory-grid drawing, used by both the Dreamer and PPO figures.

One group of panels per map, one panel per arm, R stochastic rollouts overlaid
with a different colour each so overlapping paths stay separable. Terrain is
drawn at full saturation: washing it out made the maps unreadable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RC = {"figure.dpi": 200, "savefig.dpi": 200, "font.size": 9}
# one colour per rollout, chosen to contrast with every terrain type
PALETTE = ["#2e1065", "#7c3aed", "#c026d3", "#f472b6", "#0e7490",
           "#1e3a8a", "#f97316", "#fde047", "#7f1d1d", "#0f172a"]


def _draw_panel(ax, rec, rs, cmap, T, plt, show=None, markers=True):
    """`show` limits how many rollouts are DRAWN; the returned counts always
    use every rollout, so the panel labels stay honest. `markers=False` drops
    the tool-event glyphs (X = mined block, square = placed bridge)."""
    ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
    H, W = rec.terrain.shape
    drawn = rs if show is None else rs[:show]
    # purples: they read against green grass, blue water and grey rock alike.
    # A rainbow ramp put greens on the grass and reds next to the event markers.
    cols = [PALETTE[i % len(PALETTE)] for i in range(len(drawn))]
    for r, c_ in zip(drawn, cols):
        P = np.array([[st["r"], st["c"]] for st in r["trace"]], float)
        ax.plot(P[:, 1], P[:, 0], color=c_, lw=1.3, alpha=.9,
                solid_capstyle="round", zorder=5)
    # one marker per (cell, kind), sized by how many rollouts touched it:
    # raw per-rollout markers pile into an unreadable blob
    tally = {}
    for r in (drawn if markers else ()):
        for st in r["trace"]:
            ev = st.get("ev")
            if ev:
                key = (ev["r"], ev["c"], ev["kind"])
                tally[key] = tally.get(key, 0) + 1
    for (rr, cc, kind), n in tally.items():
        ax.plot([cc], [rr], "X" if kind == "mine" else "s",
                ms=4.0 + 2.4 * np.sqrt(n / max(len(drawn), 1)), mew=1.3,
                mfc="none" if kind == "build" else "#111827",
                mec="#111827", zorder=6)
    for cells, ec in ((rec.top_goal_cells, "#b0402c"),
                      (rec.bottom_goal_cells, "#2c5f86")):
        r_ = [q[0] for q in cells]; c_ = [q[1] for q in cells]
        ax.add_patch(plt.Rectangle((min(c_) - .8, min(r_) - .8),
                                   max(c_) - min(c_) + 1.6,
                                   max(r_) - min(r_) + 1.6, fill=False,
                                   edgecolor=ec, lw=1.4, zorder=7))
    ax.plot(rec.spawn[1], rec.spawn[0], "o", color="white", mec="black",
            ms=3.6, zorder=8)
    ax.set_xlim(-.5, W - .5); ax.set_ylim(H - .5, -.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#c9cfc8")
    return (float(np.mean([r["mines"] for r in rs])),
            float(np.mean([r["builds"] for r in rs])))


def _panel_label(rs, mn, bd, door_pct):
    lab = f"{mn:.1f}$\\times$tunnel   {bd:.1f}$\\times$bridge"
    if not door_pct:
        return lab
    n = len(rs)
    top = sum(1 for r in rs if r.get("door") == "top")
    to = sum(1 for r in rs if r.get("timeout"))
    lab = f"top door {100 * top / max(n, 1):.0f}%   " + lab
    if to:
        lab += f"   TO {to}/{n}"
    return lab


def draw_grid(pool, mids, rows, arms, out_path, title, groups_per_row=3,
              show=None, markers=True, door_pct=False):
    """arms: list of (tag, colour). rows: dicts with map_id, arm, trace, ...
    `markers=False` hides the tool-event glyphs; `door_pct=True` prefixes each
    panel label with the share of rollouts that exited by the TOP door (and
    the timeout count when any rollout timed out)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cogniland.bridge_tunnel import tiles as T

    cmap = matplotlib.colormaps["turbo"]
    nrow = int(np.ceil(len(mids) / groups_per_row))
    with plt.rc_context(RC):
        ncol = groups_per_row * len(arms)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 1.95 * nrow),
                                 squeeze=False)
        for i, mid in enumerate(mids):
            gr, gc = divmod(i, groups_per_row)
            for j, (tag, colour) in enumerate(arms):
                ax = axes[gr][gc * len(arms) + j]
                rs = [r for r in rows if r["map_id"] == mid and r["arm"] == tag]
                if not rs:
                    ax.axis("off"); continue
                mn, bd = _draw_panel(ax, pool[mid], rs, cmap, T, plt, show,
                                     markers=markers)
                ax.set_title(_panel_label(rs, mn, bd, door_pct),
                             fontsize=8, loc="left", pad=3, color="#374151")
                if gr == 0:
                    ax.text(.5, 1.48, tag, transform=ax.transAxes, ha="center",
                            fontsize=11, weight="bold", color=colour)
        for k in range(len(mids), nrow * groups_per_row):
            gr, gc = divmod(k, groups_per_row)
            for j in range(len(arms)):
                axes[gr][gc * len(arms) + j].axis("off")
        fig.suptitle(title, y=1.004, fontsize=11)
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    print("wrote", Path(out_path).name)
