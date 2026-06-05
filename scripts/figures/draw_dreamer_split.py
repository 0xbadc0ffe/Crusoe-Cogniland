"""Schematic: how DreamerV3 separates belief (world model) from action
selection (actor-critic) via a stop-gradient wall.

Standalone matplotlib figure (no repo imports).
Run:  python scripts/figures/draw_dreamer_split.py --out paper/figures/dreamer_belief_action.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

GREY = "#e9e9ee"
LAV = "#dfe0f3"
GREEN = "#cdedd2"
PEACH = "#f8e0c8"


def _box(ax, x, y, w, h, text, fc, *, fs=10, ec="#444", lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4)


def _arrow(ax, x0, y0, x1, y1, color="#333", lw=1.6, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                 mutation_scale=15, linewidth=lw, color=color,
                 linestyle=ls, zorder=2))


def draw(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15.5, 6.6))
    ax.set_xlim(0, 31); ax.set_ylim(0, 13); ax.axis("off")

    # ── region backdrops ──
    ax.add_patch(Rectangle((0.3, 1.2), 15.4, 10.6, facecolor="#eef1fb",
                           edgecolor="#9aa3d0", lw=1.0, zorder=0))
    ax.text(8.0, 11.35, "WORLD MODEL  —  trained on real data, world-modeling losses only",
            ha="center", fontsize=10.5, color="#3b3f78", fontweight="bold")
    ax.add_patch(Rectangle((17.6, 1.2), 13.1, 10.6, facecolor="#fdf1e6",
                           edgecolor="#d8a86b", lw=1.0, zorder=0))
    ax.text(24.1, 11.35, "ACTOR–CRITIC  —  trained in imagination",
            ha="center", fontsize=10.5, color="#8a5a1f", fontweight="bold")

    # ── world-model trunk ──
    _box(ax, 0.7, 5.4, 2.4, 1.7, "obs\n$o_t$", GREY)
    _box(ax, 3.7, 5.4, 3.0, 1.7, "MLP\nencoder", LAV)
    _box(ax, 7.3, 4.7, 4.2, 3.1,
         "RSSM\nblock-GRU $h_t$\n+ discrete $z_t$\n$\\Rightarrow$ belief $s_t$", GREEN, fs=10)
    # WM heads
    _box(ax, 12.2, 8.4, 3.1, 1.5, "decoder $\\hat o_t$", LAV, fs=9)
    _box(ax, 12.2, 6.4, 3.1, 1.5, "reward / cont.", LAV, fs=9)
    _arrow(ax, 3.1, 6.25, 3.7, 6.25)
    _arrow(ax, 6.7, 6.25, 7.3, 6.25)
    _arrow(ax, 11.5, 6.6, 12.2, 7.15)   # to reward/cont
    _arrow(ax, 11.5, 6.9, 12.2, 9.0)    # to decoder
    # recurrent loop
    ax.add_patch(FancyArrowPatch((8.6, 4.7), (9.7, 4.7),
                 connectionstyle="arc3,rad=-1.5", arrowstyle="-|>",
                 mutation_scale=12, lw=1.3, color="#333", zorder=2))
    ax.text(9.2, 3.35, "$h_{t-1},\\,a_{t-1}$", ha="center", fontsize=9)
    # WM loss banner
    ax.text(8.0, 2.15,
            "losses: reconstruction · reward · continuation · dyn/rep KL"
            "   $\\Rightarrow\\;\\nabla$ world-model params only",
            ha="center", fontsize=9.2, color="#3b3f78",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9aa3d0"))

    # ── stop-gradient wall ──
    ax.plot([16.65, 16.65], [1.5, 11.0], color="#c0392b", lw=2.4, ls=(0, (4, 3)), zorder=5)
    ax.text(16.65, 11.45, "stop-gradient", ha="center", fontsize=10,
            color="#c0392b", fontweight="bold")

    # forward info flow across the wall (belief is read, detached)
    _arrow(ax, 11.5, 5.4, 18.4, 5.4, color="#2c7", lw=2.0)
    ax.text(14.0, 5.75, "belief features (detached)", ha="center", fontsize=8.8, color="#1c7a4f")

    # blocked gradient back into the belief
    _arrow(ax, 18.4, 7.6, 17.0, 7.6, color="#c0392b", lw=1.8, ls=(0, (3, 2)))
    ax.text(17.0, 8.5, "policy / value gradient\nCANNOT cross", ha="center",
            fontsize=8.6, color="#c0392b")
    ax.text(16.65, 7.6, "✗", ha="center", va="center", fontsize=15,
            color="#c0392b", fontweight="bold", zorder=6)

    # ── actor-critic ──
    _box(ax, 18.4, 4.85, 3.0, 1.5, "belief $s_t$\n(input, detached)", "#fff", fs=9, ls="--", ec="#888")
    _box(ax, 22.6, 7.6, 7.6, 1.6,
         "actor $\\to$ 6 move logits\n(incl. build_raft / build_harness)", PEACH, fs=9.5)
    _box(ax, 22.6, 4.9, 7.6, 1.6, "critic $\\to V(s_t)$", PEACH, fs=9.5)
    _arrow(ax, 21.4, 5.9, 22.6, 8.0)
    _arrow(ax, 21.4, 5.7, 22.6, 5.6)
    # imagine note
    ax.text(24.1, 3.55, "imagine forward with the frozen world model as simulator",
            ha="center", fontsize=8.8, color="#8a5a1f")
    ax.text(24.1, 2.15,
            "losses: actor (imagined $\\lambda$-returns) · critic"
            "   $\\Rightarrow\\;\\nabla$ actor–critic params only",
            ha="center", fontsize=9.2, color="#8a5a1f",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d8a86b"))

    fig.suptitle("DreamerV3: belief (world model) vs. strategy commitment (actor) "
                 "separated by a stop-gradient wall", fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper/figures/dreamer_belief_action.png"))
    args = ap.parse_args()
    print(f"wrote {draw(args.out)}")


if __name__ == "__main__":
    main()
