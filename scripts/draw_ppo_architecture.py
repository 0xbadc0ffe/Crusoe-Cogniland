"""Draw the PPO-GRU policy architecture diagram (updated: 6 actions + GRU pruning).

Standalone matplotlib figure — no repo imports — mirroring the policy in
``scripts/train_ppo_gru.py``:

    semantic 21x21 -> tile-embed(+CoordConv) -> 3xConv -> flatten(+skill)/Linear
    -> GRU(128) -> {actor: 6 move logits, belief: tanh map-probe, critic: V(h)}

The GRU box is annotated with the optional magnitude-pruning sparsity applied
to its recurrent weight matrices (W_ih, W_hh).

Run:  python scripts/draw_ppo_architecture.py --out paper/ppo_gru_architecture.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

LAVENDER = "#dfe0f3"
GREEN = "#cdedd2"
PEACH = "#f8e0c8"
GREY = "#e9e9ee"


def _box(ax, x, y, w, h, text, fc, *, fontsize=11, ec="#444", lw=1.2, hatch=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc, hatch=hatch, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3)


def _arrow(ax, x0, y0, x1, y1, lw=1.6, color="#333"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
        linewidth=lw, color=color, zorder=1))


def draw(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0, 30); ax.set_ylim(0, 10); ax.axis("off")

    by, bh = 4.2, 1.8           # trunk box y + height
    # ---- trunk ----
    _box(ax, 0.4, by, 3.2, bh, "semantic\n21$\\times$21", GREY)
    _box(ax, 4.4, by, 3.6, bh, "tile-embed\n+ CoordConv", LAVENDER)
    ax.text(6.2, by - 0.7, "16d embedding", ha="center", fontsize=10)
    _box(ax, 8.8, by, 3.2, bh, "3$\\times$Conv\n3$\\times$3 ReLU", LAVENDER)
    _box(ax, 12.8, by, 4.0, bh, "flatten $\\oplus$ skill\nLinear 256", LAVENDER)
    _box(ax, 17.6, by, 3.4, bh, "GRU\n$h_t \\in \\mathbb{R}^{128}$", GREEN)

    # trunk arrows
    for x0, x1 in [(3.6, 4.4), (8.0, 8.8), (12.0, 12.8), (16.8, 17.6)]:
        _arrow(ax, x0, by + bh / 2, x1, by + bh / 2)

    # GRU recurrent self-loop
    ax.add_patch(FancyArrowPatch(
        (18.6, by), (19.6, by), connectionstyle="arc3,rad=-1.6",
        arrowstyle="-|>", mutation_scale=13, linewidth=1.4, color="#333"))
    ax.text(19.3, by - 1.5, "$h_{t-1}$", ha="center", fontsize=11)

    # GRU sparsity badge — clean fill so the text stays readable
    _box(ax, 16.4, by + bh + 0.55, 5.9, 1.2,
         "magnitude pruning (optional)\n$W_{ih},\\,W_{hh}\\rightarrow$ 90% sparse",
         "#fdeaea", ec="#c0392b", lw=1.4, fontsize=10)
    _arrow(ax, 19.3, by + bh + 0.55, 19.3, by + bh + 0.02, lw=1.3, color="#c0392b")

    # ---- heads ----
    hx, hw, hh = 23.4, 6.2, 1.5
    _box(ax, hx, 7.1, hw, hh, "actor\n$\\rightarrow$ 6 move logits", PEACH, fontsize=10.5)
    _box(ax, hx, 4.45, hw, hh, "belief\ntanh $\\in [-1,1]$  (aux)", PEACH, fontsize=10.5)
    _box(ax, hx, 1.8, hw, hh, "critic\n$\\rightarrow V(h)$", PEACH, fontsize=10.5)
    gx, gy = 21.0, by + bh / 2
    _arrow(ax, gx, gy, hx, 7.1 + hh / 2)
    _arrow(ax, gx, gy, hx, 4.45 + hh / 2)
    _arrow(ax, gx, gy, hx, 1.8 + hh / 2)

    # actions legend under the actor head
    ax.text(hx + hw / 2, 6.85,
            r"$\uparrow\,\downarrow\,\leftarrow\,\rightarrow$ + build_raft + build_harness",
            ha="center", va="top", fontsize=8.5, color="#555")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper/ppo_gru_architecture.png"))
    args = ap.parse_args()
    print(f"wrote {draw(args.out)}")


if __name__ == "__main__":
    main()
