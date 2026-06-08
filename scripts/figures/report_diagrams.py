#!/usr/bin/env python3
"""Clean architecture / pipeline diagrams for the mech-interp report.
Generates: PPO+GRU forward pass, DreamerV3 (RSSM + imagination), the 2x2 study
design, and the steering-site schematic. Pure matplotlib box-and-arrow.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("outputs/report/figs"); OUT.mkdir(parents=True, exist_ok=True)
INK = "#2b3a4a"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10.5, "text.color": INK,
                     "axes.edgecolor": "white"})


def box(ax, x, y, w, h, text, fc, ec="#5b6b7b", fs=10.5, lw=1.3, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or INK, zorder=3, wrap=True)


def arrow(ax, x0, y0, x1, y1, c="#3a4a5a", lw=2.0, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16,
                                 lw=lw, color=c, connectionstyle=f"arc3,rad={rad}", zorder=1))


def _ax(w=15, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100 * h / w); ax.axis("off")
    return fig, ax


# ──────────────────────────── PPO + GRU ────────────────────────────
def ppo():
    fig, ax = _ax(16, 5.0); H = 100 * 5.0 / 16
    yc = H * 0.52; bh = 13
    blu, grn, org, gry = "#cdd8f0", "#bfe3c8", "#f6d6b0", "#e6e9ee"
    xs = [1, 16, 30, 44, 60]
    ws = [13, 12, 12, 14, 12]
    labels = [("obs\nminimap 21×21\n+ scalars(5/7)", gry),
              ("one-hot(9)\n⊕ CoordConv", blu),
              ("3×Conv 3×3\nReLU", blu),
              ("Linear→256\n⊕ scalars\n= enc_embed*", blu),
              ("GRU\nh_t ∈ ℝ¹²⁸\n= gru_h*", grn)]
    for (x, w, (t, c)) in zip(xs, ws, [labels[i] for i in range(5)]):
        box(ax, x, yc - bh / 2, w, bh, t, c, fs=9.5)
    for i in range(4):
        arrow(ax, xs[i] + ws[i], yc, xs[i + 1], yc)
    # recurrence
    arrow(ax, xs[4] + 6, yc - bh / 2, xs[4] + 6, yc - bh / 2 - 5, c="#888")
    arrow(ax, xs[4] + 6, yc - bh / 2 - 5, xs[4] - 1, yc - bh / 2 - 5, c="#888")
    arrow(ax, xs[4] - 1, yc - bh / 2 - 5, xs[4] - 1, yc - bh / 2, c="#888")
    ax.text(xs[4] + 6, yc - bh / 2 - 7.5, "h_{t-1}", ha="center", fontsize=9, color="#888")
    # heads
    hx = xs[4] + ws[4] + 9
    box(ax, hx, yc + 6, 26, 9, "actor → 6 logits\n↑ ↓ ← →  build  mine", org, fs=9.5)
    box(ax, hx, yc - 15, 26, 9, "critic → V(h)", org, fs=9.5)
    arrow(ax, xs[4] + ws[4], yc + 2, hx, yc + 10, rad=-0.15)
    arrow(ax, xs[4] + ws[4], yc - 2, hx, yc - 11, rad=0.15)
    ax.text(50, H - 3, "PPO + GRU  (~2.0 M params, model-free)", ha="center",
            fontsize=14, fontweight="bold")
    ax.text(50, 2, "* = activation sources probed: enc_embed(256) = encoder output ; "
            "gru_h(128) = recurrent belief carrier", ha="center", fontsize=9, color="#667")
    fig.savefig(OUT / "arch_ppo.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ──────────────────────────── DreamerV3 ────────────────────────────
def dreamer():
    fig, ax = plt.subplots(figsize=(16, 9.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    blu, grn, org, gry, pur, cyn = "#cdd8f0", "#bfe3c8", "#f6d6b0", "#e6e9ee", "#dcd0ee", "#cfe8ee"
    ax.text(50, 57, "DreamerV3  (~19.2 M params, model-based world model)",
            ha="center", fontsize=15, fontweight="bold")
    # encoder column
    box(ax, 2, 28, 12, 9, "obs\nminimap+scalars", gry, fs=9.5)
    box(ax, 17, 28, 12, 9, "encoder (MLP)\n→ embed", blu, fs=9.5)
    arrow(ax, 14, 32.5, 17, 32.5)
    # RSSM
    box(ax, 34, 37, 18, 9, "block-GRU\ndeter ∈ ℝ³⁰⁷²  *", grn, fs=9.5)
    box(ax, 34, 21, 18, 8, "posterior stoch\n24×24  (logits *)", pur, fs=9.5)
    arrow(ax, 29, 33, 34, 41, rad=-0.12)        # embed → deter
    arrow(ax, 29, 32, 34, 25, rad=0.12)         # embed → posterior
    arrow(ax, 43, 37, 43, 29, c="#8a93a0")      # deter → posterior
    ax.text(53, 47, "RSSM latent state", ha="center", fontsize=9, color="#667")
    # features
    box(ax, 56, 28, 12, 11, "features\n[deter⊕stoch]", cyn, fs=9.5)
    arrow(ax, 52, 41, 56, 36, rad=-0.1); arrow(ax, 52, 25, 56, 31, rad=0.1)
    # heads
    heads = [("decoder → recon obs", 47, blu), ("reward head (TwoHot)", 39, blu),
             ("continue head", 31, blu), ("actor → 6 logits", 21, org),
             ("critic → V (TwoHot)", 13, org)]
    for t, yy, c in heads:
        box(ax, 74, yy, 24, 6.6, t, c, fs=9.5)
        arrow(ax, 68, 33.5, 74, yy + 3.3, rad=-0.05)
    # imagination loop
    ax.add_patch(FancyBboxPatch((30, 3), 40, 7, boxstyle="round,pad=0.2,rounding_size=0.3",
                 fc="#fdeaea", ec="#d1495b", lw=1.5))
    ax.text(50, 6.5, "IMAGINATION: roll the RSSM forward prior-only (embed=None);\n"
            "the actor is trained on imagined returns via the reward + critic heads",
            ha="center", va="center", fontsize=9, color="#a23")
    arrow(ax, 86, 21, 86, 10.2, c="#d1495b"); arrow(ax, 70, 6.5, 52, 23, c="#d1495b", rad=-0.2)
    ax.text(50, 0.6, "* probed activation sources: rssm_deter(3072) = belief carrier ; "
            "rssm_stoch_logits(576) ; enc_embed(384)", ha="center", fontsize=9, color="#667")
    fig.savefig(OUT / "arch_dreamer.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ──────────────────────────── 2x2 design + pipeline ────────────────────────────
def pipeline():
    fig, ax = _ax(15, 6.2); H = 100 * 6.2 / 15
    blu, grn, org, gry = "#cdd8f0", "#bfe3c8", "#f6d6b0", "#e6e9ee"
    ax.text(25, H - 3, "Study design (2×2)", ha="center", fontsize=13, fontweight="bold")
    # 2x2
    cells = {(0, 1): "bt_ppo\n484k rows", (1, 1): "btc_ppo\n674k",
             (0, 0): "bt_dreamer\n323k", (1, 0): "btc_dreamer\n568k"}
    x0, y0, cw, ch = 8, 14, 17, 13
    for (cx, cy), t in cells.items():
        box(ax, x0 + cx * (cw + 2), y0 + cy * (ch + 2), cw, ch, t,
            grn if cx else blu, fs=9.5)
    ax.text(x0 - 2, y0 + ch + 2 + ch / 2, "PPO", rotation=90, va="center", ha="center", fontsize=10)
    ax.text(x0 - 2, y0 + ch / 2, "Dreamer", rotation=90, va="center", ha="center", fontsize=10)
    ax.text(x0 + cw / 2, y0 - 3, "BT (uncontrolled)", ha="center", fontsize=9.5)
    ax.text(x0 + cw + 2 + cw / 2, y0 - 3, "BTC (controlled)", ha="center", fontsize=9.5)
    # pipeline on the right
    ax.text(74, H - 3, "Analysis pipeline", ha="center", fontsize=13, fontweight="bold")
    steps = ["activation dataset\n(h5 sources + labels + maps)",
             "PCA / UMAP / t-SNE  (3-D manifolds)",
             "linear probes — belief & skill\n(grouped by map → no leakage)",
             "difference-of-means directions\n+ entanglement (cosine, principal angles)",
             "separability tests  (within-cat / E1 / E2)",
             "causal steering  (actor-input vs recurrent)"]
    sy = H - 8
    for i, s in enumerate(steps):
        box(ax, 56, sy - i * 8.0, 38, 6.4, s, [gry, blu, blu, org, org, "#fdeaea"][i], fs=8.8)
        if i:
            arrow(ax, 75, sy - (i - 1) * 8.0, 75, sy - i * 8.0 + 6.4)
    fig.savefig(OUT / "pipeline.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ──────────────────────────── steering sites ────────────────────────────
def steering():
    fig, ax = _ax(14, 4.4); H = 100 * 4.4 / 14
    grn, red, gry = "#bfe3c8", "#fdeaea", "#e6e9ee"
    box(ax, 5, 40, 16, 16, "GRU / RSSM\nrecurrent state\nh_t", gry, fs=10)
    box(ax, 40, 50, 18, 12, "actor\n(reads h_t)", "#f6d6b0", fs=10)
    box(ax, 40, 28, 18, 12, "h_t  carried\nto t+1", gry, fs=10)
    arrow(ax, 21, 50, 40, 56)            # to actor (read)
    arrow(ax, 21, 46, 40, 34)            # to carry
    # injections
    box(ax, 64, 50, 32, 12, "① inject into ACTOR INPUT (read-only)\n→ controllable, success preserved ✓",
        grn, fs=9.5, ec="#2a9d4a")
    box(ax, 64, 28, 32, 12, "② inject into RECURRENT CARRY (persistent)\n→ compounds off-manifold, agent breaks ✗",
        red, fs=9.5, ec="#d1495b")
    arrow(ax, 58, 56, 64, 56, c="#2a9d4a"); arrow(ax, 58, 34, 64, 34, c="#d1495b")
    ax.text(50, H - 2.5, "Where to steer a recurrent RL agent", ha="center", fontsize=13, fontweight="bold")
    ax.text(50, 2, "Same lesson in both agents: read-out injection works (PPO skill, Dreamer belief); "
            "transient recurrent (first-K) also works; persistent recurrent does not.",
            ha="center", fontsize=8.6, color="#667")
    fig.savefig(OUT / "steering_sites.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ppo(); dreamer(); pipeline(); steering()
    print("wrote", *(p.name for p in sorted(OUT.glob("*.png"))))
