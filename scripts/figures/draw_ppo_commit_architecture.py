#!/usr/bin/env python3
"""Slide-ready architecture drawing of the PPO+GRU policy (bridge_tunnel_commit,
one-hot obs, view_size=21, scalars=7, Discrete(6)). Pure matplotlib → PNG."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/figures/bridge_tunnel_commit/ppo_architecture.png")

C_IO   = "#e9ecef"   # obs / action
C_CNN  = "#dbe7fb"   # conv / mlp (blue)
C_GRU  = "#d4efd6"   # gru (green)
C_HEAD = "#ffe6c2"   # heads (orange)
EDGE   = "#2f3640"

fig, ax = plt.subplots(figsize=(18.5, 8.0), dpi=200)
ax.set_xlim(0, 26); ax.set_ylim(0, 10.5); ax.axis("off")


def box(cx, cy, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.04,rounding_size=0.12",
                 fc=fc, ec=EDGE, lw=1.8, zorder=2))
    ax.text(cx, cy + h / 2 - 0.34, title, ha="center", va="center",
            fontsize=12.5, fontweight="bold", zorder=3)
    ax.text(cx, cy - 0.15, "\n".join(lines), ha="center", va="center",
            fontsize=10.2, zorder=3, color="#222")


def arrow(p0, p1, label=None, rad=0.0, lblpos=0.5, dy=0.34, color=EDGE, fs=9.8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                 lw=2.0, color=color, zorder=1,
                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx = p0[0] + (p1[0] - p0[0]) * lblpos
        my = p0[1] + (p1[1] - p0[1]) * lblpos + dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs,
                color="#1450a0", fontstyle="italic", zorder=4)


# --- nodes (widely spaced so the arrow labels never touch a box) ---
box(2.1, 7.1, 2.8, 2.0, "minimap obs",
    ["21×21  int8", "tile ids 0–8", "(egocentric, POMDP)"], C_IO)
box(2.1, 3.5, 2.8, 1.9, "scalars (7)",
    ["facing 1-hot (4)", "step / max", "commit_build, commit_mine"], C_IO)
box(6.7, 7.1, 3.6, 2.4, "CoordConv CNN",
    ["one-hot →21×21×9", "+coord →(11,21,21)", "3× Conv3×3 →32, ReLU",
     "21²→19²→17²→15²", "flatten → 7200"], C_CNN)
box(11.3, 5.2, 2.2, 1.25, "concat", ["7200 ⊕ 7", "= 7207"], C_CNN)
box(15.6, 5.2, 3.0, 1.7, "MLP embed",
    ["Linear 7207→256", "ReLU", "enc_embed (256)"], C_CNN)
box(20.1, 5.2, 3.3, 2.0, "GRU  (256→128)",
    ["hₜ ∈ ℝ¹²⁸", "recurrent memory"], C_GRU)
box(24.2, 7.1, 2.8, 1.5, "actor head",
    ["Linear 128→6", "softmax"], C_HEAD)
box(24.2, 3.4, 2.8, 1.5, "critic head",
    ["Linear 128→1", "value V(hₜ)"], C_HEAD)
box(24.2, 9.5, 2.8, 1.4, "action",
    ["{↑ ↓ ← →, build, mine}"], C_IO)

# --- arrows (longer, labels sit clearly in the gaps) ---
arrow((3.5, 7.1), (4.9, 7.1), "encode")
arrow((8.5, 6.6), (10.2, 5.75), "7200")
arrow((3.5, 3.7), (10.15, 4.9), "(7)", lblpos=0.58, dy=0.30)
arrow((12.4, 5.2), (14.1, 5.2), "7207")
arrow((17.1, 5.2), (18.45, 5.2), "256")
arrow((21.75, 5.75), (22.8, 6.8))
arrow((21.75, 4.75), (22.8, 3.7))
ax.text(22.3, 5.2, "hₜ (128)", ha="center", va="center", fontsize=9.8,
        color="#1450a0", fontstyle="italic", zorder=4)
arrow((24.2, 7.85), (24.2, 8.8), "argmax / sample", dy=0.0, fs=9.2)

# GRU recurrent self-loop
ax.add_patch(FancyArrowPatch((19.3, 6.2), (20.9, 6.2), arrowstyle="-|>",
             mutation_scale=15, lw=2.0, color="#2e7d32", zorder=1,
             connectionstyle="arc3,rad=-1.1"))
ax.text(20.1, 7.55, "hₜ₋₁  (done-masked)", ha="center", va="center",
        fontsize=9.6, color="#2e7d32", fontstyle="italic")

ax.text(13, 0.7, "PPO + GRU policy  ·  bridge_tunnel_commit (one-hot obs)  ·  "
        "≈2.0 M params (the 7207→256 matrix is ~92%)",
        ha="center", va="center", fontsize=11.5, color="#444")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
