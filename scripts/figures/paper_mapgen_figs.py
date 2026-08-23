#!/usr/bin/env python3
"""Figures for the map-generation chapter, including a didactic noise primer.

All panels are computed with the *same* primitives the generator uses
(`opensimplex`, the disk/massif/capsule fields, quantile thresholding), so the
teaching figures and the real maps cannot drift apart.

  fig_noise_primer.png    value vs gradient noise; the lattice; one octave
  fig_noise_octaves.png   octave stack -> fractal sum, with 1-D cross-sections
  fig_warp.png            domain warping: before / warp fields / after
  fig_pipeline.png        the 6 stages of one real map, left to right
  fig_quantile.png        how the same heightmap yields the three categories
  fig_features.png        the three overlay primitives (disk, massif, capsule)

Usage: PYTHONPATH=src python scripts/figures/paper_mapgen_figs.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import opensimplex  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

from cogniland.bridge_tunnel.mapgen import (  # noqa: E402
    _capsule_field, _disk_field, _massif_field, _simplex_field_rect,
    category_fracs, generate_commit_map,
)
from cogniland.bridge_tunnel.tiles import TILE_COLORS  # noqa: E402

PLT_RC = {"figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
          "axes.titlesize": 9, "axes.labelsize": 8.5,
          "axes.spines.top": False, "axes.spines.right": False}
TERR = LinearSegmentedColormap.from_list(
    "terr", ["#2d5c8f", "#3d71b8", "#7fa8d8", "#6ead56", "#9ac47f",
             "#b9a888", "#8a8a8a", "#6e6e6e"])
H, W = 32, 64
FORK_KW = dict(size=32, width=64, tree_frac=0.03, goal_half=0,
               fork_wall=True, passage_half=1, wall_margin=1, mem_gap=16)


# ── 1 · what gradient noise is ───────────────────────────────────────────

def fig_noise_primer(out):
    rng = np.random.default_rng(3)
    n = 6                                   # lattice cells
    res = 30                                # samples per cell

    # value noise: random value per lattice point, smoothly interpolated
    vals = rng.random((n + 1, n + 1))
    # gradient (Perlin) noise: random unit gradient per lattice point
    ang = rng.uniform(0, 2 * np.pi, (n + 1, n + 1))
    gx, gy = np.cos(ang), np.sin(ang)

    def fade(t):                            # Perlin's quintic ease curve
        return t * t * t * (t * (t * 6 - 15) + 10)

    xs = np.linspace(0, n, n * res, endpoint=False)
    X, Y = np.meshgrid(xs, xs)
    i0, j0 = np.floor(X).astype(int), np.floor(Y).astype(int)
    tx, ty = X - i0, Y - j0
    u, v = fade(tx), fade(ty)

    def lerp(a, b, t):
        return a + t * (b - a)

    val = lerp(lerp(vals[j0, i0], vals[j0, i0 + 1], u),
               lerp(vals[j0 + 1, i0], vals[j0 + 1, i0 + 1], u), v)
    d00 = gx[j0, i0] * tx + gy[j0, i0] * ty
    d10 = gx[j0, i0 + 1] * (tx - 1) + gy[j0, i0 + 1] * ty
    d01 = gx[j0 + 1, i0] * tx + gy[j0 + 1, i0] * (ty - 1)
    d11 = gx[j0 + 1, i0 + 1] * (tx - 1) + gy[j0 + 1, i0 + 1] * (ty - 1)
    grad = lerp(lerp(d00, d10, u), lerp(d01, d11, u), v)

    with plt.rc_context(PLT_RC):
        fig, ax = plt.subplots(1, 4, figsize=(12.6, 3.25))

        a = ax[0]
        a.imshow(val, cmap="Greys_r", origin="lower", extent=[0, n, 0, n])
        a.scatter(*np.meshgrid(np.arange(n + 1), np.arange(n + 1)),
                  s=9, c="#e0b429", zorder=3)
        a.set_title("(a) value noise — interpolate random\nvalues at lattice points", loc="left")

        a = ax[1]
        a.imshow(grad, cmap="Greys_r", origin="lower", extent=[0, n, 0, n])
        gi, gj = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
        a.quiver(gi, gj, gx, gy, color="#e0b429", scale=22, width=.006, zorder=3)
        a.set_title("(b) gradient (Perlin) noise — interpolate\n"
                    "*slopes*; value is 0 at every lattice point", loc="left")

        # anatomy of one cell
        a = ax[2]
        a.set_xlim(-.30, 1.30); a.set_ylim(-.30, 1.30); a.set_aspect("equal")
        a.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#f2f4ee", ec="#c3ccba"))
        corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
        gvec = [(.92, .40), (-.30, .95), (.55, -.84), (-.89, -.45)]
        px, py = .58, .44
        for (cx, cy), (ux, uy) in zip(corners, gvec):
            a.annotate("", xy=(cx + ux * .30, cy + uy * .30), xytext=(cx, cy),
                       arrowprops=dict(arrowstyle="-|>", color="#e0b429", lw=1.7),
                       zorder=4)
            a.plot([cx, px], [cy, py], color="#96a394", lw=.8, ls=":", zorder=3)
            dot = ux * (px - cx) + uy * (py - cy)
            mx, my = cx + (px - cx) * .58, cy + (py - cy) * .58
            a.annotate(f"{dot:+.2f}", (mx, my), fontsize=7.5, ha="center", va="center",
                       color="#b3261e" if dot < 0 else "#15803d", zorder=6,
                       bbox=dict(boxstyle="round,pad=.12", fc="white", ec="none",
                                 alpha=.85))
        a.plot(px, py, "o", color="#14201a", ms=5.5, zorder=5)
        a.annotate("sample point", (px, py), xytext=(1.02, -.20), fontsize=7.5,
                   ha="right", color="#14201a",
                   arrowprops=dict(arrowstyle="-", color="#96a394", lw=.8))
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title("(c) inside one cell: dot(gradient, offset)\n"
                    "at 4 corners, then blend", loc="left")

        # the fade curve
        a = ax[3]
        t = np.linspace(0, 1, 200)
        a.plot(t, t, color="#c3ccba", lw=1.2, ls="--", label="linear $t$")
        a.plot(t, fade(t), color="#2159cf", lw=2,
               label="quintic $6t^5-15t^4+10t^3$")
        a.set_xlabel("$t$"); a.set_ylabel("blend weight")
        a.legend(frameon=False, fontsize=7.5, loc="upper left")
        a.set_title("(d) the ease curve — zero 1st and 2nd\nderivative at the "
                    "lattice, so seams vanish", loc="left")
        for a in ax[:2]:
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout()
        fig.savefig(out / "fig_noise_primer.png", bbox_inches="tight")
        plt.close(fig)


# ── 2 · octaves ──────────────────────────────────────────────────────────

def fig_noise_octaves(out):
    sim = opensimplex.OpenSimplex(seed=77)
    base = max(6.0, max(H, W) / 4.5)
    octs = {base: 1.0, base / 2: .5, base / 4: .25, base / 8: .125, base / 16: .0625}
    with plt.rc_context(PLT_RC):
        fig, ax = plt.subplots(2, 6, figsize=(14.2, 4.0),
                               gridspec_kw=dict(height_ratios=[1, .62]))
        acc = np.zeros((H, W)); tw = 0.
        for k, (s, w) in enumerate(octs.items()):
            f = _simplex_field_rect(sim, H, W, {s: 1.0})
            acc += w * f; tw += w
            ax[0, k].imshow(f, cmap=TERR)
            ax[0, k].set_title(f"octave {k}\nwavelength {s:.1f} px · weight {w}",
                               loc="left", fontsize=8)
            ax[1, k].plot(f[H // 2], color="#2159cf", lw=1)
            ax[1, k].set_ylim(-1.15, 1.15)
            ax[1, k].set_xlabel("column")
            if k == 0:
                ax[1, k].set_ylabel("value on\nthe centre row")
        ax[0, 5].imshow(acc / tw, cmap=TERR)
        ax[0, 5].set_title("Σ weighted octaves\n= fractal heightmap", loc="left",
                           fontsize=8, color="#15803d")
        ax[1, 5].plot((acc / tw)[H // 2], color="#15803d", lw=1.4)
        ax[1, 5].set_ylim(-1.15, 1.15); ax[1, 5].set_xlabel("column")
        for a in ax[0]:
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle("Fractal (fBm) summation — each octave halves the wavelength "
                     "and halves the amplitude; the sum has detail at every scale",
                     y=1.02)
        fig.tight_layout()
        fig.savefig(out / "fig_noise_octaves.png", bbox_inches="tight")
        plt.close(fig)


# ── 3 · domain warping ───────────────────────────────────────────────────

def fig_warp(out):
    seed = 12
    sim_h = opensimplex.OpenSimplex(seed=seed * 7 + 11)
    sim_wr = opensimplex.OpenSimplex(seed=seed * 7 + 12)
    sim_wc = opensimplex.OpenSimplex(seed=seed * 7 + 13)
    base = max(6.0, max(H, W) / 4.5)
    octs = {base: 1., base / 2: .5, base / 4: .25, base / 8: .125, base / 16: .0625}
    h0 = _simplex_field_rect(sim_h, H, W, octs)
    wr = _simplex_field_rect(sim_wr, H, W, {base: 1., base / 2: .5})
    wc = _simplex_field_rect(sim_wc, H, W, {base: 1., base / 2: .5})
    rr = np.arange(H, dtype=float)[:, None]; cc = np.arange(W, dtype=float)[None, :]
    RR = np.broadcast_to(rr, (H, W)); CC = np.broadcast_to(cc, (H, W))
    aw = max(H, W) * .10
    hw = map_coordinates(h0, [np.clip(RR + aw * wr, 0, H - 1),
                              np.clip(CC + aw * wc, 0, W - 1)], order=1, mode="reflect")
    with plt.rc_context(PLT_RC):
        fig, ax = plt.subplots(1, 4, figsize=(13.2, 2.5))
        ax[0].imshow(h0, cmap=TERR); ax[0].set_title("(a) heightmap $h_0$", loc="left")
        ax[1].imshow(wr, cmap="PuOr"); ax[1].set_title("(b) row-offset field", loc="left")
        ax[2].imshow(wc, cmap="PuOr"); ax[2].set_title("(c) column-offset field", loc="left")
        ax[3].imshow(hw, cmap=TERR)
        ax[3].set_title("(d) warped: $h_0(r+\\alpha w_r,\\; c+\\alpha w_c)$", loc="left")
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle("Domain warping — two more noise fields bend the coordinates "
                     "before sampling ($\\alpha$ = 10 % of the map), turning round "
                     "blobs into organic coastlines", y=1.06)
        fig.tight_layout()
        fig.savefig(out / "fig_warp.png", bbox_inches="tight")
        plt.close(fig)


# ── 4 · overlay primitives ───────────────────────────────────────────────

def fig_features(out):
    rr = np.arange(H, dtype=float)[:, None]; cc = np.arange(W, dtype=float)[None, :]
    rng = np.random.default_rng(5)
    disk = _disk_field(rr, cc, 16, 16, 6.0, 1.8)
    massif = _massif_field(rr, cc, 16, 32, 4.0, 4, rng)
    caps = _capsule_field(rr, cc, 16, 48, 20.0, 2.6, 0.35, wiggle=3.8, period=28.)
    with plt.rc_context(PLT_RC):
        fig, ax = plt.subplots(1, 3, figsize=(12.6, 2.3))
        for a, f, t in zip(ax, [disk, massif, caps], [
                "(a) logistic disk — round lake / knoll",
                "(b) massif — 3–5 overlapping Gaussian lobes",
                "(c) capsule — meandering ridge (sine wiggle)"]):
            a.imshow(f, cmap=TERR); a.set_title(t, loc="left")
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle("The three overlay primitives, added to (mountains) or "
                     "subtracted from (lakes) the fractal heightmap with weight ±1.3",
                     y=1.1)
        fig.tight_layout()
        fig.savefig(out / "fig_features.png", bbox_inches="tight")
        plt.close(fig)


# ── 5 · quantile thresholding is what makes a category ───────────────────

def fig_quantile(out):
    seed = 21
    sim = opensimplex.OpenSimplex(seed=seed * 7 + 11)
    base = max(6.0, max(H, W) / 4.5)
    hf = _simplex_field_rect(sim, H, W, {base: 1., base / 2: .5, base / 4: .25,
                                         base / 8: .125, base / 16: .0625})
    cats = ["rocky", "balanced", "lakes"]
    with plt.rc_context(PLT_RC):
        fig = plt.figure(figsize=(13.0, 3.5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=.22)

        a = fig.add_subplot(gs[0])
        a.hist(hf.ravel(), bins=60, color="#c3ccba", edgecolor="none")
        for cat, col in zip(cats, ["#6e6e6e", "#86684a", "#3d71b8"]):
            wf, rf = category_fracs(cat)
            a.axvline(np.quantile(hf, wf), color=col, lw=1.6)
            a.axvline(np.quantile(hf, 1 - rf), color=col, lw=1.6, ls="--")
        a.set_xlabel("heightmap value"); a.set_ylabel("cells")
        a.set_title("(a) one heightmap, three pairs of cut points\n"
                    "solid = water level, dashed = rock level", loc="left")

        for k, cat in enumerate(cats):
            wf, rf = category_fracs(cat)
            terr = np.zeros((H, W), int)
            terr[hf < np.quantile(hf, wf)] = 1
            terr[hf > np.quantile(hf, 1 - rf)] = 2
            a = fig.add_subplot(gs[k + 1])
            a.imshow(TILE_COLORS[terr], interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"({'bcd'[k]}) {cat}\nwater {wf:.1%} · rock {rf:.1%}", loc="left")
        fig.suptitle("The category is a pair of quantiles, not a different world. "
                     "The same heightmap thresholded three ways gives three "
                     "categories with identical geometry.", y=1.04)
        fig.tight_layout()
        fig.savefig(out / "fig_quantile.png", bbox_inches="tight")
        plt.close(fig)


# ── 6 · the whole pipeline on one real map ───────────────────────────────

def fig_pipeline(out):
    seed, cat = 4, "lakes"
    wf, rf = category_fracs(cat)
    sim_h = opensimplex.OpenSimplex(seed=seed * 7 + 11)
    sim_wr = opensimplex.OpenSimplex(seed=seed * 7 + 12)
    sim_wc = opensimplex.OpenSimplex(seed=seed * 7 + 13)
    rng = np.random.default_rng(seed * 911 + 17)
    base = max(6.0, max(H, W) / 4.5)
    octs = {base: 1., base / 2: .5, base / 4: .25, base / 8: .125, base / 16: .0625}
    h0 = _simplex_field_rect(sim_h, H, W, octs)
    wr = _simplex_field_rect(sim_wr, H, W, {base: 1., base / 2: .5})
    wc = _simplex_field_rect(sim_wc, H, W, {base: 1., base / 2: .5})
    rr = np.arange(H, dtype=float)[:, None]; cc = np.arange(W, dtype=float)[None, :]
    RR = np.broadcast_to(rr, (H, W)); CC = np.broadcast_to(cc, (H, W))
    aw = max(H, W) * .10
    hw = map_coordinates(h0, [np.clip(RR + aw * wr, 0, H - 1),
                              np.clip(CC + aw * wc, 0, W - 1)], order=1, mode="reflect")
    hov = hw.copy()
    for _ in range(3):
        r, c = rng.uniform(.14, .86) * H, rng.uniform(.14, .86) * W
        hov -= 1.3 * _disk_field(rr, cc, r, c, 64 * .06, 2.0)
    for _ in range(3):
        r, c = rng.uniform(.14, .86) * H, rng.uniform(.14, .86) * W
        hov += 1.3 * _massif_field(rr, cc, r, c, 64 * .05, 4, rng)
    thr = np.zeros((H, W), int)
    thr[hov < np.quantile(hov, wf)] = 1
    thr[hov > np.quantile(hov, 1 - rf)] = 2
    final = generate_commit_map(seed=seed, category=cat, **FORK_KW)

    panels = [(h0, "1 · fractal heightmap", "field"),
              (hw, "2 · domain-warped", "field"),
              (hov, "3 · lakes & massifs overlaid", "field"),
              (thr, "4 · quantile threshold → tiles", "tiles"),
              (None, "5 · fringes, edge bands, trees", "tiles"),
              (None, "6 · fork wall, passage, doors", "tiles")]
    stage5 = final.terrain.copy()
    stage5[:, 64 - 1 - 1] = np.where(stage5[:, 64 - 1 - 1] == 6, 0, stage5[:, 64 - 1 - 1])
    stage5[:, 63] = np.where(stage5[:, 63] == 4, 0, stage5[:, 63])

    with plt.rc_context(PLT_RC):
        fig, ax = plt.subplots(2, 3, figsize=(13.6, 3.9))
        ax = ax.flat
        imgs = [h0, hw, hov, TILE_COLORS[thr], TILE_COLORS[stage5],
                TILE_COLORS[final.terrain]]
        for a, im, (_, title, kind) in zip(ax, imgs, panels):
            if kind == "field":
                a.imshow(im, cmap=TERR)
            else:
                a.imshow(im, interpolation="nearest")
            a.set_title(title, loc="left")
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"Six stages of one real map  (seed {seed}, category “{cat}”)", y=1.005)
        fig.tight_layout()
        fig.savefig(out / "fig_pipeline.png", bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPO / "paper/figures/forkwall_paper"))
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig_noise_primer(out)
    fig_noise_octaves(out)
    fig_warp(out)
    fig_features(out)
    fig_quantile(out)
    fig_pipeline(out)
    print("wrote mapgen figures ->", out)


if __name__ == "__main__":
    main()
