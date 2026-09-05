#!/usr/bin/env python3
"""Figures for the IAB paper (PPO only, 'passage' wording, no suptitles).

  dataset   3 x 5 grid of held-out maps by category
  bins      the eight position bins drawn on one held-out map
  pca       (a) lakes/rocky states per bin in PC1-PC2 with the per-bin belief
            axis; (b) balanced-map states projected into the same plane
  belief    results figure: probe accuracy per bin for the three readouts, and
            the cosine matrix between the per-bin single directions
  steer     results figure: P(top flag) against steering strength, one write of
            h <- h + alpha v at corr2 entry, lakes / rocky / balanced maps

  PYTHONPATH=src:scripts/mechinterp/belief_report python scripts/figures/paper/iab_appendix_figs.py all
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "mechinterp" / "belief_report"))
import data as D  # noqa: E402
from cogniland.bridge_tunnel import tiles as T  # noqa: E402

OUTS = [REPO / "paper/iab2026/paper/figures", REPO / "paper/figures/iab2026"]
INK, INK2, MUTE = "#0b0b0b", "#52514e", "#a8a7a1"
C_LAKES, C_ROCKY, C_BAL = "#0e7490", "#b91c1c", "#6b7280"
RC = {"figure.dpi": 200, "savefig.dpi": 200, "font.size": 9,
      "axes.spines.top": False, "axes.spines.right": False,
      "axes.edgecolor": MUTE, "xtick.color": INK2, "ytick.color": INK2,
      "axes.labelcolor": INK2, "figure.facecolor": "white", "axes.facecolor": "white"}
LABELS = [l.split("\n")[0].replace("wall", "passage") for l in D.BIN_LABELS]
PHASE_COL = {"evidence": "#16a34a", "corridor": "#d97706", "past_wall": "#7c3aed"}
PHASE_LABEL = {"evidence": "evidence phase", "corridor": "memory corridor",
               "past_wall": "passage"}


def save(fig, name):
    for o in OUTS:
        o.mkdir(parents=True, exist_ok=True)
        fig.savefig(o / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def draw_map(ax, rec, spawn=True, flags=True):
    ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
    if flags:
        for cells, name in ((rec.top_goal_cells, "top"), (rec.bottom_goal_cells, "bottom")):
            ok = rec.correct_target in ("either", name)
            for r, c in cells:
                ax.scatter([c], [r], c=("lime" if ok else "red"), s=34, marker="s",
                           edgecolors="k", lw=.7, zorder=5)
    if spawn:
        ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=28, marker="o",
                   edgecolors="k", zorder=5)
    ax.set_xticks([]); ax.set_yticks([])


# ─────────────────────────────────────────────────────────────── dataset ──
def fig_dataset(n=5):
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    with plt.rc_context(RC):
        fig, axs = plt.subplots(3, n, figsize=(2.6 * n, 4.6))
        for ri, cat in enumerate(("balanced", "lakes", "rocky")):
            recs = [r for r in pool if r.category == cat][:n]
            for ci, rec in enumerate(recs):
                draw_map(axs[ri, ci], rec)
                for s in axs[ri, ci].spines.values():
                    s.set_edgecolor("#c9cfc8")
            axs[ri, 0].set_ylabel(cat, fontsize=10, color=INK)
        fig.subplots_adjust(wspace=.05, hspace=.08)
        save(fig, "fig_dataset_maps.png")


# ────────────────────────────────────────────────────────────────── bins ──
def fig_bins(map_id=99):
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    rec = pool[map_id]
    H, W = rec.terrain.shape
    wall = int(rec.wall_col)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(13.0, 13.0 * 40 / 64))
        ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
        ax.set_xlim(-.5, W - .5); ax.set_ylim(H + 4.2, -3.0)
        nb = len(D.BIN_EDGES) - 1
        for b in range(nb):
            lo = max(D.BIN_EDGES[b] + wall, 0); hi = min(D.BIN_EDGES[b + 1] + wall, W)
            col = PHASE_COL[D.PHASE_OF_BIN[b]]
            ax.add_patch(Rectangle((lo - .5, -.5), hi - lo, H, facecolor=col,
                                   alpha=.13 + .07 * (b % 2), edgecolor="none", zorder=3))
            ax.plot([lo - .5, lo - .5], [-.5, H - .5], color="black", lw=2.0, zorder=5)
            if b == nb - 1:
                ax.plot([hi - .5, hi - .5], [-.5, H - .5], color="black", lw=2.0, zorder=5)
            mid = (lo + hi) / 2 - .5
            ax.text(mid, -1.3, LABELS[b], ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=col, zorder=6)
            ax.text(mid, H + 1.4, f"{hi - lo} col", ha="center", va="top",
                    fontsize=7, color="#6b7280", zorder=6)
        runs, start = [], 0
        for b in range(1, nb + 1):
            if b == nb or D.PHASE_OF_BIN[b] != D.PHASE_OF_BIN[start]:
                runs.append((start, b - 1, D.PHASE_OF_BIN[start])); start = b
        for b0, b1, phase in runs:
            lo = max(D.BIN_EDGES[b0] + wall, 0) - .5; hi = min(D.BIN_EDGES[b1 + 1] + wall, W) - .5
            y = H + 2.6
            ax.plot([lo, hi], [y, y], color=PHASE_COL[phase], lw=2.4, solid_capstyle="butt", zorder=6)
            ax.text((lo + hi) / 2, y + 0.7, PHASE_LABEL[phase], ha="center", va="top",
                    fontsize=8, color=PHASE_COL[phase], zorder=6)
        ax.plot(rec.spawn[1], rec.spawn[0], "o", color="white", mec="black", ms=7, zorder=8)
        ax.annotate("spawn", (rec.spawn[1], rec.spawn[0]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color="white", zorder=8)
        for cells, name in ((rec.top_goal_cells, "top"), (rec.bottom_goal_cells, "bottom")):
            good = rec.correct_target in ("either", name)
            for (r, c) in cells:
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                       edgecolor="#22c55e" if good else "#ef4444", lw=1.8, zorder=8))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        fig.tight_layout()
        save(fig, "fig_bins.png")


# ─────────────────────────────────────────────────────────────────── pca ──
RAMP = {"lakes": "GnBu", "rocky": "YlOrRd", "balanced": "Greys"}


def shade(cat, b, nb):
    return matplotlib.colormaps[RAMP[cat]](0.34 + 0.60 * b / max(nb - 1, 1))


def fig_pca():
    from sklearn.decomposition import PCA
    X, df = D.load("ppo")
    tr, _ = D.split_maps(df)
    bins = D.bin_states(X, df)
    nb = len(D.BIN_EDGES) - 1
    P, C, B = [], [], []
    for b in range(nb):
        ids, cats, M = bins[b]
        P.append(M); C.append(cats); B.append(np.full(len(ids), b))
    P = np.concatenate(P); C = np.concatenate(C); B = np.concatenate(B)
    lr = C != "balanced"
    pca = PCA(n_components=3).fit(P[lr])           # the lakes/rocky plane
    Z = pca.transform(P); ev = pca.explained_variance_ratio_
    seg, frac = [], []
    for b in range(nb):
        ids, cats, M = bins[b]
        m = np.isin(ids, tr)
        mu_l = M[m][cats[m] == "lakes"].mean(0); mu_r = M[m][cats[m] == "rocky"].mean(0)
        v = mu_r - mu_l; v = v / (np.linalg.norm(v) + 1e-12)
        seg.append((pca.transform(mu_l[None])[0], pca.transform(mu_r[None])[0]))
        frac.append(float(np.linalg.norm(pca.components_[[0, 1]] @ v)))
    stats = dict(explained_variance=[float(x) for x in ev],
                 axis_fraction_in_plane={LABELS[b]: frac[b] for b in range(nb)})

    def key(ax, cats):
        x0, y0, dx, dy = .013, .975, .020, .048
        for i, cat in enumerate(cats):
            ax.text(x0, y0 - i * dy + .012, cat, transform=ax.transAxes, fontsize=8.2, va="bottom")
            for b in range(nb):
                ax.add_patch(plt.Rectangle((x0 + .085 + b * dx, y0 - i * dy - .004), dx * .88, .024,
                                           transform=ax.transAxes, facecolor=shade(cat, b, nb),
                                           edgecolor="none", clip_on=False, zorder=9))
        ax.text(x0 + .085, y0 - len(cats) * dy + .006, "early", transform=ax.transAxes, fontsize=6.8, color=INK2)
        ax.text(x0 + .085 + (nb - 1) * dx, y0 - len(cats) * dy + .006, "passage",
                transform=ax.transAxes, fontsize=6.8, color=INK2, ha="right")

    with plt.rc_context(RC):
        # (a) lakes / rocky with the per-bin axis
        fig, ax = plt.subplots(figsize=(8.4, 6.0))
        for cat in ("lakes", "rocky"):
            for b in range(nb):
                m = (C == cat) & (B == b)
                ax.scatter(Z[m, 0], Z[m, 1], s=4.6, alpha=.45, color=shade(cat, b, nb), lw=0, zorder=2)
        for b, (p_l, p_r) in enumerate(seg):
            ok = frac[b] >= .25
            ax.plot([p_l[0], p_r[0]], [p_l[1], p_r[1]], color=INK, lw=2.2 if ok else 1.2,
                    ls="-" if ok else ":", alpha=1 if ok else .4, solid_capstyle="round", zorder=6)
            ax.scatter([p_l[0]], [p_l[1]], s=44, color=shade("lakes", b, nb), ec="black", lw=.8, zorder=7)
            ax.scatter([p_r[0]], [p_r[1]], s=44, color=shade("rocky", b, nb), ec="black", lw=.8, zorder=7)
            ax.annotate(LABELS[b], ((p_l[0] + p_r[0]) / 2, (p_l[1] + p_r[1]) / 2), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=7.4, color=INK, zorder=8)
        ax.plot([], [], color=INK, lw=2.2, label="difference of means (lakes $\\to$ rocky), per bin")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}% of variance)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}% of variance)")
        ax.grid(alpha=.18, lw=.5)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + (hi - lo) * .26)
        key(ax, ("lakes", "rocky"))
        fig.tight_layout(); save(fig, "fig_pca_lakes_rocky.png")

        # (b) balanced maps in the same plane
        fig, ax = plt.subplots(figsize=(8.4, 6.0))
        ax.scatter(Z[lr, 0], Z[lr, 1], s=3, alpha=.10, color="#9ca3af", lw=0, zorder=1,
                   label="lakes and rocky states (reference)")
        means = []
        for b in range(nb):
            m = (C == "balanced") & (B == b)
            ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=.55, color=shade("balanced", b, nb), lw=0, zorder=3)
            means.append(Z[m, :2].mean(0))
        means = np.array(means)
        ax.plot(means[:, 0], means[:, 1], "-", color=INK, lw=1.4, zorder=6)
        for b in range(nb):
            ax.scatter([means[b, 0]], [means[b, 1]], s=52, color=shade("balanced", b, nb), ec="black", lw=.8, zorder=7)
            ax.annotate(LABELS[b], means[b], xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=7.4, color=INK, zorder=8)
        for b, (p_l, p_r) in enumerate(seg):
            if b in (0, nb - 1):
                ax.scatter([p_l[0]], [p_l[1]], s=40, color=shade("lakes", b, nb), ec="black", lw=.8, zorder=5)
                ax.scatter([p_r[0]], [p_r[1]], s=40, color=shade("rocky", b, nb), ec="black", lw=.8, zorder=5)
                ax.annotate(f"lakes {LABELS[b]}", p_l[:2], xytext=(0, -11), textcoords="offset points",
                            ha="center", fontsize=6.8, color=C_LAKES)
                ax.annotate(f"rocky {LABELS[b]}", p_r[:2], xytext=(0, -11), textcoords="offset points",
                            ha="center", fontsize=6.8, color=C_ROCKY)
        ax.plot([], [], "-", color=INK, lw=1.4, label="balanced maps: per-bin mean")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}% of variance)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}% of variance)")
        ax.grid(alpha=.18, lw=.5)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + (hi - lo) * .20)
        key(ax, ("balanced",))
        fig.tight_layout(); save(fig, "fig_pca_balanced.png")
    (OUTS[1] / "pca_stats.json").write_text(json.dumps(stats, indent=1))


# ──────────────────────────────────────────────────────── results: belief ──
def fig_belief():
    res = json.loads((REPO / "outputs/belief_report/probes.json").read_text())
    A = res["agents"]["ppo"]["bins"]
    bs = sorted(A, key=int)
    nb = len(bs)
    keys = A[bs[0]].keys()
    kd = [k for k in keys if k.startswith("dm") and k.endswith("acc2")][0]
    log = np.array([A[b]["logistic_acc2"] for b in bs]); ci = np.array([A[b]["logistic_ci2"] for b in bs])
    mlp = np.array([A[b]["mlp_acc2"] for b in bs]); dm = np.array([A[b][kd] for b in bs])
    # per-bin directions -> cosine matrix. Panel (b) uses the weight vector of a
    # full-state logistic probe (rocky vs lakes) fitted on the RAW states of each bin
    # (no standardisation: the scaler-mapped weights of the standardised probe are
    # dominated by one near-constant unit, sigma ~ 6e-5). The difference-of-means
    # directions are computed too and both matrices are saved in belief_stats.json.
    from sklearn.linear_model import LogisticRegression
    X, df = D.load("ppo"); tr, _ = D.split_maps(df); bins = D.bin_states(X, df)
    V, Wp = [], []
    for b in range(nb):
        ids, cats, M = bins[b]; m = np.isin(ids, tr) & np.isin(cats, ["rocky", "lakes"])
        v, _ = D.fit_dm(M[m], cats[m]); V.append(v)
        y = (cats[m] == "rocky").astype(int)
        w = LogisticRegression(max_iter=5000, C=1.0).fit(M[m], y).coef_[0]; Wp.append(w / np.linalg.norm(w))
    C_dm = np.stack(V) @ np.stack(V).T
    C_probe = np.stack(Wp) @ np.stack(Wp).T
    Cm = C_dm            # panel (b) shows the difference-of-means directions; the probe matrix is kept in belief_stats.json
    with plt.rc_context(RC):
        # explicit axes in inches so every panel keeps its natural proportions
        W, Hf = 15.2, 4.5
        def inch(x, y, w, h): return fig.add_axes([x / W, y / Hf, w / W, h / Hf])
        fig = plt.figure(figsize=(W, Hf))
        top = 3.9                                             # common top edge of the three panels
        axa = inch(0.75, top - 3.0, 3.7, 3.0)                 # (a) 3.7 x 3.0
        axb = inch(5.55, top - 3.0, 3.0, 3.0)                 # (b) square
        cax = inch(8.68, top - 3.0, 0.12, 3.0)                # its colour bar
        axc = inch(10.1, top - 2.45, 4.9, 2.45)               # (c) 4.9 x 2.45, top-aligned
        axes = [axa, axb]
        ax = axes[0]; x = np.arange(nb)
        for b in range(nb):
            ax.axvspan(b - .5, b + .5, color=PHASE_COL[D.PHASE_OF_BIN[b]], alpha=.06, lw=0, zorder=0)
        ax.fill_between(x, ci[:, 0], ci[:, 1], color="#2a78d6", alpha=.15, lw=0)
        ax.plot(x, log, "-o", color="#2a78d6", lw=2, ms=5, label="logistic probe, full state", zorder=4)
        ax.plot(x, mlp, "--s", color="#eb6834", lw=1.6, ms=4.5, label="MLP probe, full state", zorder=4)
        ax.plot(x, dm, ":^", color=INK, lw=1.6, ms=5, label="single direction $\\mathbf{v}$", zorder=5)
        ax.axhline(.5, color=MUTE, lw=.8, ls="--", zorder=1)
        ax.text(nb - .6, .515, "chance", ha="right", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=40, ha="right")
        ax.set_ylim(0, 1.03); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.set_ylabel("held-out accuracy, rocky vs lakes"); ax.set_xlabel("position bin")
        ax.set_title("(a) belief readout per bin", loc="left", color=INK)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax = axes[1]
        im = ax.imshow(Cm, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(nb):
            for j in range(nb):
                ax.text(j, i, f"{Cm[i, j]:.2f}".lstrip("0").replace("-0.", "-."), ha="center", va="center",
                        fontsize=6.2, color="white" if abs(Cm[i, j]) > .62 else INK)
        for k in (4.5, 6.5):
            ax.axhline(k, color=INK, lw=1.2); ax.axvline(k, color=INK, lw=1.2)
        ax.set_xticks(range(nb)); ax.set_yticks(range(nb))
        ax.set_xticklabels(LABELS, rotation=40, ha="right", fontsize=7.4); ax.set_yticklabels(LABELS, fontsize=7.4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title("(b) cosine between the per-bin directions", loc="left", color=INK)
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=7.5)
        st = _sigmoid_panel(axc, title_prefix="(c) ")
        save(fig, "fig_results_belief.png")
    off = Cm[~np.eye(nb, dtype=bool)]
    (OUTS[1] / "belief_stats.json").write_text(json.dumps(dict(
        bins=LABELS, logistic=log.tolist(), mlp=mlp.tolist(), single_direction=dm.tolist(),
        cosine=Cm.round(4).tolist(), off_diag_mean=float(off.mean()), off_diag_min=float(off.min()),
        cosine_dm=C_dm.round(4).tolist(), cosine_probe=C_probe.round(4).tolist(),
        probe_vs_dm_diag=[float(a @ b) for a, b in zip(Wp, V)]), indent=1))


# ────────────────────────────────────────────────────────── results: steer ──
def fig_steer(site="corr2", xmax=1.0, tag="finectrl", out="fig_results_causal.png",
              xlabel=r"steering strength $\alpha$"):
    # `tag="finectrl"` is the 0.1-step dose ladder with the matched random-direction control
    # (steer_alpha.py --alphas 0,...,2 --control --tag finectrl);
    # falls back to the coarse ladder if the fine file has not been produced yet.
    path = REPO / f"outputs/belief_report/steer_alpha_ppo_{site}_{tag}.json"
    if not path.exists(): path = REPO / f"outputs/belief_report/steer_alpha_ppo_{site}.json"
    d = json.loads(path.read_text())
    alphas = [x for x in d["alphas"] if x <= xmax + 1e-9]
    cols = [("lakes", [(+1, C_ROCKY, "pushed to rocky")], "lakes maps"),
            ("rocky", [(-1, C_LAKES, "pushed to lakes")], "rocky maps"),
            ("balanced", [(+1, C_ROCKY, "pushed to rocky"), (-1, C_LAKES, "pushed to lakes")], "balanced maps")]

    has_ctrl = any(r.get("kind") == "control" for r in d["rows"])

    def curve(cat, sgn, kind="steer"):
        xs, ys, es, tos = [], [], [], []
        for al in alphas:
            g = [r for r in d["rows"] if r["cat"] == cat and r["sign"] == sgn and r["alpha"] == al
                 and r.get("kind", "steer") == kind]
            if not g: continue
            top = np.array([r["door"] == "top" for r in g], float); p, n = top.mean(), len(g)
            xs.append(al); ys.append(p); es.append(np.sqrt(p * (1 - p) / n))
            tos.append(np.mean([r["door"] not in ("top", "bottom") for r in g]))
        return map(np.array, (xs, ys, es, tos))

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), sharey=True)
        for ax, (cat, arms, title) in zip(axes, cols):
            for sgn, col, lab in arms:
                x, y, e, to = curve(cat, sgn)
                dense = len(x) > 12
                ax.errorbar(x, y, yerr=e, color=col, lw=1.8, ms=3.2 if dense else 4.5, marker="o",
                            capsize=1.5 if dense else 2.5, elinewidth=1, label=lab)
                bad = to > .2
                if bad.any(): ax.plot(x[bad], y[bad], "x", color=INK, ms=9, mew=2, zorder=6)
                if has_ctrl:
                    xc, yc, ec, _ = curve(cat, sgn, "control")
                    ax.errorbar(xc, yc, yerr=ec, color="#9ca3af", lw=1.2, ls="--", ms=2.5, marker="o",
                                capsize=1.2, elinewidth=.8, zorder=2,
                                label="random direction" if (sgn > 0 or cat != "balanced") else None)
            ax.axhline(.5, color="#d1d5db", lw=.8, zorder=0)
            ax.set_ylim(-.03, 1.03); ax.set_xlabel(xlabel)
            ax.set_xticks(np.arange(0, xmax + 1e-9, 0.2 if xmax <= 1 else 0.5)); ax.set_xticks(alphas, minor=True)
            ax.set_title(title, loc="left", color=INK); ax.legend(fontsize=7.5, loc="best", frameon=False)
        axes[0].set_ylabel("P(top flag)")
        fig.tight_layout(); save(fig, out)
    print("  n maps per category:", d["n"], " site:", site, " file:", path.name, " alphas:", len(alphas))


FIGS = dict(dataset=fig_dataset, bins=fig_bins, pca=fig_pca, belief=fig_belief, steer=fig_steer)



# ──────────────────────────────────────────────────────── results: sigmoid ──
def _sigmoid_panel(ax, title_prefix=""):
    """Per-map belief scalar at corr2 by true category, with the 1-D logistic. Returns stats."""
    from sklearn.linear_model import LogisticRegression
    z = np.load(REPO / "outputs/belief_report/steer_axis_ppo.npz"); v = z["v"] / np.linalg.norm(z["v"])
    X, df = D.load("ppo"); tr, te = D.split_maps(df); bins = D.bin_states(X, df)
    ids, cats, M = bins[6]
    s = M @ v
    fit = np.isin(ids, tr) & np.isin(cats, ["lakes", "rocky"])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(s[fit, None], (cats[fit] == "rocky").astype(int))
    test = np.isin(ids, te) & np.isin(cats, ["lakes", "rocky"])
    acc = clf.score(s[test, None], (cats[test] == "rocky").astype(int))
    mid = -clf.intercept_[0] / clf.coef_[0, 0]
    rng = np.random.default_rng(0)
    rows = {"lakes": 0, "balanced": 1, "rocky": 2}
    col = {"lakes": C_LAKES, "balanced": C_BAL, "rocky": C_ROCKY}
    for cat in rows:
        m = cats == cat
        ax.scatter(s[m], rows[cat] + rng.uniform(-.18, .18, m.sum()), s=7, alpha=.55, color=col[cat], lw=0, zorder=3)
    xs = np.linspace(s.min() - .3, s.max() + .3, 400)
    ax.plot(xs, 2 * clf.predict_proba(xs[:, None])[:, 1], color=INK, lw=2, zorder=4)
    ax.axvline(mid, color=INK2, lw=1, ls="--", zorder=2)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["lakes", "balanced", "rocky"])
    ax.set_ylim(-.5, 2.5); ax.set_xlabel(r"belief scalar $\hat b=\mathbf{h}^{\top}\mathbf{v}$, last corridor bin")
    ax.set_title(f"{title_prefix}belief scalar by category", loc="left", color=INK)
    ax.text(.02, .97, f"{len(ids):,} held-out maps\n1-D readout accuracy {acc:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, color=INK2)
    ax.text(mid + .15, 2.45, "decision boundary", fontsize=7.5, color=INK2, va="top")
    ax.grid(axis="x", color="#e8e7e3", lw=.6, zorder=0)
    return dict(n_maps=int(len(ids)), n_fit=int(fit.sum()), n_test=int(test.sum()), acc_test=float(acc), midpoint=float(mid),
                coef=float(clf.coef_[0, 0]), intercept=float(clf.intercept_[0]),
                per_category_mean={c: float(s[cats == c].mean()) for c in rows},
                balanced_frac_rocky_side=float((s[cats == "balanced"] > mid).mean()))


def fig_sigmoid():
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.0, 3.3))
        st = _sigmoid_panel(ax)
        fig.tight_layout(); save(fig, "fig_results_sigmoid.png")
    (OUTS[1] / "sigmoid_stats.json").write_text(json.dumps(st, indent=1))


FIGS["sigmoid"] = fig_sigmoid
FIGS["steer_set"] = lambda: fig_steer(tag="set", out="fig_results_causal_set.png", xlabel=r"dose $\lambda$")



# ───────────────────────────────────────────── pca: one panel per category ──
def fig_pca_categories():
    """Lakes, rocky and balanced maps each in the lakes/rocky PCA plane, with the
    per-bin mean path; the full lakes+rocky cloud is drawn faintly in every panel."""
    from sklearn.decomposition import PCA
    X, df = D.load("ppo")
    bins = D.bin_states(X, df)
    nb = len(D.BIN_EDGES) - 1
    P, C, B = [], [], []
    for b in range(nb):
        ids, cats, M = bins[b]
        P.append(M); C.append(cats); B.append(np.full(len(ids), b))
    P = np.concatenate(P); C = np.concatenate(C); B = np.concatenate(B)
    lr = C != "balanced"
    pca = PCA(n_components=2).fit(P[lr]); Z = pca.transform(P); ev = pca.explained_variance_ratio_
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharex=True, sharey=True)
        for ax, cat in zip(axes, ("lakes", "balanced", "rocky")):
            ax.scatter(Z[lr, 0], Z[lr, 1], s=3, alpha=.07, color="#9ca3af", lw=0, zorder=1)
            means = []
            for b in range(nb):
                m = (C == cat) & (B == b)
                ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=.5, color=shade(cat, b, nb), lw=0, zorder=3)
                means.append(Z[m].mean(0))
            means = np.array(means)
            ax.plot(means[:, 0], means[:, 1], "-", color=INK, lw=1.4, zorder=6)
            for b in range(nb):
                ax.scatter([means[b, 0]], [means[b, 1]], s=52, color=shade(cat, b, nb), ec="black", lw=.8, zorder=7)
                dy = 7 if b % 2 == 0 else -11
                ax.annotate(LABELS[b], means[b], xytext=(0, dy), textcoords="offset points", ha="center",
                            fontsize=7, color=INK, zorder=8)
            ax.set_title(f"{cat} maps, per-bin mean", loc="left", color=INK)
            ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}% of variance)")
            ax.grid(alpha=.18, lw=.5)
        axes[0].set_ylabel(f"PC2 ({ev[1]*100:.0f}% of variance)")
        lo, hi = axes[0].get_ylim(); axes[0].set_ylim(lo, hi + (hi - lo) * .12)
        fig.tight_layout(); save(fig, "fig_pca_categories.png")


FIGS["pca_categories"] = fig_pca_categories



# ───────────────────────────────────────────────────────── table 1 (LaTeX) ──
def fig_table():
    """Table 1: the act-11 clamp on every eligible map, with the 1-D probe's P(rocky)."""
    from sklearn.linear_model import LogisticRegression
    z = np.load(REPO / "outputs/belief_report/steer_axis_ppo.npz"); v = z["v"] / np.linalg.norm(z["v"])
    X, df = D.load("ppo"); tr, _ = D.split_maps(df); ids, cats, M = D.bin_states(X, df)[6]
    fit = np.isin(ids, tr) & np.isin(cats, ["lakes", "rocky"])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit((M[fit] @ v)[:, None], (cats[fit] == "rocky").astype(int))
    P = lambda b: float(clf.predict_proba(np.array(b, float)[:, None])[:, 1].mean())
    res = {}
    for cat in ("balanced", "lakes", "rocky"):
        rows = json.loads((REPO / f"outputs/behavior_steering/act11/rows_all_{cat}.json").read_text())
        res[cat] = {}
        for arm in ("unsteered", "suppress bridge", "suppress tunnel"):
            rs = [r for r in rows if r["arm"] == arm]
            res[cat][arm] = dict(bridges=np.mean([r["builds"] for r in rs]), tunnels=np.mean([r["mines"] for r in rs]),
                                 timeout=np.mean([r["timeout"] for r in rs]), rocky=np.mean([r["door"] == "top" for r in rs]),
                                 lakes=np.mean([r["door"] == "bottom" for r in rs]),
                                 p_rocky=P([r["proj"] for r in rs if r["proj"] == r["proj"]]))
    def cell(cat, arm, k):
        d = res[cat][arm]; low = min(res[cat][a][k] for a in res[cat]); t = f"{d[k]:.2f}"
        return f"\\textbf{{{t}}}" if d[k] == low else t
    n = {c: json.loads((REPO / f"outputs/behavior_steering/act11/summary_all_{c}.json").read_text())["n_maps"]
         for c in ("balanced", "lakes", "rocky")}
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}",
         r"\caption{\textbf{Behaviour steering on every eligible held-out map.} Gated gradient clamp at the frozen operating points, six sampled rollouts per map and regime with identical seeds; a map is eligible when two unsteered rollouts used both tools (" + f"{n['balanced']} balanced, {n['lakes']} lakes, {n['rocky']} rocky maps" + r"). \emph{\# bridges} and \emph{\# tunnels} are successful tool uses per episode, the lowest of each category in bold; \emph{timeout}, \emph{rocky flag} and \emph{lakes flag} are shares of episodes and sum to one; $\hat b(\text{rocky})$ is the mean probability that the map is rocky under the logistic probe on the single belief scalar $\hat b$. On lakes maps the rocky flag pays nothing and on rocky maps the lakes flag pays nothing; those cells are in red.}",
         r"\label{tab:clamp_all}", r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"& & & & \multicolumn{3}{c}{share of episodes} & \\", r"\cmidrule(lr){5-7}",
         r"Category & Regime & \# bridges & \# tunnels & timeout & rocky flag & lakes flag & $\hat b(\text{rocky})$ \\", r"\midrule"]
    for i, cat in enumerate(("balanced", "lakes", "rocky")):
        if i: L.append(r"\midrule")
        for j, arm in enumerate(("unsteered", "suppress bridge", "suppress tunnel")):
            d = res[cat][arm]; rk, lk = f"{d['rocky']:.2f}", f"{d['lakes']:.2f}"
            if cat == "lakes" and arm == "suppress bridge": rk = f"\\textcolor{{red}}{{\\textbf{{{rk}}}}}"
            if cat == "rocky" and arm == "suppress tunnel": lk = f"\\textcolor{{red}}{{\\textbf{{{lk}}}}}"
            L.append(f"{cat if j == 0 else ''} & {arm} & {cell(cat, arm, 'bridges')} & {cell(cat, arm, 'tunnels')} & "
                     f"{d['timeout']:.2f} & {rk} & {lk} & {d['p_rocky']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUTS[0] / "tab_clamp_all_eligible.tex").write_text("\n".join(L) + "\n"); print("wrote tab_clamp_all_eligible.tex")


FIGS["table"] = fig_table


if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    for n in (list(FIGS) if names == ["all"] else names):
        FIGS[n]()
