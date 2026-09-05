#!/usr/bin/env python3
"""Is the crossing plan in the state before the obstacle is reached?  (rectangular obstacles;
every step pooled into two-column bins, 5 folds grouped by episode per bin)

Toy maps: one balanced held-out map with all terrain replaced by grass, plus a
single circular obstacle (a lake or a mountain) on the spawn row. The radius is
chosen per obstacle so that the stochastic released agent crosses (bridges /
tunnels) on about half of its rollouts and walks around on the other half.
A 2-class logistic regression on the full GRU state, grouped 5-fold over
episodes, then decodes the eventual strategy at every distance before contact.

  sbatch scripts/bridge_tunnel/slurm/iab_crossing_plan.sbatch     (CPU, ~5 min)
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ("src", "scripts/figures"):
    sys.path.insert(0, str(REPO / p))

OUTS = [REPO / "paper/iab2026/paper/figures", REPO / "paper/figures/iab2026"]
DATA = REPO / "outputs/iab2026"
CKPT = REPO / "final_models/ppo/ppo_plain_noaux.pt"
BASE_ID, CX, ROW = 99, 28, 16          # balanced test map 99; obstacle centre
RADII = [(3, hh) for hh in (2, 3, 4, 5, 6, 7, 8, 10, 12)]   # rectangle: (width in columns, half-height in rows)
KINDS = {"lake": "WATER", "mountain": "ROCK"}
_G = {}


def toy_map(pool, kind, r):
    from cogniland.bridge_tunnel import tiles as T
    rec = copy.deepcopy(pool[BASE_ID])
    terr = rec.terrain
    keep = rec.wall_col - 1
    terr[:, 1:keep] = T.GRASS
    tile = getattr(T, KINDS[kind])
    w, hh = r
    H = terr.shape[0]
    x0 = CX - w // 2
    terr[max(1, ROW - hh):min(H - 1, ROW + hh + 1), x0:x0 + w] = tile
    return rec


def _init():
    import torch
    torch.set_num_threads(1)
    from paper_rollouts import make_ppo
    _G["act"], _G["reset"] = make_ppo(str(CKPT), sampled=True)
    _G["pool"] = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))


def episode(job):
    import torch
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS
    kind, r, seed = job
    rec = toy_map(_G["pool"], kind, r)
    np.random.seed(seed); torch.manual_seed(seed)
    act, reset = _G["act"], _G["reset"]
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset(); reset()
    hs, cols, rows, n_ev = [], [], [], 0
    for t in range(FORKWALL_KWARGS["max_steps"]):
        pr, pc = env._pos
        a = act(obs, False)
        hs.append(act.get_state().astype(np.float16))
        cols.append(int(pc)); rows.append(int(pr))
        obs, _, term, trunc, info = env.step(a)
        if info.get("placed") or info.get("mined"):
            n_ev += 1
        if term or trunc:
            break
    return dict(kind=kind, r=r, seed=seed, cols=np.array(cols, np.int16), rows=np.array(rows, np.int16),
                h=np.stack(hs), cross=n_ev > 0, n_events=n_ev,
                success=bool(env._pos in (env._correct_cells or set())), steps=len(cols))


def run(jobs, workers):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        return list(ex.map(episode, jobs, chunksize=4))


BINS = [(20, 19), (18, 17), (16, 15), (14, 13), (12, 11), (10, 9), (8, 7), (6, 5), (4, 3), (2, 1), (0, -1)]
BIN_LABELS = [f"{hi}–{lo}" if lo >= 0 else f"{hi}–−{-lo}" for hi, lo in BINS]


def collect(E, contact, dhi, dlo, feat):
    """Every step whose distance d = contact - col lies in [dlo, dhi], all episodes."""
    X, Y, G = [], [], []
    for i, e in enumerate(E):
        d = contact - e["cols"].astype(int)
        for t in np.where((d <= dhi) & (d >= dlo))[0]:
            X.append(e["h"][t].astype(np.float32) if feat == "h" else np.array([e["rows"][t], t], float))
            Y.append(int(e["cross"])); G.append(i)
    return np.array(X), np.array(Y), np.array(G)


def fit_fold(X, Y, trn, tst):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    sc = StandardScaler().fit(X[trn])
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5).fit(sc.transform(X[trn]), Y[trn])
    return float(balanced_accuracy_score(Y[tst], clf.predict(sc.transform(X[tst]))))


def probe_binned(E, contact, feat, workers):
    """5 x 11 classifiers: per two-column bin, five folds grouped by episode."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold
    tasks, meta = [], []
    for bi, (hi, lo) in enumerate(BINS):
        X, Y, G = collect(E, contact, hi, lo, feat)
        if len(Y) < 50 or len(set(Y)) < 2:
            meta.append(dict(bin=BIN_LABELS[bi], n_steps=int(len(Y)), n_episodes=int(len(set(G))), folds=[])); continue
        folds = list(GroupKFold(n_splits=5).split(X, Y, G))
        meta.append(dict(bin=BIN_LABELS[bi], n_steps=int(len(Y)), n_episodes=int(len(set(G))), folds=[]))
        for trn, tst in folds:
            tasks.append((bi, X, Y, trn, tst))
    accs = Parallel(n_jobs=workers)(delayed(fit_fold)(X, Y, trn, tst) for _, X, Y, trn, tst in tasks)
    for (bi, *_), acc in zip(tasks, accs):
        meta[bi]["folds"].append(acc)
    for m in meta:
        m["acc"] = float(np.mean(m["folds"])) if m["folds"] else float("nan")
        m["sd"] = float(np.std(m["folds"])) if m["folds"] else float("nan")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=44)
    ap.add_argument("--n-screen", type=int, default=100)
    ap.add_argument("--n-probe", type=int, default=400)
    ap.add_argument("--figs-only", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if not a.figs_only:
        # 1. radius screen
        jobs = [(k, r, 1000 * ri + s) for k in KINDS for ri, r in enumerate(RADII) for s in range(a.n_screen)]  # noqa
        got = run(jobs, a.workers)
        screen = {}
        for k in KINDS:
            for r in RADII:
                g = [e for e in got if e["kind"] == k and e["r"] == r]
                screen[f"{k}_{r[0]:g}x{r[1]:g}"] = dict(kind=k, r=r, p_cross=float(np.mean([e["cross"] for e in g])),
                                          success=float(np.mean([e["success"] for e in g])), n=len(g))
                print(f"  {k:9s} r={r[0]:g}x{r[1]:g}  P(cross) {screen[f'{k}_{r[0]:g}x{r[1]:g}']['p_cross']:.2f}  success {screen[f'{k}_{r[0]:g}x{r[1]:g}']['success']:.2f}", flush=True)
        chosen = {}
        for k in KINDS:
            cand = [v for v in screen.values() if v["kind"] == k and v["success"] >= .9]
            chosen[k] = tuple(min(cand, key=lambda v: abs(v["p_cross"] - .5))["r"])
            print(f"  chosen semi-axes {k}: {chosen[k]}", flush=True)
        # 2. probe set at the chosen radius
        jobs = [(k, chosen[k], 50_000 + s) for k in KINDS for s in range(a.n_probe)]
        eps = run(jobs, a.workers)
        pickle.dump(dict(screen=screen, chosen=chosen, eps=eps), open(DATA / "crossing_plan_rollouts.pkl", "wb"))
    d = pickle.load(open(DATA / "crossing_plan_rollouts.pkl", "rb"))
    screen, chosen, eps = d["screen"], d["chosen"], d["eps"]
    res = {}
    for k in KINDS:
        E = [e for e in eps if e["kind"] == k]
        contact = CX - chosen[k][0] // 2
        res[k] = dict(r=list(chosen[k]), contact_col=int(contact), n=len(E),
                      p_cross=float(np.mean([e["cross"] for e in E])),
                      success=float(np.mean([e["success"] for e in E])),
                      curve=probe_binned(E, contact, "h", a.workers),
                      curve_pos=probe_binned(E, contact, "pos", a.workers))
        print(f"  {k}: r={chosen[k]} n={len(E)} P(cross)={res[k]['p_cross']:.2f} success={res[k]['success']:.2f}", flush=True)
    (DATA / "crossing_plan.json").write_text(json.dumps(dict(screen=screen, chosen={k: list(v) for k, v in chosen.items()}, results=res), indent=1))

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cogniland.bridge_tunnel import tiles as T
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))
    INK, INK2, MUTE = "#0b0b0b", "#52514e", "#a8a7a1"
    COL = {"lake": "#2a78d6", "mountain": "#eb6834"}
    RC = {"figure.dpi": 200, "savefig.dpi": 200, "font.size": 9, "axes.spines.top": False,
          "axes.spines.right": False, "axes.edgecolor": MUTE, "xtick.color": INK2, "ytick.color": INK2,
          "axes.labelcolor": INK2, "figure.facecolor": "white", "axes.facecolor": "white"}
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(12.0, 3.9))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=.25, wspace=.12)
        ax = fig.add_subplot(gs[:, 0])
        xs = np.arange(len(BINS))
        for k in KINDS:
            cv, cp = res[k]["curve"], res[k]["curve_pos"]
            y = np.array([c["acc"] for c in cv]); sd = np.array([c["sd"] for c in cv])
            yp = np.array([c["acc"] for c in cp]); sdp = np.array([c["sd"] for c in cp])
            ax.plot(xs, yp, "--", color=COL[k], lw=1.3, alpha=.8, zorder=3, label=f"{k}: row and time only")
            ax.fill_between(xs, yp - sdp, yp + sdp, color=COL[k], alpha=.08, lw=0, zorder=1)
            ax.errorbar(xs, y, yerr=sd, fmt="-o", color=COL[k], lw=2, ms=4.5, capsize=2.5, elinewidth=1,
                        zorder=4, label=f"{k}: full state $\\mathbf{{h}}_t$")
        ax.axvline(4.5, color=INK, lw=1.0, ls=":", zorder=2)
        ax.text(4.6, .06, "obstacle enters view", rotation=90, va="bottom", ha="left", fontsize=7.5, color=INK2)
        ax.axhline(.5, color=MUTE, lw=.9, ls="--", zorder=1)
        ax.text(0, .515, "chance", fontsize=7.5, color=INK2, va="bottom")
        ax.set_xticks(xs); ax.set_xticklabels(BIN_LABELS, fontsize=8)
        ax.set_xlim(-.5, len(BINS) - .5); ax.set_ylim(0, 1.03)
        ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("columns before contact, two-column bins")
        ax.set_ylabel("balanced accuracy, eventual strategy")
        ax.set_title("(a) cross or go around, decoded from every step in the bin", loc="left", color=INK)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        for i, k in enumerate(KINDS):
            axm = fig.add_subplot(gs[i, 1])
            rec = toy_map(pool, k, tuple(chosen[k]))
            axm.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
            shown = {True: 0, False: 0}
            shades = [0.45, 0.65, 0.88]                      # one shade per rollout
            for e in [e for e in eps if e["kind"] == k and e["success"]]:
                if shown[e["cross"]] >= 3: continue
                cmap = matplotlib.colormaps["Reds" if e["cross"] else "Blues"]
                axm.plot(e["cols"], e["rows"], color=cmap(shades[shown[e["cross"]]]), lw=1.5, alpha=.95)
                shown[e["cross"]] += 1
            axm.plot(rec.spawn[1], rec.spawn[0], "o", color="white", mec="black", ms=4)
            axm.set_xticks([]); axm.set_yticks([])
            for s in axm.spines.values(): s.set_edgecolor("#c9cfc8")
            axm.set_title(f"({'bc'[i]}) {k}, {res[k]['r'][0]:g}$\\times${2 * res[k]['r'][1] + 1:g} cells, P(cross) {res[k]['p_cross']:.2f}: cross (reds) vs around (blues)", loc="left", fontsize=8, color=INK)
        for o in OUTS:
            o.mkdir(parents=True, exist_ok=True); fig.savefig(o / "fig_crossing_plan.png", bbox_inches="tight")
        plt.close(fig)
    print("wrote fig_crossing_plan.png")


if __name__ == "__main__":
    main()
