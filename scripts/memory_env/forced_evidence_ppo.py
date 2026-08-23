#!/usr/bin/env python
"""Behavior forcing -> belief revision through EVIDENCE (marker-door env).

The agent is action-forced (hidden state never touched) from the fork through
the WRONG branch corridor, opening its marker door — a branch-identity
observation that never co-occurs with the true cue in training — and released.
Readout: does the cue-identity belief update on that evidence, and does the
final door follow the (possibly revised) belief?

Predictions from the training correlations:
  2cue: wrong marker is only consistent with the OTHER training cue ->
        belief collapses to the other attractor -> wrong-color door.
  4cue: wrong marker only carries direction evidence -> belief moves to the
        direction-flipped SAME-color cue -> door stays color-correct.

Outputs (per run-dir): a quant table (door / success / P(cue) at the pre-door
readout) and marker-open-aligned mean belief curves P(cue = c)(t).

Usage:
  python forced_evidence_ppo.py --run-dirs outputs/ppo_runs/ppo_2cue_mk2 ... \
      --n 96 --policy sample --out outputs/forced_evidence.png
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import jax

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

import steer2_ppo as S  # noqa: E402
import diag_jax as D  # noqa: E402
from cogniland.memory_env.jax import constants as C  # noqa: E402


def pre_door_index(mms, dnn):
    """Per-episode index of the last step before any final door tile is visible."""
    door_vis = np.isin(mms, [C.DOOR_GREEN, C.DOOR_BLUE]).any(axis=(2, 3))
    done_before = np.zeros(door_vis.shape, bool)
    done_before[1:] = np.cumsum(dnn[:-1], axis=0) > 0
    tg = np.arange(door_vis.shape[0])[:, None]
    firstvis = np.where(door_vis & ~done_before, tg, door_vis.shape[0]).min(0)
    lastalive = np.where(~done_before, tg, -1).max(0)
    idx = np.where(firstvis < door_vis.shape[0], firstvis - 1, lastalive)
    return np.clip(idx, 0, door_vis.shape[0] - 1)


def mark_open_index(mms, wrong_is_down):
    """Per-episode step of the wrong-marker OPEN event, read off the minimap
    trace: the first step t where the closed marker tile was in view at t-1 and
    is gone at t (the tile turns EMPTY in place when opened). Returns -1 for
    episodes where the marker was never opened.
    """
    tile = C.MARK_B if wrong_is_down else C.MARK_A
    vis = (mms == tile).any(axis=(2, 3))                     # (T, n)
    T = vis.shape[0]
    opened = vis[:-1] & ~vis[1:]                             # visible -> gone
    tg = np.arange(T - 1)[:, None]
    idx = np.where(opened, tg, T).min(0)                     # first such step
    return np.where(idx < T, idx + 1, -1)                    # step AFTER open; -1 = never


def run(run_dirs, out, n=96, sample=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white"})
    W_PRE, W_POST = 8, 14          # event-aligned window around marker open
    ncols = 4                      # up to 4 trained cues per model
    fig, axs = plt.subplots(len(run_dirs), ncols,
                            figsize=(3.6 * ncols, 2.9 * len(run_dirs)),
                            squeeze=False, sharey=True)
    key = jax.random.PRNGKey(11)

    for row, rdir in enumerate(run_dirs):
        cfg, net, params, pr = S.load_all(rdir)
        trained, clf = pr["trained"], pr["clf"]
        print(f"\n== {cfg['cue']} model — FORCED through the WRONG corridor "
              f"(action replacement, marker opened; policy="
              f"{'softmax' if sample else 'greedy'}, n={n})")
        print(f"   {'cue':11s} | {'door-correct':>12s} {'success':>8s} | "
              + "  ".join(f"P({S.CUE_NAMES[t]:>10s})" for t in trained)
              + "  (pre-door readout)")
        for col in range(ncols):
            if col >= len(trained):
                axs[row, col].axis("off")
                continue
            c = trained[col]
            cue = S.CUE_NAMES[c]
            p_env = D._env_params(cfg, cue)
            wrong_is_down = not S.IS_DOWN[c]
            wrong_row = int(p_env.row_lo) if wrong_is_down else int(p_env.row_up)
            d_ok = C.SEL_BLUE if S.IS_BLUE[c] else C.SEL_GREEN
            key, k = jax.random.split(key)
            _, (succ, tb, sd, dacc, fin, hend), outs = S.rollout(
                cfg, net, params, cue, k, n, "force_thru", wrong_row=wrong_row,
                sample=sample, T=None)
            xs, ys, ds, mms, hs, win, dnn, reach, mtop, mbot = outs
            probs = S.cue_probs(hs.reshape(-1, hs.shape[-1]), clf).reshape(
                hs.shape[0], hs.shape[1], -1)                   # (T, n, K)
            idx = pre_door_index(mms, dnn)
            P_pre = probs[idx, np.arange(n)]                    # (n, K)
            dok = float((sd == d_ok).mean())
            print(f"   {cue:11s} | {dok:12.2f} {float(succ.mean()):8.2f} | "
                  + "  ".join(f"{P_pre[:, i].mean():13.2f}" for i in range(len(trained))))

            # event-aligned mean belief curves around the wrong-marker open
            t_open = mark_open_index(mms, wrong_is_down)
            ax = axs[row, col]
            rel = np.arange(-W_PRE, W_POST + 1)
            for i, t in enumerate(trained):
                curves = []
                for e in range(n):
                    if t_open[e] < 0:
                        continue
                    ts = t_open[e] + rel
                    ok = (ts >= 0) & (ts < probs.shape[0])
                    v = np.full(rel.shape, np.nan)
                    v[ok] = probs[ts[ok], e, i]
                    curves.append(v)
                if curves:
                    m = np.nanmean(np.stack(curves), 0)
                    ax.plot(rel, m, "-", lw=2, color=S.CUE_COL[t],
                            label=f"P({S.CUE_NAMES[t]})")
            ax.axvline(0, color="#8e44ad", lw=1.4, alpha=0.8)
            ax.text(0.3, 1.01, "wrong marker\nopened", fontsize=6.5, color="#8e44ad")
            ax.axhline(1 / len(trained), ls=":", c="#999", lw=0.7)
            ax.set_ylim(-0.05, 1.1)
            ax.set_title(f"{cfg['cue']} · true cue {cue} · door-ok {dok:.2f}",
                         fontsize=9, fontweight="bold")
            ax.set_xlabel("steps from marker open", fontsize=8)
            if col == 0:
                ax.set_ylabel("P(cue)", fontsize=8)
            ax.legend(fontsize=6, loc="center left")
    fig.suptitle("Action-forced into the WRONG corridor: belief revision at the "
                 "marker-door evidence (hidden state never edited)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outp = pathlib.Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=150)
    print(f"\n[forced_evidence] wrote {outp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--out", default="outputs/forced_evidence.png")
    ap.add_argument("--policy", choices=["greedy", "sample"], default="sample")
    a = ap.parse_args()
    run(a.run_dirs, a.out, n=a.n, sample=(a.policy == "sample"))


if __name__ == "__main__":
    main()
