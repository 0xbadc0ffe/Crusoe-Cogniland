#!/usr/bin/env python3
"""Thesis figure 7.2 -- training return over seeds, one panel per agent.

Three panels (PPO + GRU, DreamerV3, STORM), each a mean +/- standard-deviation
band over that agent's seeds, all sharing one y-scale and one x-tick interval.

Return is normalised to a 0-100 per cent scale,

    return_pct = 100 * (r - r_random) / (r_optimal - r_random)

so 0 per cent is a uniform-random policy and 100 per cent is optimal play. Both
anchors are measured by scripts/figures/paper/compute_max_return.py and read
from outputs/belief_report/max_return.json. Plain r/r_optimal would not work:
raw returns start near -8 against an optimum near +3, so early training would
sit far below -100 per cent. The green line is the 100 per cent ceiling.

Data sources, all keyed the same way (environment frames -> undiscounted
episode return):

  PPO      outputs/ppo_noaux/noaux_ent15_s*/metrics.jsonl   "return/rolling100"
  Dreamer  r2dreamer_model/runs/fw_seed_*/metrics.jsonl     "episode/score"
  STORM    STORM_model/results/*/metrics.jsonl              "episode/score"
           or outputs/storm_train_curves/seed*.jsonl        (released seeds,
                                                             recovered from the
                                                             offline W&B logs)

PPO already logs a 100-episode rolling mean. Dreamer and STORM log raw
per-episode scores, so they are smoothed to a comparable window before the
panels are placed side by side.

  python scripts/figures/paper/fig_reward_curves.py
  python scripts/figures/paper/fig_reward_curves.py --storm-source runs
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "paper/figures/forkwall_paper"
MAX_RETURN_JSON = REPO / "outputs/belief_report/max_return.json"

# one tick per million frames on every panel: PPO/STORM run to 6M and Dreamer to
# 3M, so 0.5 would crowd the two wide panels with 13 labels each
X_TICK_M = 1.0
GRID_POINTS = 240

COL_PPO = "#d97706"
COL_DREAMER = "#2563eb"
COL_STORM = "#16a34a"
# darker + dashed so it stays distinguishable from the (also green) STORM curve
COL_MAX = "#15803d"
COL_RANDOM = "#9aa0a8"

RC = {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
      "axes.spines.top": False, "axes.spines.right": False}


# ── data ────────────────────────────────────────────────────────────────────

def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a run still being written can end mid-line
    return rows


def series(path, key, step_key="step"):
    """(frames, value) for one run, dropping missing/NaN entries."""
    xs, ys = [], []
    for row in read_jsonl(path):
        val = row.get(key)
        step = row.get(step_key, row.get("frame"))
        if val is None or step is None:
            continue
        val = float(val)
        if np.isnan(val):
            continue
        xs.append(float(step))
        ys.append(val)
    return np.array(xs), np.array(ys)


def smooth(xs, ys, frac=40, lo=5, hi=100):
    """Rolling mean over ~1/frac of the run, to match PPO's rolling100."""
    if ys.size < 2 * lo:
        return xs, np.full_like(ys, np.nan)
    k = int(min(hi, max(lo, ys.size // frac)))
    out = np.convolve(ys, np.ones(k) / k, mode="same")
    out[:k] = np.nan
    out[-k:] = np.nan                      # convolution edge effects
    return xs, out


def ppo_seeds():
    out = []
    for f in sorted(glob.glob(str(REPO / "outputs/ppo_noaux/noaux_ent15_s*/metrics.jsonl"))):
        xs, ys = series(f, "return/rolling100")
        if xs.size > 10:
            out.append((xs, ys))
    return out


def dreamer_seeds():
    out = []
    for d in sorted(glob.glob(str(REPO / "r2dreamer_model/runs/fw_seed_*"))):
        f = Path(d) / "metrics.jsonl"
        if not f.exists():
            continue
        xs, ys = series(f, "episode/score")
        if xs.size > 10:
            out.append(smooth(xs, ys))
    return out


def storm_seeds(source="auto", budget=6_000_000, done_frac=0.95):
    """(seeds, label) for STORM.

    ``runs``    fresh training runs that write metrics.jsonl directly.
    ``archive`` the released seeds, recovered from the offline W&B datastores
                by scripts/figures/paper/extract_storm_wandb_returns.py.
    ``auto``    the fresh runs once all of them have essentially finished,
                otherwise the archive, otherwise nothing.
    """
    def load(paths):
        out = []
        for f in sorted(paths):
            xs, ys = series(f, "episode/score")
            if xs.size > 10:
                out.append(smooth(xs, ys))
        return out

    runs = load(glob.glob(str(REPO / "STORM_model/results/*/metrics.jsonl")))
    arch = load(glob.glob(str(REPO / "outputs/storm_train_curves/seed*.jsonl")))
    complete = bool(runs) and all(s[0].max() >= done_frac * budget for s in runs)

    if source == "runs":
        return runs, "fresh runs"
    if source == "archive":
        return arch, "released seeds"
    if complete:
        return runs, "fresh runs"
    if arch:
        return arch, "released seeds"
    return [], "none"


def anchors():
    """(r_random, r_optimal) -- the 0 and 100 per cent ends of the y-scale."""
    if not MAX_RETURN_JSON.exists():
        raise SystemExit(
            f"missing {MAX_RETURN_JSON}\n"
            "run: python scripts/figures/paper/compute_max_return.py")
    d = json.loads(MAX_RETURN_JSON.read_text())
    return float(d["r_random"]), float(d["r_optimal"])


def to_pct(ys, r_random, r_optimal):
    """Raw episode return -> per cent of the random..optimal span."""
    return 100.0 * (ys - r_random) / (r_optimal - r_random)


# ── drawing ─────────────────────────────────────────────────────────────────

def band(seeds):
    """Mean/std over seeds on a shared grid, truncated at the shortest seed.

    Past the shortest seed the band would be built from a shrinking subset of
    seeds, which shows up as a spurious jump in the tail.
    """
    valid = [(xs[~np.isnan(ys)], ys[~np.isnan(ys)]) for xs, ys in seeds]
    valid = [(xs, ys) for xs, ys in valid if xs.size >= 2]
    # start where every seed already has data, so no grid column is all-NaN
    xmin = max(xs.min() for xs, _ in valid)
    xmax = min(xs.max() for xs, _ in valid)
    grid = np.linspace(xmin, xmax, GRID_POINTS)
    stack = np.vstack([np.interp(grid, xs, ys) for xs, ys in valid])
    return grid, stack.mean(axis=0), stack.std(axis=0)


def panel(ax, seeds, title, colour):
    ax.set_title(f"{title}  (n={len(seeds)} seed{'s' if len(seeds) != 1 else ''})"
                 if seeds else title, loc="left", fontsize=9.5)
    ax.set_xlabel("environment frames (M)")
    ax.xaxis.set_major_locator(MultipleLocator(X_TICK_M))
    if not seeds:
        ax.text(.5, .5, "training in progress", transform=ax.transAxes,
                ha="center", va="center", color="#9aa0a8", fontsize=9)
        return None
    grid, mean, std = band(seeds)
    if len(seeds) > 1:
        ax.fill_between(grid / 1e6, mean - std, mean + std,
                        color=colour, alpha=.22, lw=0)
    ax.plot(grid / 1e6, mean, color=colour, lw=1.8)
    return {"mean_lo": float(mean.min()), "mean_hi": float(mean.max()),
            "band_lo": float((mean - std).min()),
            "band_hi": float((mean + std).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storm-source", choices=("auto", "runs", "archive"),
                    default="auto")
    ap.add_argument("--out", default=str(OUT / "fig_res_reward.png"))
    args = ap.parse_args()

    r_random, r_optimal = anchors()
    ppo = ppo_seeds()
    drm = dreamer_seeds()
    storm, storm_label = storm_seeds(args.storm_source)

    # normalise every seed of every agent through the same transform, so the
    # mean/std band below is computed on per-cent values
    pct = lambda seeds: [(xs, to_pct(ys, r_random, r_optimal)) for xs, ys in seeds]
    agents = [("PPO + GRU", pct(ppo), COL_PPO),
              ("DreamerV3", pct(drm), COL_DREAMER),
              ("STORM", pct(storm), COL_STORM)]

    spans = {}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
        for ax, (title, seeds, colour) in zip(axes, agents):
            span = panel(ax, seeds, title, colour)
            if span:
                spans[title] = span

        for i, ax in enumerate(axes):
            ax.set_ylim(0, 100)                      # fixed 0-100 on every panel
            if i:
                ax.set_yticklabels([])
        axes[0].set_ylabel("Return (%)")

        fig.suptitle("Training return over time, mean $\\pm$ standard deviation "
                     "across seeds", y=1.02, fontsize=11)
        fig.tight_layout()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, bbox_inches="tight")

    print(f"wrote {args.out}")
    print(f"  PPO      {len(ppo)} seeds")
    print(f"  Dreamer  {len(drm)} seeds")
    print(f"  STORM    {len(storm)} seeds ({storm_label})")
    print(f"  anchors: r_random={r_random:.4f}  r_optimal={r_optimal:.4f}")
    print(f"  y-limits 0..100 on all panels; x tick interval {X_TICK_M} M frames")
    for name, s in spans.items():
        flag = "" if 0 <= s["band_lo"] and s["band_hi"] <= 100 else "   <-- OUTSIDE 0-100"
        print(f"  {name:<10} mean {s['mean_lo']:7.2f}..{s['mean_hi']:7.2f}%"
              f"   band {s['band_lo']:7.2f}..{s['band_hi']:7.2f}%{flag}")


if __name__ == "__main__":
    main()
