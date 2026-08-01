#!/usr/bin/env python3
"""Training curves for the fork_wall (no-commit, NO auxiliary belief loss)
PPO+GRU seed sweep: mean +/- std across seeds.

Reads the full-resolution history straight out of the OFFLINE wandb run
directories (no sync/network needed) by scanning the protobuf datastore, then
matches each run to its seed via the run's config.

Panels:
  (A) episode return vs env steps          mean +/- std band across seeds
  (B) success rate (correct door)          mean +/- std band across seeds
  (C) per-category success at end of training, per seed
  (D) episode length vs env steps

    python scripts/figures/forkwall_noaux_training_curves.py \
        --run-glob 'ppo_gru_forkwall_noaux_seed*' \
        --out paper/figures/forkwall_noaux_training.png
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
CATS = ["balanced", "lakes", "rocky"]
CAT_COLOR = {"balanced": "#5C6B57", "lakes": "#1E6FA6", "rocky": "#A3572A"}


def read_offline_history(run_dir: Path) -> tuple[dict, dict]:
    """(config, history) from an offline wandb run dir. history: key -> (steps, values)."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    def item_key(it):
        # nested_key is a protobuf repeated container, not a list — convert it
        nk = list(it.nested_key)
        return ".".join(nk) if nk else it.key

    def load_cfg(update, into):
        for it in update:
            try:
                v = json.loads(it.value_json)
            except Exception:
                continue
            into[item_key(it)] = v.get("value", v) if isinstance(v, dict) else v

    files = list(run_dir.glob("*.wandb"))
    if not files:
        return {}, {}
    ds = datastore.DataStore()
    ds.open_for_scan(str(files[0]))

    cfg, rows = {}, []
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break                       # truncated tail on an in-flight run
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        kind = rec.WhichOneof("record_type")
        if kind == "history":
            row = {}
            for it in rec.history.item:
                try:
                    row[item_key(it)] = json.loads(it.value_json)
                except Exception:
                    pass
            if row:
                rows.append(row)
        elif kind == "config":
            load_cfg(rec.config.update, cfg)
        elif kind == "run":
            # the initial config lands here, not in a `config` record
            load_cfg(rec.run.config.update, cfg)

    hist: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if rows:
        keys = {k for r in rows for k in r} - {"_step", "_runtime", "_timestamp"}
        for k in keys:
            pts = [(r["_step"], r[k]) for r in rows if k in r and "_step" in r
                   and isinstance(r[k], (int, float))]
            if pts:
                pts.sort()
                hist[k] = (np.array([p[0] for p in pts], dtype=float),
                           np.array([p[1] for p in pts], dtype=float))
    return cfg, hist


def resample(steps, vals, grid):
    """Interpolate one run onto a shared step grid (NaN outside its range)."""
    out = np.interp(grid, steps, vals, left=np.nan, right=np.nan)
    out[grid > steps.max()] = np.nan
    return out


def smooth(v, w):
    if w <= 1 or len(v) < w:
        return v
    k = np.ones(w) / w
    pad = np.concatenate([np.full(w - 1, v[0]), v])
    return np.convolve(pad, k, mode="valid")


def band(ax, grid, mat, color, label, smooth_w):
    """mean +/- std band across seeds, ignoring seeds that ended early."""
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # all-NaN columns are expected past the shortest run's end
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0)
    ok = np.isfinite(mu)
    g, mu, sd = grid[ok], mu[ok], sd[ok]
    mu_s, sd_s = smooth(mu, smooth_w), smooth(sd, smooth_w)
    ax.plot(g, mu_s, color=color, lw=1.9, label=label, zorder=3)
    ax.fill_between(g, mu_s - sd_s, mu_s + sd_s, color=color, alpha=0.20,
                    linewidth=0, zorder=2)
    return mu_s, sd_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-glob", default="ppo_gru_forkwall_noaux_seed*",
                   help="match against the wandb run NAME (config run_name)")
    p.add_argument("--wandb-dir", type=Path, default=REPO / "wandb")
    p.add_argument("--out", type=Path,
                   default=REPO / "paper/figures/forkwall_noaux_training.png")
    p.add_argument("--smooth", type=int, default=9, help="moving-average window")
    p.add_argument("--n-grid", type=int, default=400)
    p.add_argument("--seeds", default=None,
                   help="comma list restricting which seeds enter the figure, e.g. "
                        "'1,3,4' to use only the seeds that condition on the belief "
                        "(the constant-door seeds make a mean±std band meaningless, "
                        "since the population is bimodal rather than noisy)")
    p.add_argument("--highlight", type=int, default=None,
                   help="draw this seed as a solid emphasised line (the released model)")
    args = p.parse_args()

    pat = re.compile(glob.fnmatch.translate(args.run_glob))
    runs = {}
    for d in sorted(args.wandb_dir.glob("offline-run-*")):
        cfg, hist = read_offline_history(d)
        name = str(cfg.get("run_name", ""))
        if not name or not pat.match(name) or not hist:
            continue
        seed = cfg.get("seed")
        # keep the longest run per seed (guards against restarts)
        prev = runs.get(seed)
        cur_len = max((len(v[0]) for v in hist.values()), default=0)
        if prev is None or cur_len > prev[2]:
            runs[seed] = (name, hist, cur_len, cfg)
    if not runs:
        raise SystemExit(f"no offline runs matched {args.run_glob!r} in {args.wandb_dir}")

    if args.seeds:
        keep = {int(x) for x in args.seeds.split(",")}
        runs = {s: v for s, v in runs.items() if s in keep}
        if not runs:
            raise SystemExit(f"--seeds {args.seeds} matched none of the discovered runs")
    seeds = sorted(runs)
    print(f"found {len(seeds)} runs: " +
          ", ".join(f"seed{s} ({runs[s][0]}, {runs[s][2]} pts)" for s in seeds))
    if args.highlight is not None and args.highlight not in seeds:
        raise SystemExit(f"--highlight {args.highlight} is not among the selected seeds {seeds}")
    bc = {runs[s][3].get("belief_coef") for s in seeds}
    print(f"belief_coef across runs: {bc}  (expect {{0.0}} for the no-aux control)")

    max_step = min(runs[s][1]["return/mean"][0].max() for s in seeds
                   if "return/mean" in runs[s][1])
    grid = np.linspace(0, max_step, args.n_grid)

    def stack(key):
        rows = []
        for s in seeds:
            h = runs[s][1]
            if key in h:
                rows.append(resample(h[key][0], h[key][1], grid))
        return np.vstack(rows) if rows else None

    hl = args.highlight
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    m = stack("return/mean")
    mu, sd = band(ax, grid, m, "#1E6FA6", f"mean ± std ({len(seeds)} seeds)", args.smooth)
    for i, s in enumerate(seeds):
        if s == hl:
            continue
        ax.plot(grid, smooth(m[i], args.smooth), color="#1E6FA6", lw=0.7, alpha=0.35, zorder=1)
    if hl is not None:
        ax.plot(grid, smooth(m[seeds.index(hl)], args.smooth), color="#12324f", lw=2.2,
                ls=(0, (5, 1.6)), zorder=4, label=f"seed {hl} (released)")
    ax.set_title("(A) episode return", fontsize=11)
    ax.set_xlabel("environment steps"); ax.set_ylabel("mean episode return")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.15)

    ax = axes[0, 1]
    m = stack("success/mean")
    band(ax, grid, m, "#2F8F63", f"mean ± std ({len(seeds)} seeds)", args.smooth)
    for i, s in enumerate(seeds):
        if s == hl:
            continue
        ax.plot(grid, smooth(m[i], args.smooth), color="#2F8F63", lw=0.7, alpha=0.35, zorder=1)
    if hl is not None:
        ax.plot(grid, smooth(m[seeds.index(hl)], args.smooth), color="#14503a", lw=2.2,
                ls=(0, (5, 1.6)), zorder=4, label=f"seed {hl} (released)")
    ax.axhline(2/3, color="#B4791E", ls="--", lw=1.2,
               label="constant-door baseline (2/3)")
    ax.set_ylim(0, 1.02)
    ax.set_title("(B) success rate — category-correct door", fontsize=11)
    ax.set_xlabel("environment steps"); ax.set_ylabel("success")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.15)

    ax = axes[1, 0]
    width = 0.8 / max(len(seeds), 1)
    for i, s in enumerate(seeds):
        h = runs[s][1]
        vals = []
        for c in CATS:
            k = f"success/{c}"
            vals.append(float(np.nanmean(h[k][1][-10:])) if k in h else np.nan)
        xs = np.arange(len(CATS)) + i * width - 0.4 + width / 2
        ax.bar(xs, vals, width=width, label=f"seed {s}",
               color=[CAT_COLOR[c] for c in CATS], alpha=0.55 + 0.09 * i,
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CATS)
    ax.set_ylim(0, 1.02); ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_title("(C) per-category success at end of training (last 10 logs)", fontsize=11)
    ax.set_ylabel("success"); ax.legend(fontsize=8, ncol=len(seeds)); ax.grid(alpha=0.15, axis="y")

    ax = axes[1, 1]
    m = stack("rollout/episode_length")
    if m is not None:
        band(ax, grid, m, "#8c564b", f"mean ± std ({len(seeds)} seeds)", args.smooth)
    ax.set_title("(D) episode length", fontsize=11)
    ax.set_xlabel("environment steps"); ax.set_ylabel("steps"); ax.grid(alpha=0.15)
    ax.legend(fontsize=9)

    fig.suptitle("fork_wall (no-commit) PPO+GRU — NO auxiliary belief loss — "
                 f"{len(seeds)} seeds", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")

    # numeric summary
    summary = {"seeds": seeds, "belief_coef": sorted(str(x) for x in bc), "final": {}}
    print(f"\n{'seed':>5s} {'return':>16s} {'success':>16s} " +
          " ".join(f"{c:>10s}" for c in CATS))
    for s in seeds:
        h = runs[s][1]
        r = float(np.nanmean(h["return/mean"][1][-10:]))
        su = float(np.nanmean(h["success/mean"][1][-10:]))
        cats = {c: (float(np.nanmean(h[f"success/{c}"][1][-10:]))
                    if f"success/{c}" in h else float("nan")) for c in CATS}
        summary["final"][s] = {"return": r, "success": su, "per_category": cats}
        print(f"{s:>5d} {r:>16.3f} {su:>16.3f} " +
              " ".join(f"{cats[c]:>10.3f}" for c in CATS))
    rs = [summary["final"][s]["return"] for s in seeds]
    ss = [summary["final"][s]["success"] for s in seeds]
    print(f"{'mean':>5s} {np.mean(rs):>16.3f} {np.mean(ss):>16.3f}")
    print(f"{'std':>5s} {np.std(rs):>16.3f} {np.std(ss):>16.3f}")
    jp = args.out.with_suffix(".json")
    jp.write_text(json.dumps(summary, indent=2))
    print(f"saved {jp}")


if __name__ == "__main__":
    main()
