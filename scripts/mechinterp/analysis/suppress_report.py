#!/usr/bin/env python3
"""Report for the probe-gated skill-suppression experiments.

Per model (ppo / dreamer):
  * summary table (reach, tunnel/bridge crossings per episode, mined/placed,
    interventions) across modes none|tunnel|bridge|both
  * trajectory grids: rows = eval maps, cols = [dataset (before), baseline
    rollouts, suppress tunnel, suppress bridge, suppress both]; paths overlaid
    on the original terrain. Tunnel crossings happen where paths traverse rock;
    bridge crossings where they traverse water.

    python -m scripts.mechinterp.analysis.suppress_report
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mechinterp.analysis.bundle import ActivationBundle  # noqa: E402

DIR = Path("outputs/suppression")
MODES = ("none", "tunnel", "bridge", "both")
COL_TITLES = {"dataset": "dataset (before)", "none": "baseline rollout",
              "tunnel": "suppress TUNNEL", "bridge": "suppress BRIDGE",
              "both": "suppress BOTH"}


def dataset_paths(bundle, mi, n=10):
    lab = bundle.labels
    g = lab[lab["map_id"] == mi]
    out = []
    for tid in sorted(g["traj_id"].unique())[:n]:
        t = g[g["traj_id"] == tid].sort_values("t")
        out.append(np.stack([t["pos_r"].to_numpy(), t["pos_c"].to_numpy()], 1))
    return out


def draw(ax, terrain, paths, title, reach=None, mined=None, placed=None):
    from cogniland.bridge_tunnel import tiles as T
    ax.imshow(T.TILE_COLORS[terrain], interpolation="nearest")
    for p in paths:
        ax.plot(p[:, 1], p[:, 0], color="#0a2a8a", lw=1.0, alpha=0.45)
        ax.scatter(p[-1, 1], p[-1, 0], s=6, color="#0a2a8a", alpha=0.6)
    if mined is not None and len(mined):                  # mined blocks = yellow
        ax.scatter(mined[:, 1], mined[:, 0], s=14, marker="s", color="#ffd800",
                   edgecolors="none", alpha=0.95, zorder=4)
    if placed is not None and len(placed):                # placed blocks = red
        ax.scatter(placed[:, 1], placed[:, 0], s=14, marker="s", color="#e0151a",
                   edgecolors="none", alpha=0.95, zorder=4)
    if paths:
        ax.scatter(paths[0][0, 1], paths[0][0, 0], s=28, facecolors="white",
                   edgecolors="k", zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    t = title if reach is None else f"{title}\nreach {reach:.0%}"
    ax.set_title(t, fontsize=8.5)


def grids(model, dataset, n_show_maps=4, n_show_traj=10,
          metrics_file=None, rollouts_file=None, tag=""):
    import pandas as pd
    b = ActivationBundle(f"activation_datasets/{dataset}")
    met = pd.read_csv(metrics_file or DIR / f"{model}_metrics.csv")
    roll = np.load(rollouts_file or DIR / f"{model}_rollouts.npz")
    ev_maps = roll["eval_maps"].tolist()
    terr = b.maps["terrain"]

    # show the maps where suppression had the most to do (baseline crossings)
    base = met[met["mode"] == "none"].groupby("map_id")[["tunnel_cross", "bridge_cross"]].mean()
    show = base.sum(1).sort_values(ascending=False).index.tolist()[:n_show_maps]

    cols = ["dataset"] + list(MODES)
    fig, axes = plt.subplots(len(show), len(cols),
                             figsize=(2.9 * len(cols), 1.62 * len(show)))
    axes = np.atleast_2d(axes)
    from cogniland.bridge_tunnel import tiles as T
    for r, mi in enumerate(show):
        for c, col in enumerate(cols):
            ax = axes[r, c]
            if col == "dataset":
                dps = dataset_paths(b, mi, n_show_traj)
                ap = np.concatenate(dps) if dps else np.zeros((0, 2), int)
                t0 = terr[mi][ap[:, 0], ap[:, 1]] if len(ap) else np.array([])
                mined = np.unique(ap[t0 == T.ROCK], axis=0) if len(ap) else None
                placed = np.unique(ap[t0 == T.WATER], axis=0) if len(ap) else None
                draw(ax, terr[mi], dps, COL_TITLES[col] if r == 0 else "",
                     mined=mined, placed=placed)
            else:
                ps, mn, pl = [], [], []
                for t in range(n_show_traj):
                    k = f"{col}/{mi}/{t}"
                    if k in roll:
                        ps.append(roll[k])
                        if f"{k}/mined" in roll:
                            mn.append(roll[f"{k}/mined"]); pl.append(roll[f"{k}/placed"])
                mined = np.unique(np.concatenate(mn), axis=0) if mn and sum(len(x) for x in mn) else None
                placed = np.unique(np.concatenate(pl), axis=0) if pl and sum(len(x) for x in pl) else None
                sub = met[(met["mode"] == col) & (met["map_id"] == mi)]
                draw(ax, terr[mi], ps, COL_TITLES[col] if r == 0 else "",
                     reach=sub["reached"].mean(), mined=mined, placed=placed)
            if c == 0:
                ax.set_ylabel(f"map {mi}", fontsize=9)
    fig.suptitle(f"{model.upper()} (BT) — probe-gated skill suppression ({tag}): "
                 "trajectories before / after", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = DIR / f"grid_{model}_{tag}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)
    return met


def table(model, met):
    agg = met.groupby("mode").agg(
        reach=("reached", "mean"), steps=("steps", "mean"),
        tunnel_cross=("tunnel_cross", "mean"), bridge_cross=("bridge_cross", "mean"),
        mined=("mined_rocks", "mean"), placed=("placed_water", "mean"),
        interventions=("n_interventions", "mean"), dnorm=("mean_dnorm", "mean"),
    ).reindex(list(MODES)).round(2)
    print(f"\n=== {model.upper()} summary (per-episode means, held-out maps) ===")
    print(agg.to_string())
    agg.to_csv(DIR / f"{model}_summary.csv")
    return agg


def compare_tags(model):
    """Condensed cross-config comparison over every saved metrics csv."""
    import pandas as pd
    rows = []
    for p in sorted(DIR.glob(f"{model}_metrics*.csv")):
        tag = p.stem.replace(f"{model}_metrics", "").lstrip("_") or "pure"
        met = pd.read_csv(p)
        for mode in MODES:
            d = met[met["mode"] == mode]
            rows.append(dict(config=tag, mode=mode, reach=d.reached.mean(),
                             tunnel=d.tunnel_cross.mean(), bridge=d.bridge_cross.mean(),
                             mined=d.mined_rocks.mean(), placed=d.placed_water.mean(),
                             interv=d.n_interventions.mean()))
    cmp_df = pd.DataFrame(rows).round(2)
    print(f"\n===== {model.upper()} all configs =====")
    print(cmp_df.to_string(index=False))
    cmp_df.to_csv(DIR / f"{model}_all_configs.csv", index=False)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-tag", default="hyb12")
    ap.add_argument("--dreamer-tag", default="pgd3")
    args = ap.parse_args()
    global grids_tag
    for model, dataset, tag in [("ppo", "bt_ppo", args.ppo_tag),
                                ("dreamer", "bt_dreamer", args.dreamer_tag)]:
        suffix = f"_{tag}" if tag else ""
        mfile = DIR / f"{model}_metrics{suffix}.csv"
        if not mfile.exists():
            print(f"({model}: {mfile} not found, skipping)")
            continue
        met = grids(model, dataset, metrics_file=mfile,
                    rollouts_file=DIR / f"{model}_rollouts{suffix}.npz", tag=tag)
        table(model, met)
        compare_tags(model)


if __name__ == "__main__":
    main()
