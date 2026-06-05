#!/usr/bin/env python3
"""Standalone decoder for a bridge_tunnel(_commit) PPO activation dataset.

NO repository / cogniland dependency: reads only the bundle next to this file
(`manifest.json`, `maps.npz`, `labels.*`, `activations.h5`). Renders a single
labeled frame, a trajectory path on its map, or a trajectory video. Every output
is captioned with the dataset `row_id` so any frame pins back to a dataset row.

    python decode_dataset.py --row 12345                       # one labeled frame
    python decode_dataset.py --traj <map_id> <traj_seed> --plot path.png
    python decode_dataset.py --traj <map_id> <traj_seed> --video out.mp4

Deps: numpy, matplotlib, pandas, h5py (h5py only if activations are .h5).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE = Path(__file__).resolve().parent
# commitment → path colour (matches the trajectory grids): none/build/mine
_COMMIT_COL = {"none": "#1f5fd0", "build": "#ffd000", "mine": "#a800e6", "": "#1f5fd0"}


def load_bundle(d: Path):
    man = json.loads((d / "manifest.json").read_text())
    palette = np.asarray(man["tile_colors"], dtype=np.uint8)
    maps = np.load(d / "maps.npz", allow_pickle=True)
    import pandas as pd
    if (d / "labels.parquet").exists():
        labels = pd.read_parquet(d / "labels.parquet")
    else:
        labels = pd.read_csv(d / "labels.csv")
    return man, palette, maps, labels


def open_acts(d: Path):
    if (d / "activations.h5").exists():
        import h5py
        return h5py.File(d / "activations.h5", "r")
    return np.load(d / "activations.npz")


def _caption(row, is_commit):
    s = f"row_id={int(row.row_id)}  map {int(row.map_id)}  t={int(row.t)}  a={row.action_name}"
    if is_commit and "commit_state" in row.index:
        s += f"  commit={row.commit_state}"
        if "category" in row.index:
            s += f"  cat={row.category}"
    return s


def cmd_row(d, man, palette, maps, labels, row_id, out):
    is_commit = man.get("is_commit", False)
    row = labels.loc[labels.row_id == row_id].iloc[0]
    mid = int(row.map_id); terr = maps["terrain"][mid]
    acts = open_acts(d); mm = np.asarray(acts["minimap"][row_id])
    fig, (axm, axo) = plt.subplots(1, 2, figsize=(11, 4.6))
    axm.imshow(palette[terr], interpolation="nearest")
    axm.scatter([row.pos_c], [row.pos_r], c="white", s=60, edgecolors="k", zorder=3)
    axm.set_title("map (white = agent)"); axm.set_xticks([]); axm.set_yticks([])
    axo.imshow(palette[mm], interpolation="nearest")
    axo.set_title(f"egocentric obs {mm.shape}"); axo.set_xticks([]); axo.set_yticks([])
    fig.suptitle(_caption(row, is_commit), fontsize=12)
    fig.tight_layout()
    out = out or (d / f"frame_row{row_id}.png")
    fig.savefig(out, dpi=120); print(f"saved {out}")


def _traj_rows(labels, map_id, traj_seed):
    sub = labels[(labels.map_id == map_id) & (labels.traj_seed == traj_seed)].sort_values("t")
    if len(sub) == 0:
        raise SystemExit(f"no rows for map_id={map_id} traj_seed={traj_seed}")
    return sub


def cmd_plot(d, man, palette, maps, labels, map_id, traj_seed, out):
    is_commit = man.get("is_commit", False)
    sub = _traj_rows(labels, map_id, traj_seed)
    terr = maps["terrain"][map_id]
    pos = sub[["pos_r", "pos_c"]].to_numpy(float)
    commit = sub["commit_state"].tolist() if (is_commit and "commit_state" in sub) else ["none"] * len(sub)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.imshow(palette[terr], interpolation="nearest")
    xy = np.stack([pos[:, 1], pos[:, 0]], 1)
    segs = np.stack([xy[:-1], xy[1:]], 1)
    cols = [_COMMIT_COL.get(commit[i + 1], "#1f5fd0") for i in range(len(segs))]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=2.0))
    ax.scatter([pos[0, 1]], [pos[0, 0]], c="white", s=45, edgecolors="k", marker="s", zorder=4)
    reached = bool(sub.iloc[-1].reached)
    ttl = f"map {map_id}  traj_seed {traj_seed}  {'reached' if reached else 'timeout'}"
    if is_commit:
        ttl += f"  cat={sub.iloc[0].category}  final={sub.iloc[-1].final_commit}"
    ax.set_title(ttl + "   (line: blue none / yellow build / purple mine)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); out = out or (d / f"traj_{map_id}_{traj_seed}.png")
    fig.savefig(out, dpi=120); print(f"saved {out}")


def cmd_video(d, man, palette, maps, labels, map_id, traj_seed, out):
    import matplotlib.animation as animation
    is_commit = man.get("is_commit", False)
    sub = _traj_rows(labels, map_id, traj_seed).reset_index(drop=True)
    terr = maps["terrain"][map_id]; rids = sub.row_id.to_numpy()
    acts = open_acts(d)
    pos = sub[["pos_r", "pos_c"]].to_numpy(float)
    fig, (axm, axo) = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax in (axm, axo):
        ax.set_xticks([]); ax.set_yticks([])
    axm.imshow(palette[terr], interpolation="nearest")
    (dot,) = axm.plot([], [], "ws", ms=8, mec="k")
    (line,) = axm.plot([], [], "-", color="cyan", lw=1.5)
    obs_im = axo.imshow(palette[np.asarray(acts["minimap"][rids[0]])], interpolation="nearest")

    def upd(i):
        r = sub.iloc[i]
        dot.set_data([pos[i, 1]], [pos[i, 0]])
        line.set_data(pos[:i + 1, 1], pos[:i + 1, 0])
        obs_im.set_data(palette[np.asarray(acts["minimap"][rids[i]])])
        fig.suptitle(_caption(r, is_commit), fontsize=12)
        return dot, line, obs_im

    ani = animation.FuncAnimation(fig, upd, frames=len(sub), interval=120, blit=False)
    out = out or (d / f"traj_{map_id}_{traj_seed}.mp4")
    try:
        ani.save(str(out), writer=animation.FFMpegWriter(fps=8))
    except Exception as e:   # noqa: BLE001
        out = Path(str(out).rsplit(".", 1)[0] + ".gif")
        print(f"  [ffmpeg unavailable: {e!r}] writing {out} instead")
        ani.save(str(out), writer=animation.PillowWriter(fps=8))
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=HERE, help="bundle dir (default: next to this script)")
    p.add_argument("--row", type=int, default=None)
    p.add_argument("--traj", nargs=2, type=int, metavar=("MAP_ID", "TRAJ_SEED"), default=None)
    p.add_argument("--plot", nargs="?", const=True, default=None)
    p.add_argument("--video", nargs="?", const=True, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    d = args.dataset
    man, palette, maps, labels = load_bundle(d)
    if args.row is not None:
        cmd_row(d, man, palette, maps, labels, args.row, args.out)
    elif args.traj is not None:
        mid, ts = args.traj
        if args.video is not None:
            cmd_video(d, man, palette, maps, labels, mid, ts,
                      args.out or (None if args.video is True else Path(args.video)))
        else:
            cmd_plot(d, man, palette, maps, labels, mid, ts,
                     args.out or (None if args.plot in (True, None) else Path(args.plot)))
    else:
        raise SystemExit("give --row N  or  --traj MAP_ID TRAJ_SEED [--plot|--video]")


if __name__ == "__main__":
    main()
