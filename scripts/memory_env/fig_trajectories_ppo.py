#!/usr/bin/env python
"""Reference trajectory figure for a PPO+GRU MemoryEnv model: greedy rollouts on
all four cue types (2x2 god's-eye panels), trained vs held-out annotated.
Terminology: cue features are DIRECTION (up/down) and COLOR (green/blue).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import jax
import orbax.checkpoint as ocp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

from cogniland.memory_env.jax import constants as C  # noqa: E402
import diag_jax as D  # noqa: E402
import train_ppo_memory as P  # noqa: E402
import video_rollout as VR  # noqa: E402
from viz_rollout_dream import TILE_RGB, CUE_MARK  # noqa: E402

ALL_CUES = ["green_up", "blue_up", "green_down", "blue_down"]


def taken_branch_ok(p, cue, xs, ys, dn):
    """Infer the taken branch from the trajectory; True iff it matches the cue direction."""
    nd = int(np.argmax(dn)) + 1 if dn.any() else len(xs)
    inb = (xs[:nd] >= p.x_branch_start) & (xs[:nd] <= p.x_branch_end)
    on_up = inb & (ys[:nd] == p.row_up)
    on_lo = inb & (ys[:nd] == p.row_lo)
    if not (on_up.any() or on_lo.any()):
        return None                              # never entered a branch
    first_up = np.argmax(on_up) if on_up.any() else 10**9
    first_lo = np.argmax(on_lo) if on_lo.any() else 10**9
    took_down = first_lo < first_up
    return took_down == bool(D.IS_DOWN[cue])


def draw(ax, p, s0, xs, ys, dn, reach, title, trained, dir_ok):
    terr = np.asarray(p.base_terrain)
    full = terr.copy()
    ct = int(np.asarray(s0.cue_type))
    full[int(s0.cue_y), int(s0.cue_x)] = C.CUE_TILE[ct]
    dgt = bool(np.asarray(s0.door_green_top))
    full[p.row_door_top, p.x_doorcol] = C.DOOR_GREEN if dgt else C.DOOR_BLUE
    full[p.row_door_bot, p.x_doorcol] = C.DOOR_BLUE if dgt else C.DOOR_GREEN
    ax.imshow(TILE_RGB[full], interpolation="nearest")
    for r in range(full.shape[0]):
        for c in range(full.shape[1]):
            if full[r, c] in CUE_MARK:
                ax.text(c, r, CUE_MARK[full[r, c]], ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
    nd = int(np.argmax(dn)) + 1 if dn.any() else len(xs)
    xx = np.concatenate([[int(s0.agent_x)], xs[:nd]])
    yy = np.concatenate([[int(s0.agent_y)], ys[:nd]])
    ax.plot(xx, yy, "-", color="#888", lw=1.3, alpha=0.85, zorder=4)
    ax.scatter(xx, yy, c=np.arange(len(xx)), cmap="autumn", s=13, zorder=5,
               edgecolor="k", lw=0.2)
    ax.scatter([xx[0]], [yy[0]], marker="s", c="red", s=40, zorder=6)
    door_ok = bool(reach[nd - 1]) if dn.any() else False
    dpart = ("correct direction" if dir_ok else
             ("wrong direction" if dir_ok is not None else "no branch"))
    fpart = "correct door" if door_ok else ("wrong door" if dn.any() else "no door (truncated)")
    all_ok = bool(dir_ok) and door_ok
    ax.set_title(f"{title}  [{'trained' if trained else 'HELD OUT'}]  ->  {dpart}, {fpart}",
                 fontsize=9.5, color=("#1a7d36" if all_ok else "#b02418"),
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmax", type=int, default=110)
    a = ap.parse_args()
    rd = pathlib.Path(a.run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    params = ocp.PyTreeCheckpointer().restore(
        str(sorted((rd / "checkpoints").glob("step_*"))[-1].resolve()))["params"]
    net = P.ActorCriticRNN(action_dim=C.NUM_ACTIONS, view_size=cfg["view_size"],
                           token_dim=cfg["token_dim"], embed_hidden=cfg["embed_hidden"],
                           gru_hidden=cfg["gru_hidden"])
    trained = set(D.TRAIN_CUES[cfg["cue"]])
    key = jax.random.PRNGKey(3)
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.2))
    # alternate door layouts across columns so both are displayed
    for ax, cue, fdoor in zip(axes.flat, ALL_CUES, [True, False, False, True]):
        key, k = jax.random.split(key)
        p, s0, (xs, ys, ds, mms, hs, dn, reach) = VR.rollout(
            cfg, net, params, cue, k, a.tmax, fdoor)
        layout = "green-top" if fdoor else "blue-top"
        dir_ok = taken_branch_ok(p, cue, np.asarray(xs), np.asarray(ys), np.asarray(dn))
        draw(ax, p, s0, xs, ys, dn, reach, f"{cue} · doors {layout}", cue in trained, dir_ok)
    fig.suptitle(f"{cfg['cue']} model — greedy trajectories on all four cues "
                 "(direction -> branch, color -> door; dot colour = time)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(a.out, dpi=140)
    print(f"[fig] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
