#!/usr/bin/env python3
"""Task figures 1 & 2 for the Cogniland paper.

  fig_task_categories.png  three map types side by side (horizontal)
  fig_task_anatomy.png     one trajectory + a door-to-door zoom + three
                           texture-rendered observations pulled from the run
                           (start / mid-corridor / at the fork)

Figure 2 absorbs what used to be a separate observation figure: the callouts
*are* the agent's input at three moments of the same episode.

Usage: PYTHONPATH=src python scripts/figures/paper_task_figs.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "figures"))

from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelEnv  # noqa: E402
from paper_rollouts import FORKWALL_KWARGS, make_ppo  # noqa: E402
from paper_rollouts_textured import BASE, FACE_SPRITE, OVERLAY, load_sprites  # noqa: E402

CATS = ("rocky", "balanced", "lakes")
CAT_DOOR = {"lakes": "bottom door", "rocky": "top door", "balanced": "either door"}
PLT_RC = {"figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
          "axes.titlesize": 9.5, "axes.labelsize": 9}


def load_by_cat(maps_path, n=8):
    with open(maps_path, "rb") as f:
        recs = pickle.load(f)
    out = {c: [] for c in CATS}
    for r in recs:
        if r.category in out and len(out[r.category]) < n:
            out[r.category].append(r)
    return recs, out


# ── Figure 1 ─────────────────────────────────────────────────────────────

def fig_categories(by_cat, out):
    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(1, 3, figsize=(13.4, 2.9))
        for ax, cat in zip(axes, CATS):
            rec = by_cat[cat][0]
            ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
            for cells, name in ((rec.top_goal_cells, "top"),
                                (rec.bottom_goal_cells, "bottom")):
                good = rec.correct_target in ("either", name)
                for (r, c) in cells:
                    ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                           edgecolor="#22c55e" if good else "#ef4444",
                                           lw=2.2, zorder=6))
            ax.plot(rec.spawn[1], rec.spawn[0], "o", color="white", mec="black",
                    ms=6, zorder=7)
            ax.set_title(f"{cat}   →   {CAT_DOOR[cat]}", loc="left")
            ax.set_xticks([]); ax.set_yticks([])
        handles = [
            Patch(facecolor=np.array(T.TILE_COLORS[T.WATER]) / 255, label="water"),
            Patch(facecolor=np.array(T.TILE_COLORS[T.ROCK]) / 255, label="rock"),
            Patch(facecolor=np.array(T.TILE_COLORS[T.TREE]) / 255, label="tree"),
            Line2D([], [], color="#22c55e", lw=2, label="rewarded door"),
            Line2D([], [], color="#ef4444", lw=2, label="decoy door"),
            Line2D([], [], marker="o", color="white", mec="black", ls="", label="spawn"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
                   bbox_to_anchor=(.5, -.06), fontsize=8.5)
        fig.suptitle("Cogniland map types", y=1.03, fontsize=11)
        fig.tight_layout(rect=[0, .02, 1, .97])
        fig.savefig(out / "fig_task_categories.png", bbox_inches="tight")
        plt.close(fig)


# ── Figure 2 ─────────────────────────────────────────────────────────────

def obs_rgb(crop, facing, sprites, tp):
    """Render one 21x21 observation to an RGB array using the Crafter tiles."""
    V = crop.shape[0]
    surf = pygame.Surface((V * tp, V * tp))
    surf.fill((8, 10, 8))
    for r in range(V):
        for c in range(V):
            t = int(crop[r, c])
            if t == T.OOB:
                continue
            surf.blit(sprites[BASE.get(t, "grass")], (c * tp, r * tp))
            if t in OVERLAY:
                surf.blit(sprites[OVERLAY[t]], (c * tp, r * tp))
    surf.blit(sprites[FACE_SPRITE[facing]], (V // 2 * tp, V // 2 * tp))
    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)).copy()


def fig_anatomy(by_cat, out, ppo_ckpt):
    pygame.init(); pygame.display.set_mode((1, 1))
    sprites = load_sprites(10)

    rec = by_cat["rocky"][1]
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset()
    act, reset = make_ppo(ppo_ckpt)
    reset()

    # record the whole episode, then choose the three most informative frames:
    # peak evidence, deepest point of the corridor, and the first sight of a door.
    traj, log = [env._pos], []
    wall = rec.wall_col
    for t in range(FORKWALL_KWARGS["max_steps"]):
        crop = np.asarray(obs["minimap"])
        log.append(dict(t=t, pos=env._pos, facing=env._facing, crop=crop.copy(),
                        evidence=int(((crop == T.WATER) | (crop == T.ROCK)).sum()),
                        doors=int((crop == T.TARGET).sum())))
        a = act(obs, False)
        obs, r, term, trunc, _ = env.step(a)
        traj.append(env._pos)
        if term or trunc:
            break
    traj = np.asarray(traj, dtype=float)

    i_ev = max(range(len(log)), key=lambda i: log[i]["evidence"])
    # the blind moment: no category evidence left in view, doors not yet in view
    blind = [i for i in range(i_ev, len(log))
             if log[i]["evidence"] == 0 and log[i]["doors"] == 0]
    i_mid = blind[len(blind) // 2] if blind else min(i_ev + 10, len(log) - 1)
    i_fork = len(log) - 1                       # last frame before the choice

    shots = []
    for i, lab in ((i_ev, "peak evidence — rock and water in view"),
                   (i_mid, "the corridor — no evidence, no doors yet"),
                   (i_fork, "at the fork — both doors in view")):
        e = log[i]
        shots.append((f"t = {e['t']} · {lab}", e["pos"],
                      obs_rgb(e["crop"], e["facing"], sprites, 10)))

    Hm, Wm = rec.terrain.shape
    top_r = rec.top_goal_cells[0][0]
    bot_r = rec.bottom_goal_cells[0][0]
    pad_r = 2
    r0, r1 = max(0, top_r - pad_r), min(Hm, bot_r + pad_r + 1)   # door to door
    zh = r1 - r0
    c0 = max(0, Wm - zh)                                          # square crop
    c1 = Wm

    with plt.rc_context(PLT_RC):
        fig = plt.figure(figsize=(13.4, 7.4))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=.34, wspace=.14)

        axm = fig.add_subplot(gs[0, :2])
        axm.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
        mem_lo = max(0, wall - 16)
        axm.add_patch(Rectangle((mem_lo - .5, -.5), wall - mem_lo, Hm,
                                facecolor="black", alpha=.20, zorder=3))
        axm.plot(traj[:, 1], traj[:, 0], color="#fde68a", lw=1.8, zorder=6)
        axm.plot(traj[0, 1], traj[0, 0], "o", color="white", mec="black", ms=6, zorder=7)
        for cells, name in ((rec.top_goal_cells, "top"), (rec.bottom_goal_cells, "bottom")):
            good = rec.correct_target in ("either", name)
            for (r, c) in cells:
                axm.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                        edgecolor="#22c55e" if good else "#ef4444",
                                        lw=2.2, zorder=8))
        axm.add_patch(Rectangle((c0 - .5, r0 - .5), c1 - c0, zh, fill=False,
                                edgecolor="#facc15", lw=1.6, zorder=9))
        for x, y, txt in ((mem_lo / 2, 2.2, "1 · evidence\n(terrain reveals the type)"),
                          ((mem_lo + wall) / 2, 2.2, "2 · memory corridor\n(16 columns, no evidence)")):
            axm.annotate(txt, (x, y), color="white", fontsize=8, ha="center", va="center",
                         zorder=10, bbox=dict(boxstyle="round,pad=.25", fc="black",
                                              alpha=.72, ec="none"))
        axm.set_xticks([]); axm.set_yticks([])
        axm.set_title("(a) the full map — one PPO episode (yellow)", loc="left")

        axz = fig.add_subplot(gs[0, 2])
        axz.imshow(T.TILE_COLORS[rec.terrain[r0:r1, c0:c1]], interpolation="nearest")
        m = (traj[:, 0] >= r0) & (traj[:, 0] < r1) & (traj[:, 1] >= c0)
        axz.plot(traj[m, 1] - c0, traj[m, 0] - r0, color="#fde68a", lw=2.0, zorder=6)
        for cells, name in ((rec.top_goal_cells, "top"), (rec.bottom_goal_cells, "bottom")):
            good = rec.correct_target in ("either", name)
            for (r, c) in cells:
                axz.add_patch(Rectangle((c - c0 - .5, r - r0 - .5), 1, 1, fill=False,
                                        edgecolor="#22c55e" if good else "#ef4444",
                                        lw=2.4, zorder=8))
        if rec.passage_cells:
            pr = [c[0] for c in rec.passage_cells]; pc = rec.passage_cells[0][1]
            axz.add_patch(Rectangle((pc - c0 - .5, min(pr) - r0 - .5), 1, len(pr),
                                    fill=False, edgecolor="#38bdf8", lw=2.0, zorder=8))
            axz.annotate("3 · passage", (pc - c0 - 1.2, float(np.mean(pr)) - r0),
                         color="#38bdf8", fontsize=8, ha="right", va="center", zorder=10,
                         bbox=dict(boxstyle="round,pad=.2", fc="black", alpha=.72, ec="none"))
        axz.annotate("4 · door choice", (c1 - c0 - 1.5, (top_r - r0) - 1.5),
                     color="white", fontsize=8, ha="right", va="center", zorder=10,
                     bbox=dict(boxstyle="round,pad=.2", fc="black", alpha=.72, ec="none"))
        axz.set_xticks([]); axz.set_yticks([])
        for s in axz.spines.values():
            s.set_edgecolor("#facc15"); s.set_linewidth(1.6)
        axz.set_title("(b) the fork, door to door", loc="left")

        # three observations, wired back to where they were taken
        for k, (label, pos, img) in enumerate(shots[:3]):
            axo = fig.add_subplot(gs[1, k])
            axo.imshow(img, interpolation="nearest")
            axo.set_xticks([]); axo.set_yticks([])
            for s in axo.spines.values():
                s.set_edgecolor("#94a3b8"); s.set_linewidth(1.2)
            axo.set_title(f"({'cde'[k]}) {label}", loc="left", fontsize=8.5)
            con = ConnectionPatch(xyA=(pos[1], pos[0]), coordsA=axm.transData,
                                  xyB=(img.shape[1] / 2, 0), coordsB=axo.transData,
                                  color="#94a3b8", lw=1.0, ls=(0, (4, 3)),
                                  arrowstyle="-|>", mutation_scale=9)
            fig.add_artist(con)
            axm.plot(pos[1], pos[0], "o", color="#94a3b8", mec="black", ms=5, zorder=11)

        fig.suptitle("Anatomy of an episode", y=.995, fontsize=12)
        fig.text(.5, .945, "what the agent receives — a 21×21 egocentric crop "
                 "(Crafter tiles) plus heading and elapsed time; black = out of bounds",
                 ha="center", fontsize=8.5, color="#6d7a70")
        fig.savefig(out / "fig_task_anatomy.png", bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default=str(REPO / "data/bridge_tunnel/forkwall6k/train.pkl"))
    p.add_argument("--out", default=str(REPO / "paper/figures/forkwall_paper"))
    p.add_argument("--ppo-ckpt", default=str(REPO / "final_models/ppo/ppo_plain.pt"))
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    _, by_cat = load_by_cat(a.maps)
    fig_categories(by_cat, out)
    fig_anatomy(by_cat, out, a.ppo_ckpt)
    print("wrote task figures ->", out)


if __name__ == "__main__":
    main()
