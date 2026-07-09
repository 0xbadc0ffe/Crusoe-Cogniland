#!/usr/bin/env python
"""Video rollouts of a solved PPO+GRU MemoryEnv agent with live belief overlay.

Four synced panels per frame:
  1. God's-eye maze (trajectory trail + agent + facing)
  2. Agent's egocentric 5x5 symbolic view (what the policy sees)
  3. Belief plane: the GRU hidden projected onto the shape-axis x colour-axis,
     moving in real time against the 4 cue-class landmarks
  4. Belief timeline: P(shape=down) & P(colour=blue) from the linear probes,
     with the cue-visibility window shaded

One episode per cue type, stitched into a single MP4 (with outcome hold frames).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import imageio.v2 as imageio

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

from cogniland.memory_env.jax import (  # noqa: E402
    reset as jreset, step as jstep, build_obs, constants as C,
)
import diag_jax as D  # noqa: E402
import train_ppo_memory as P  # noqa: E402
from viz_rollout_dream import TILE_RGB, CUE_MARK  # noqa: E402

CUE_NAMES = ["green_up", "blue_up", "green_down", "blue_down"]
CUE_COL = ["#1b9e77", "#3b6fb6", "#7fd4b8", "#9ec9ec"]
_CUE_TILES = np.asarray([C.CUE_GREEN_UP, C.CUE_BLUE_UP, C.CUE_GREEN_DOWN, C.CUE_BLUE_DOWN])
DIR_ARROW = {0: (0.35, 0), 1: (0, 0.35), 2: (-0.35, 0), 3: (0, -0.35)}  # E,S,W,N

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})


def rollout(cfg, net, params, cue, key, T, force_door):
    p = D._env_params(cfg, cue)
    state0 = jreset(key, p).replace(door_green_top=jnp.bool_(force_door))
    obs0 = D._flat({k: v[None] for k, v in build_obs(state0, p).items()})
    hidden0 = P.ScannedRNN.initialize_carry(1, cfg["gru_hidden"])

    def body(carry, _):
        state, obs, hidden, last_done, dacc, key = carry
        new_hidden, logits, _ = net.apply(params, hidden,
                                          (obs[None], jnp.asarray(last_done)[None, None]))
        a = jnp.argmax(logits[0], axis=-1)
        key, sk = jax.random.split(key)
        ns, r, dn, info = jstep(sk, state, a[0], p)
        nobs = D._flat({k: v[None] for k, v in build_obs(ns, p).items()})
        mm = build_obs(state, p)["minimap"]
        out = (state.agent_x, state.agent_y, state.agent_dir, mm, new_hidden[0],
               dn, info["reached_target"])
        return (ns, nobs, new_hidden, dn, dacc | dn, key), out

    carry = (state0, obs0, hidden0, jnp.bool_(False), jnp.bool_(False), key)
    _, (xs, ys, ds, mms, hs, dn, reach) = jax.lax.scan(body, carry, None, length=T)
    return p, state0, tuple(np.asarray(v) for v in (xs, ys, ds, mms, hs, dn, reach))


def draw_maze(ax, p, s0, xs, ys, dirs, t):
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
    ax.plot(xs[:t + 1], ys[:t + 1], "-", color="#f28e2b", lw=2, alpha=0.75, zorder=4)
    ax.scatter([xs[t]], [ys[t]], s=110, c="#d1495b", edgecolor="k", zorder=6)
    dx, dy = DIR_ARROW[int(dirs[t])]
    ax.annotate("", xy=(xs[t] + dx * 2, ys[t] + dy * 2), xytext=(xs[t], ys[t]),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6), zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("high-level view", fontsize=11, fontweight="bold")


def draw_agent_view(ax, mm, dirs, t):
    V = mm.shape[0]
    ax.imshow(TILE_RGB[mm], interpolation="nearest")
    for r in range(V):
        for c in range(V):
            if mm[r, c] in CUE_MARK:
                ax.text(c, r, CUE_MARK[mm[r, c]], ha="center", va="center",
                        color="white", fontsize=13, fontweight="bold")
    m = V // 2
    ax.scatter([m], [m], s=200, c="#d1495b", edgecolor="k", zorder=6)
    dx, dy = DIR_ARROW[int(dirs[t])]
    ax.annotate("", xy=(m + dx * 1.4, m + dy * 1.4), xytext=(m, m),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2), zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("agent view (5x5 symbolic)", fontsize=11, fontweight="bold")


def draw_plane(ax, land, ps, pc, t, lims):
    for c in range(4):
        ax.scatter(*land[c], s=260, c=CUE_COL[c], alpha=0.25, edgecolor=CUE_COL[c], zorder=2)
        ax.annotate(CUE_NAMES[c], land[c], fontsize=7.5, ha="center", va="center",
                    color="#333", zorder=3)
    ax.plot(ps[:t + 1], pc[:t + 1], "-", color="#d1495b", lw=1.6, alpha=0.7, zorder=4)
    ax.scatter([ps[t]], [pc[t]], s=90, c="#d1495b", edgecolor="k", zorder=6)
    ax.scatter([ps[0]], [pc[0]], s=40, marker="s", c="#999", zorder=5)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
    ax.set_xlabel("shape axis  (h · w_shape)", fontsize=9)
    ax.set_ylabel("colour axis  (h · w_colour)", fontsize=9)
    ax.set_title("belief plane (GRU hidden)", fontsize=11, fontweight="bold")


def draw_timeline(ax, pd, pb, vis, t, T, true_shape, true_col):
    xs = np.arange(len(pd))
    ax.fill_between(xs, 0, vis, color="0.9", step="mid", label="cue in view")
    ax.plot(xs[:t + 1], pd[:t + 1], "-", color="#4e79a7", lw=2, label="shape belief  P(down)")
    ax.plot(xs[:t + 1], pb[:t + 1], "-", color="#e15759", lw=2, label="colour belief  P(blue)")
    ax.axhline(0.5, ls="--", c="#999", lw=0.8)
    ax.axhline(float(true_shape), ls=":", c="#4e79a7", lw=1.0, alpha=0.6)
    ax.axhline(float(true_col), ls=":", c="#e15759", lw=1.0, alpha=0.6)
    ax.axvline(t, color="#d1495b", lw=1.2, alpha=0.8)
    ax.set_xlim(0, T - 1); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("timestep", fontsize=9); ax.set_ylabel("probe belief", fontsize=9)
    ax.set_title("belief over time (dotted = ground truth)", fontsize=11, fontweight="bold")
    ax.legend(loc="center right", fontsize=7.5, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/ppo_runs/ppo_4cue_vs4")
    ap.add_argument("--out", default="outputs/rollout_beliefs.mp4")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--hold", type=int, default=10)
    a = ap.parse_args()
    rd = pathlib.Path(a.run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    params = ocp.PyTreeCheckpointer().restore(
        str(sorted((rd / "checkpoints").glob("step_*"))[-1].resolve()))["params"]
    net = P.ActorCriticRNN(action_dim=C.NUM_ACTIONS, view_size=cfg["view_size"],
                           token_dim=cfg["token_dim"], embed_hidden=cfg["embed_hidden"],
                           gru_hidden=cfg["gru_hidden"])

    # probes + landmarks from the activation dataset
    dset = np.load(rd / "activations.npz", allow_pickle=True)
    X = dset["feat"].astype(np.float64)
    clf_s = LogisticRegression(max_iter=3000).fit(X, dset["shape"])
    clf_c = LogisticRegression(max_iter=3000).fit(X, dset["colour"])
    w_s, w_c = clf_s.coef_[0], clf_c.coef_[0]
    w_s /= np.linalg.norm(w_s); w_c /= np.linalg.norm(w_c)
    m = dset["phase"] >= 2
    land = {c: (float((X[m][dset["cue_type"][m] == c] @ w_s).mean()),
                float((X[m][dset["cue_type"][m] == c] @ w_c).mean())) for c in range(4)}
    allp = np.stack([X @ w_s, X @ w_c], 1)
    lims = ((np.percentile(allp[:, 0], 1) - 0.5, np.percentile(allp[:, 0], 99) + 0.5),
            (np.percentile(allp[:, 1], 1) - 0.5, np.percentile(allp[:, 1], 99) + 0.5))

    episodes = [("green_up", True), ("blue_up", True), ("green_down", False), ("blue_down", False)]
    trained = set(D.TRAIN_CUES[cfg["cue"]])
    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8,
                                macro_block_size=1)
    key = jax.random.PRNGKey(5)
    for cue, fdoor in episodes:
        if cue not in trained:
            continue
        key, k = jax.random.split(key)
        p, s0, (xs, ys, ds, mms, hs, dn, reach) = rollout(cfg, net, params, cue, k, 60, fdoor)
        nd = int(np.argmax(dn)) + 1 if dn.any() else len(xs)
        ok = bool(reach[nd - 1])
        pd = 1 / (1 + np.exp(-(hs @ clf_s.coef_[0] + clf_s.intercept_[0])))   # P(down)
        pb = 1 / (1 + np.exp(-(hs @ clf_c.coef_[0] + clf_c.intercept_[0])))   # P(blue)
        ps, pc = hs @ w_s, hs @ w_c
        vis = np.array([(mm[..., None] == _CUE_TILES).any() for mm in mms], dtype=float)
        true_shape = float("down" in cue); true_col = float("blue" in cue)
        layout = "green-top" if fdoor else "BLUE-top"
        for t in list(range(nd)) + [nd - 1] * a.hold:
            fig = plt.figure(figsize=(13.2, 7.2))
            gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.55, 1.0],
                                  height_ratios=[1, 1.12], hspace=0.3, wspace=0.25)
            ax_maze = fig.add_subplot(gs[0, :2])
            ax_view = fig.add_subplot(gs[0, 2])
            ax_plane = fig.add_subplot(gs[1, 2])
            ax_time = fig.add_subplot(gs[1, :2])
            draw_maze(ax_maze, p, s0, xs[:, ], ys, ds, t)
            draw_agent_view(ax_view, mms[t], ds, t)
            draw_plane(ax_plane, land, ps, pc, t, lims)
            draw_timeline(ax_time, pd, pb, vis, t, len(pd), true_shape, true_col)
            outcome = ""
            if t == nd - 1 and dn[nd - 1]:
                outcome = "   →   " + ("REACHED colour-correct door ✓" if ok else "wrong door ✗")
            fig.suptitle(f"{cfg['cue']} PPO+GRU  ·  cue = {cue}  ·  doors {layout}  ·  step {t}{outcome}",
                         fontsize=13, fontweight="bold",
                         color=("#1a7d36" if (outcome and ok) else ("#b02418" if outcome else "#222")))
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            writer.append_data(frame)
            plt.close(fig)
        print(f"[video] {cue} ({layout}): {nd} steps, success={ok}", flush=True)
    writer.close()
    print(f"[video] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
