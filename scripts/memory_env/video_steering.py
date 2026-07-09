#!/usr/bin/env python
"""Steering videos with live belief overlay (same 4-panel format as video_rollout).

Episodes (stitched into one MP4), each pair = same episode with/without the
shape-belief intervention (window: post cue-room -> branch decision):
  1. 2cue green_up BASELINE          -> correct door
  2. 2cue green_up STEERED (add a=4) -> same branch, WRONG door (entangled)
  3. 4cue green_down BASELINE        -> correct door
  4. 4cue green_down CLAMPED         -> branch FLIPS, door still correct (separable)

Steering is visible in every panel: purple diamonds on the maze trail, a purple
band in the belief timeline, and the belief-plane dot turning purple while it is
being dragged.
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
DIR_ARROW = {0: (0.35, 0), 1: (0, 0.35), 2: (-0.35, 0), 3: (0, -0.35)}
PURPLE = "#8e44ad"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white",
})


def load_model(run_dir):
    rd = pathlib.Path(run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    params = ocp.PyTreeCheckpointer().restore(
        str(sorted((rd / "checkpoints").glob("step_*"))[-1].resolve()))["params"]
    net = P.ActorCriticRNN(action_dim=C.NUM_ACTIONS, view_size=cfg["view_size"],
                           token_dim=cfg["token_dim"], embed_hidden=cfg["embed_hidden"],
                           gru_hidden=cfg["gru_hidden"])
    dset = np.load(rd / "activations.npz", allow_pickle=True)
    X = dset["feat"].astype(np.float64)
    clf_s = LogisticRegression(max_iter=3000).fit(X, dset["shape"])
    clf_c = LogisticRegression(max_iter=3000).fit(X, dset["colour"])
    w_s = clf_s.coef_[0] / np.linalg.norm(clf_s.coef_[0])
    w_c = clf_c.coef_[0] / np.linalg.norm(clf_c.coef_[0])
    m = dset["phase"] >= 2
    proj_s = X[m] @ w_s
    land = {c: (float((X[m][dset["cue_type"][m] == c] @ w_s).mean()),
                float((X[m][dset["cue_type"][m] == c] @ w_c).mean())) for c in range(4)}
    allp = np.stack([X @ w_s, X @ w_c], 1)
    lims = ((np.percentile(allp[:, 0], 1) - 0.6, np.percentile(allp[:, 0], 99) + 0.6),
            (np.percentile(allp[:, 1], 1) - 0.6, np.percentile(allp[:, 1], 99) + 0.6))
    tgt_up = float(proj_s[dset["shape"][m] == 0].mean())
    tgt_down = float(proj_s[dset["shape"][m] == 1].mean())
    return cfg, net, params, dict(clf_s=clf_s, clf_c=clf_c, w_s=w_s, w_c=w_c,
                                  land=land, lims=lims, tgt_up=tgt_up, tgt_down=tgt_down)


def steered_rollout(cfg, net, params, cue, key, T, force_door, w, alpha, sign, mode, tgt):
    p = D._env_params(cfg, cue)
    wj = jnp.asarray(w, dtype=jnp.float32)
    state0 = jreset(key, p).replace(door_green_top=jnp.bool_(force_door))
    obs0 = D._flat({k: v[None] for k, v in build_obs(state0, p).items()})
    hidden0 = P.ScannedRNN.initialize_carry(1, cfg["gru_hidden"])

    def body(carry, _):
        state, obs, hidden, last_done, dacc, key = carry
        window = ((state.agent_x > p.x_room_end)
                  & (state.taken_branch == C.BRANCH_NONE) & (~dacc)) & (alpha != 0.0)
        if mode == "clamp":
            delta = (tgt - hidden @ wj)[:, None] * wj[None, :]
        else:
            delta = (sign * alpha) * wj[None, :]
        hidden = hidden + jnp.where(window, 1.0, 0.0) * delta
        new_hidden, logits, _ = net.apply(params, hidden,
                                          (obs[None], jnp.asarray(last_done)[None, None]))
        a = jnp.argmax(logits[0], axis=-1)
        key, sk = jax.random.split(key)
        ns, r, dn, info = jstep(sk, state, a[0], p)
        nobs = D._flat({k: v[None] for k, v in build_obs(ns, p).items()})
        mm = build_obs(state, p)["minimap"]
        out = (state.agent_x, state.agent_y, state.agent_dir, mm, new_hidden[0],
               window, dn, info["reached_target"])
        return (ns, nobs, new_hidden, dn, dacc | dn, key), out

    carry = (state0, obs0, hidden0, jnp.bool_(False), jnp.bool_(False), key)
    _, outs = jax.lax.scan(body, carry, None, length=T)
    return p, state0, tuple(np.asarray(v) for v in outs)


def draw_maze(ax, p, s0, xs, ys, dirs, win, t):
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
    ax.plot(xs[:t + 1], ys[:t + 1], "-", color="#f28e2b", lw=2, alpha=0.7, zorder=4)
    wmask = win[:t + 1].astype(bool)
    if wmask.any():
        ax.scatter(np.asarray(xs[:t + 1])[wmask], np.asarray(ys[:t + 1])[wmask],
                   s=42, c=PURPLE, marker="D", edgecolor="k", lw=0.3, zorder=5,
                   label="steered")
    ax.scatter([xs[t]], [ys[t]], s=110,
               c=(PURPLE if win[t] else "#d1495b"), edgecolor="k", zorder=6)
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


def draw_plane(ax, land, lims, ps, pc, win, t):
    for c in range(4):
        ax.scatter(*land[c], s=260, c=CUE_COL[c], alpha=0.25, edgecolor=CUE_COL[c], zorder=2)
        ax.annotate(CUE_NAMES[c], land[c], fontsize=7.5, ha="center", va="center",
                    color="#333", zorder=3)
    ax.plot(ps[:t + 1], pc[:t + 1], "-", color="#d1495b", lw=1.6, alpha=0.6, zorder=4)
    wmask = win[:t + 1].astype(bool)
    if wmask.any():
        ax.scatter(ps[:t + 1][wmask], pc[:t + 1][wmask], s=30, c=PURPLE, marker="D",
                   zorder=5, alpha=0.8)
    ax.scatter([ps[t]], [pc[t]], s=100, c=(PURPLE if win[t] else "#d1495b"),
               edgecolor="k", zorder=6)
    if win[t]:
        ax.text(0.03, 0.95, "STEERING ON", transform=ax.transAxes, fontsize=9,
                color=PURPLE, fontweight="bold", va="top")
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
    ax.set_xlabel("shape axis  (h . w_shape)", fontsize=9)
    ax.set_ylabel("colour axis  (h . w_colour)", fontsize=9)
    ax.set_title("belief plane (GRU hidden)", fontsize=11, fontweight="bold")


def draw_timeline(ax, pd, pb, vis, win, t, T, true_shape, true_col):
    xs = np.arange(len(pd))
    ax.fill_between(xs, 0, vis, color="0.9", step="mid", label="cue in view")
    if win.any():
        w0, w1 = np.where(win)[0][[0, -1]]
        ax.axvspan(w0 - 0.5, w1 + 0.5, color=PURPLE, alpha=0.13, label="intervention")
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


def render_episode(writer, tagline, cfg, net, params, aux, cue, key, force_door,
                   alpha, sign, mode, tgt, hold, fps_meta=""):
    p, s0, (xs, ys, ds, mms, hs, win, dn, reach) = steered_rollout(
        cfg, net, params, cue, key, 60, force_door, aux["w_s"], alpha, sign, mode, tgt)
    nd = int(np.argmax(dn)) + 1 if dn.any() else len(xs)
    ok = bool(reach[nd - 1]) if dn.any() else False
    clf_s, clf_c = aux["clf_s"], aux["clf_c"]
    pd = 1 / (1 + np.exp(-(hs @ clf_s.coef_[0] + clf_s.intercept_[0])))
    pb = 1 / (1 + np.exp(-(hs @ clf_c.coef_[0] + clf_c.intercept_[0])))
    ps, pc = hs @ aux["w_s"], hs @ aux["w_c"]
    vis = np.array([(mm[..., None] == _CUE_TILES).any() for mm in mms], dtype=float)
    true_shape = float("down" in cue); true_col = float("blue" in cue)
    win = win.astype(bool)
    for t in list(range(nd)) + [nd - 1] * hold:
        fig = plt.figure(figsize=(13.2, 7.2))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.55, 1.0],
                              height_ratios=[1, 1.12], hspace=0.3, wspace=0.25)
        draw_maze(fig.add_subplot(gs[0, :2]), p, s0, xs, ys, ds, win, t)
        draw_agent_view(fig.add_subplot(gs[0, 2]), mms[t], ds, t)
        draw_plane(fig.add_subplot(gs[1, 2]), aux["land"], aux["lims"], ps, pc, win, t)
        draw_timeline(fig.add_subplot(gs[1, :2]), pd, pb, vis, win[:nd], t, len(pd),
                      true_shape, true_col)
        outcome = ""
        if t == nd - 1 and dn[nd - 1]:
            outcome = "   ->   " + ("colour-correct door OK" if ok else "WRONG door X")
        col = "#1a7d36" if (outcome and ok) else ("#b02418" if outcome else
                                                  (PURPLE if win[t] else "#222"))
        steer_flag = "  ·  STEERING ACTIVE" if win[t] else ""
        fig.suptitle(f"{tagline}  ·  step {t}{steer_flag}{outcome}",
                     fontsize=13, fontweight="bold", color=col)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    print(f"[video] {tagline}: {nd} steps success={ok}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/steering_beliefs.mp4")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--dir2", default="outputs/ppo_runs/ppo_2cue_vs2")
    ap.add_argument("--dir4", default="outputs/ppo_runs/ppo_4cue_vs4")
    a = ap.parse_args()

    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8,
                                macro_block_size=1)
    # ---- 2cue: entangled ----
    cfg, net, params, aux = load_model(a.dir2)
    key = jax.random.PRNGKey(11)
    render_episode(writer, "2cue · green_up · BASELINE", cfg, net, params, aux,
                   "green_up", key, True, 0.0, 1.0, "add", 0.0, a.hold)
    render_episode(writer, "2cue · green_up · STEERED shape axis (entangled)", cfg, net,
                   params, aux, "green_up", key, True, 4.0, 1.0, "add", 0.0, a.hold)
    # ---- 4cue: separable ----
    cfg, net, params, aux = load_model(a.dir4)
    key = jax.random.PRNGKey(7)
    render_episode(writer, "4cue · green_down · BASELINE", cfg, net, params, aux,
                   "green_down", key, True, 0.0, -1.0, "add", aux["tgt_up"], a.hold)
    render_episode(writer, "4cue · green_down · shape-CLAMPED (separable)", cfg, net,
                   params, aux, "green_down", key, True, 1.0, -1.0, "clamp",
                   aux["tgt_up"], a.hold)
    writer.close()
    print(f"[video] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
