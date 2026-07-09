#!/usr/bin/env python
"""Qualitative steering figure: baseline vs steered trajectories on the maze.

Same episode (same key, same door layout) run twice — with and without the
shape-belief intervention (window: post-cue-room until the branch decision).
Row 1: 2cue green_up + additive steer  -> branch UNCHANGED but door flips to the
        wrong colour (entangled belief -> task failure).
Row 2: 4cue green_down + clamp steer   -> branch FLIPS but door stays colour-
        correct (separable beliefs -> task still solved).
The steered segment of the trajectory is highlighted.
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

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

from cogniland.memory_env.jax import (  # noqa: E402
    reset as jreset, step as jstep, build_obs, constants as C,
)
import diag_jax as D  # noqa: E402
import train_ppo_memory as P  # noqa: E402
from steer_ppo import fit_directions  # noqa: E402
from viz_rollout_dream import TILE_RGB, CUE_MARK  # noqa: E402


def load(run_dir):
    rd = pathlib.Path(run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    ckpt = sorted((rd / "checkpoints").glob("step_*"))[-1]
    params = ocp.PyTreeCheckpointer().restore(str(ckpt.resolve()))["params"]
    net = P.ActorCriticRNN(action_dim=C.NUM_ACTIONS, view_size=cfg["view_size"],
                           token_dim=cfg["token_dim"], embed_hidden=cfg["embed_hidden"],
                           gru_hidden=cfg["gru_hidden"])
    dirs = fit_directions(rd / "activations.npz")
    return cfg, net, params, dirs


def steered_traj(cfg, net, params, cue, key, w, alpha, sign, mode, tgt,
                 T=110, force_door=True):
    p = D._env_params(cfg, cue)
    state0 = jreset(key, p).replace(door_green_top=jnp.bool_(force_door))
    obs0 = D._flat({k: v[None] for k, v in build_obs(state0, p).items()})
    hidden0 = P.ScannedRNN.initialize_carry(1, cfg["gru_hidden"])

    def body(carry, _):
        state, obs, hidden, last_done, dacc, key = carry
        window = ((state.agent_x > p.x_room_end)
                  & (state.taken_branch == C.BRANCH_NONE) & (~dacc))
        if mode == "clamp":
            delta = (tgt - hidden @ w)[:, None] * w[None, :]
        else:
            delta = (sign * alpha) * w[None, :]
        hidden = hidden + jnp.where(window & (alpha != 0.0), 1.0, 0.0) * delta
        new_hidden, logits, _ = net.apply(params, hidden, (obs[None], jnp.asarray(last_done)[None, None]))
        a = jnp.argmax(logits[0], axis=-1)
        key, sk = jax.random.split(key)
        ns, r, dn, info = jstep(sk, state, a[0], p)
        nobs = D._flat({k: v[None] for k, v in build_obs(ns, p).items()})
        out = (ns.agent_x, ns.agent_y, window, dn, info["reached_target"])
        return (ns, nobs, new_hidden, dn, dacc | dn, key), out

    carry = (state0, obs0, hidden0, jnp.bool_(False), jnp.bool_(False), key)
    _, (xs, ys, win, dn, reach) = jax.lax.scan(body, carry, None, length=T)
    return p, state0, (np.asarray(xs), np.asarray(ys), np.asarray(win),
                       np.asarray(dn), np.asarray(reach))


def draw(ax, p, s0, xs, ys, win, dn, reach, title, col_ok):
    t = np.asarray(p.base_terrain); H, W = t.shape
    full = t.copy()
    ct = int(np.asarray(s0.cue_type))
    full[int(s0.cue_y), int(s0.cue_x)] = C.CUE_TILE[ct]
    dgt = bool(np.asarray(s0.door_green_top))
    full[p.row_door_top, p.x_doorcol] = C.DOOR_GREEN if dgt else C.DOOR_BLUE
    full[p.row_door_bot, p.x_doorcol] = C.DOOR_BLUE if dgt else C.DOOR_GREEN
    ax.imshow(TILE_RGB[full], interpolation="nearest")
    for r in range(H):
        for c in range(W):
            if full[r, c] in CUE_MARK:
                ax.text(c, r, CUE_MARK[full[r, c]], ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
    nd = int(np.argmax(dn)) + 1 if dn.any() else len(xs)
    xx = np.concatenate([[int(s0.agent_x)], xs[:nd]])
    yy = np.concatenate([[int(s0.agent_y)], ys[:nd]])
    ww = np.concatenate([[False], win[:nd]]).astype(bool)
    ax.plot(xx, yy, "-", color="#888", lw=1.4, alpha=0.8, zorder=4)
    ax.scatter(xx[~ww], yy[~ww], s=16, c="#f28e2b", edgecolor="k", lw=0.2, zorder=5,
               label="normal")
    if ww.any():
        ax.scatter(xx[ww], yy[ww], s=26, c="#b07aa1", marker="D", edgecolor="k",
                   lw=0.3, zorder=6, label="steered")
    ax.set_title(title, fontsize=10, color=("#1a7d36" if col_ok else "#b02418"),
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/steering_qualitative.png")
    a = ap.parse_args()

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 6.4))

    # ---- Row 1: 2cue green_up, additive alpha=4 (entangled) ----
    cfg, net, params, (w_s, w_c, cos_sc, t_up, t_dn) = load("outputs/ppo_runs/ppo_2cue_vs2")
    key = jax.random.PRNGKey(11)
    p, s0, (xs, ys, win, dn, reach) = steered_traj(cfg, net, params, "green_up", key,
                                                   w_s, 0.0, 1.0, "add", 0.0)
    ok = bool(reach[np.argmax(dn)]) if dn.any() else False
    draw(ax[0, 0], p, s0, xs, ys, win * 0, dn, reach,
         "2cue · green_up · BASELINE → correct door ✓", ok)
    p, s0, (xs, ys, win, dn, reach) = steered_traj(cfg, net, params, "green_up", key,
                                                   w_s, 4.0, 1.0, "add", 0.0)
    ok = bool(reach[np.argmax(dn)]) if dn.any() else False
    draw(ax[0, 1], p, s0, xs, ys, win, dn, reach,
         "2cue · shape-steered → SAME branch, WRONG-colour door ✗  (entangled)", ok)

    # ---- Row 2: 4cue green_down, clamp (separable) ----
    cfg, net, params, (w_s, w_c, cos_sc, t_up, t_dn) = load("outputs/ppo_runs/ppo_4cue_vs4")
    key = jax.random.PRNGKey(7)
    p, s0, (xs, ys, win, dn, reach) = steered_traj(cfg, net, params, "green_down", key,
                                                   w_s, 0.0, -1.0, "add", t_up)
    ok = bool(reach[np.argmax(dn)]) if dn.any() else False
    draw(ax[1, 0], p, s0, xs, ys, win * 0, dn, reach,
         "4cue · green_down · BASELINE → correct door ✓", ok)
    p, s0, (xs, ys, win, dn, reach) = steered_traj(cfg, net, params, "green_down", key,
                                                   w_s, 1.0, -1.0, "clamp", t_up)
    ok = bool(reach[np.argmax(dn)]) if dn.any() else False
    draw(ax[1, 1], p, s0, xs, ys, win, dn, reach,
         "4cue · shape-CLAMPED → branch FLIPS, door still colour-correct ✓  (separable)", ok)

    ax[0, 1].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("Steering the SHAPE belief (window: after cue room → branch decision) — same episode, belief intervened",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    outp = pathlib.Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=145)
    print(f"[viz_steering] wrote {outp}")


if __name__ == "__main__":
    main()
