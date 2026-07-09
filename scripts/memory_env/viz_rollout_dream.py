#!/usr/bin/env python
"""Test-time rollouts + world-model DREAMS for a frozen JAX Dreamer MemoryEnv model.

Two artefacts:
  1. God's-eye rollouts: 4 episodes (one per cue type), the maze + cue + doors +
     the agent's actual trajectory + which door it ends on (success/fail).
  2. Dreams: at each observation, roll the RSSM *prior* forward H steps with the
     greedy actor (no obs input) and decode each imagined latent back to a
     minimap — "what the model thinks happens next" — beside the true obs.

Uses the categorical decoder (argmax over per-cell tile logits).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

from cogniland.memory_env.jax import (  # noqa: E402
    reset as jreset, step as jstep, build_obs, EnvParams, constants as C,
)
import dreamerv3_memory as M  # noqa: E402
import diag_jax as D  # noqa: E402

ALL_CUES = ["green_up", "blue_up", "green_down", "blue_down"]
A = C.NUM_ACTIONS

TILE_RGB = np.array([
    [0.95, 0.95, 0.95],  # 0 EMPTY
    [0.23, 0.23, 0.23],  # 1 WALL
    [0.17, 0.63, 0.17],  # 2 CUE_GREEN_UP
    [0.12, 0.47, 0.71],  # 3 CUE_BLUE_UP
    [0.17, 0.63, 0.17],  # 4 CUE_GREEN_DOWN
    [0.12, 0.47, 0.71],  # 5 CUE_BLUE_DOWN
    [0.60, 0.90, 0.60],  # 6 DOOR_GREEN
    [0.60, 0.78, 0.92],  # 7 DOOR_BLUE
    [0.82, 0.82, 0.82],  # 8 OOB
])
CUE_MARK = {2: "^", 3: "^", 4: "v", 5: "v"}


def _params(cfg, cue, door_random_prob):
    return EnvParams.from_config(
        cue_distribution="custom", custom_cues=[cue],
        max_steps=cfg["max_steps"], view_size=cfg["view_size"],
        center_wall_thickness=cfg["center_wall_thickness"], pre_cue_steps=cfg["pre_cue_steps"],
        pre_branch_corridor_len=cfg["pre_branch_corridor_len"], branch_len=cfg["branch_len"],
        post_branch_corridor_len=cfg["post_branch_corridor_len"], step_penalty=cfg["step_penalty"],
        branch_bonus=cfg["branch_bonus"], success_reward=cfg["success_reward"],
        wrong_door_reward=cfg["wrong_door_reward"], shaping_coef=cfg["shaping_coef"],
        door_random_prob=door_random_prob)


def build_decoder(cfg):
    V, K = cfg["view_size"], C.NUM_TILES
    flat_dim = V * V * K + M.SCALAR_DIM
    return M.BridgeTunnelDecoder(hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"],
                                 out_dim=flat_dim)


def decode_mm(flat, V, K):
    return jnp.argmax(flat[..., :V * V * K].reshape(*flat.shape[:-1], V, V, K), axis=-1)


def rollout(cfg, wm, ac, enc, rssm, actor, cue, key, door_random_prob, T, force_door=None):
    """Single-episode greedy rollout; returns true minimaps, xy, posteriors, params, state0.

    force_door: if not None, override the sampled door layout (True=green top) so
    a figure can show BOTH randomised layouts side by side.
    """
    p = _params(cfg, cue, door_random_prob)
    enc_p, rssm_p, act_p = wm["encoder"], wm["rssm"], ac["actor"]
    state0 = jreset(key, p)
    if force_door is not None:
        state0 = state0.replace(door_green_top=jnp.bool_(force_door))
    obs0 = D._flat({k: v[None] for k, v in build_obs(state0, p).items()})  # (1, flat)

    def body(carry, _):
        state, obs, rss, la, isf, key = carry
        embed = enc.apply(enc_p, obs)
        key, sk = jax.random.split(key)
        _, post = rssm.apply(rssm_p, rss, la, embed, isf, rngs={"stoch": sk})
        a = jnp.argmax(actor.apply(act_p, post.features()), axis=-1)   # (1,)
        key, sk2 = jax.random.split(key)
        ns, r, d, info = jstep(sk2, state, a[0], p)
        nobs = D._flat({k: v[None] for k, v in build_obs(ns, p).items()})
        mm = build_obs(state, p)["minimap"]
        return ((ns, nobs, post, jax.nn.one_hot(a, A), jnp.zeros((1,), bool), key),
                (mm, ns.agent_x, ns.agent_y, a[0], d, info["reached_target"], post))

    carry = (state0, obs0, rssm.initial_state((1,)), jnp.zeros((1, A)), jnp.ones((1,), bool), key)
    _, (mms, xs, ys, acts, dones, reached, posts) = jax.lax.scan(body, carry, None, length=T)
    return (p, state0, np.asarray(mms), np.asarray(xs), np.asarray(ys),
            np.asarray(acts), np.asarray(dones), np.asarray(reached), posts)


def imagine(cfg, wm, ac, rssm, actor, decoder, seed_post, H, key):
    """From a (1,)-batch posterior, roll H prior steps w/ greedy actor; decode each."""
    V, K = cfg["view_size"], C.NUM_TILES
    rssm_p, act_p, dec_p = wm["rssm"], ac["actor"], wm["decoder"]

    def body(carry, _):
        st, key = carry
        a = jnp.argmax(actor.apply(act_p, st.features()), axis=-1)
        key, sk = jax.random.split(key)
        prior = rssm.apply(rssm_p, st, jax.nn.one_hot(a, A), None, None,
                           training=False, rngs={"stoch": sk})
        mm = decode_mm(decoder.apply(dec_p, prior.features()), V, K)[0]
        return (prior, key), (mm, a[0])

    _, (mms, acts) = jax.lax.scan(body, (seed_post, key), None, length=H)
    return np.asarray(mms), np.asarray(acts)


# ── drawing ────────────────────────────────────────────────────────────────
def draw_minimap(ax, mm, title=None, agent_center=True):
    mm = np.asarray(mm)
    V = mm.shape[0]
    rgb = TILE_RGB[mm]
    ax.imshow(rgb, interpolation="nearest")
    for r in range(V):
        for c in range(V):
            if mm[r, c] in CUE_MARK:
                ax.text(c, r, CUE_MARK[mm[r, c]], ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
    if agent_center:
        m = V // 2
        ax.add_patch(Rectangle((m - 0.5, m - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def draw_godseye(ax, p, state0, xs, ys, dones, cue, success):
    t = np.asarray(p.base_terrain)
    H, W = t.shape
    full = t.copy()
    ct = int(np.asarray(state0.cue_type))
    full[int(state0.cue_y), int(state0.cue_x)] = C.CUE_TILE[ct]
    dgt = bool(np.asarray(state0.door_green_top))
    full[p.row_door_top, p.x_doorcol] = C.DOOR_GREEN if dgt else C.DOOR_BLUE
    full[p.row_door_bot, p.x_doorcol] = C.DOOR_BLUE if dgt else C.DOOR_GREEN
    ax.imshow(TILE_RGB[full], interpolation="nearest")
    for r in range(H):
        for c in range(W):
            if full[r, c] in CUE_MARK:
                ax.text(c, r, CUE_MARK[full[r, c]], ha="center", va="center",
                        color="white", fontsize=7, fontweight="bold")
    # trajectory up to first done (prepend the true start cell)
    nd = int(np.argmax(dones)) + 1 if dones.any() else len(xs)
    xx = np.concatenate([[int(state0.agent_x)], xs[:nd]])
    yy = np.concatenate([[int(state0.agent_y)], ys[:nd]])
    ax.plot(xx, yy, "-", color="red", lw=1.5, alpha=0.7)
    ax.scatter(xx, yy, c=np.arange(len(xx)), cmap="autumn", s=14, zorder=5, edgecolor="k", lw=0.2)
    ax.scatter([xx[0]], [yy[0]], marker="s", c="red", s=40, zorder=6, label="start")
    res = "REACHED colour door ✓" if success else "wrong door ✗"
    layout = "green-top" if dgt else "BLUE-top"
    ax.set_title(f"{cue}  |  doors {layout}  →  {res}", fontsize=9,
                 color=("green" if success else "firebrick"))
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--door-random-prob", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--tmax", type=int, default=30)
    ap.add_argument("--alt-doors", action="store_true",
                    help="force alternating door layouts across the 4 panels (shows randomisation)")
    ap.add_argument("--conditional", action="store_true",
                    help="2 cues x BOTH door layouts: shows the model goes to the same side regardless of colour")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    rd = pathlib.Path(a.run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    V, K = cfg["view_size"], C.NUM_TILES
    import orbax.checkpoint as ocp
    ckpt = sorted((rd / "checkpoints").glob("step_*"))[-1]
    payload = ocp.PyTreeCheckpointer().restore(str(ckpt.resolve()))
    wm = dict(payload["wm_params"]); ac = payload["ac_params"]
    enc, rssm, actor = D._build_model(cfg)
    decoder = build_decoder(cfg)
    outdir = pathlib.Path(a.outdir or rd)

    if a.conditional:  # same cue under BOTH door layouts
        panels = [("blue_up", True), ("blue_up", False), ("green_down", True), ("green_down", False)]
    else:
        panels = [(c, (i % 2 == 0) if a.alt_doors else None) for i, c in enumerate(ALL_CUES)]

    roll = []
    key = jax.random.PRNGKey(3)
    for cue, fd in panels:
        key, k = jax.random.split(key)
        p, s0, mms, xs, ys, acts, dones, reached, posts = rollout(
            cfg, wm, ac, enc, rssm, actor, cue, k, a.door_random_prob, a.tmax, force_door=fd)
        nd = int(np.argmax(dones)) + 1 if dones.any() else len(xs)
        success = bool(reached[nd - 1])
        roll.append(dict(cue=cue, p=p, s0=s0, mms=mms, xs=xs, ys=ys, dones=dones,
                         posts=posts, success=success, nd=nd))

    # ── Figure 1: 4 god's-eye rollouts ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.2))
    for ax, r in zip(axes.flat, roll):
        draw_godseye(ax, r["p"], r["s0"], r["xs"], r["ys"], r["dones"], r["cue"], r["success"])
    sub = ("SAME cue, BOTH door layouts — the agent goes to the same side regardless of colour"
           if a.conditional else
           "colour bar = time; the branch (shape) is correct but the door side is fixed (colour ignored)")
    fig.suptitle(f"Test-time rollouts — {cfg['cue']} model ({ckpt.name}), door_random_prob={a.door_random_prob}\n{sub}",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    f1 = outdir / ("rollouts_conditional.png" if a.conditional else "rollouts_godseye.png")
    fig.savefig(f1, dpi=130); print("wrote", f1)

    # ── Figure 2: dreams for the first two rollout panels ──
    for r in roll[:2]:
        cue = r["cue"]
        nd = r["nd"]
        steps = list(range(0, min(nd, a.tmax)))
        steps = steps[::max(1, len(steps) // 8)][:8]     # ~8 rows
        Hh = a.horizon
        fig2, ax2 = plt.subplots(len(steps), Hh + 1, figsize=(1.5 * (Hh + 1), 1.5 * len(steps)))
        if len(steps) == 1:
            ax2 = ax2[None, :]
        key, k = jax.random.split(key)
        for i, t in enumerate(steps):
            seed = jax.tree_util.tree_map(lambda x: x[t], r["posts"])   # (1,) posterior at t
            dmm, dact = imagine(cfg, wm, ac, rssm, actor, decoder, seed, Hh, k)
            draw_minimap(ax2[i, 0], r["mms"][t], title=f"t={t}  TRUE obs")
            for h in range(Hh):
                draw_minimap(ax2[i, h + 1], dmm[h], title=f"dream t+{h+1}")
        lay = "gtop" if bool(np.asarray(r["s0"].door_green_top)) else "btop"
        fig2.suptitle(f"{cfg['cue']} model — dreamed next steps at each observation ({cue}, doors {lay})\n"
                      "col 0 = real observation; cols 1..H = RSSM prior imagined & decoded forward",
                      fontsize=10)
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        f2 = outdir / f"dream_{cue}_{lay}.png"; fig2.savefig(f2, dpi=120); print("wrote", f2)


if __name__ == "__main__":
    main()
