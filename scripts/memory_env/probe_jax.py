#!/usr/bin/env python
"""Linear-probe a frozen JAX Dreamer MemoryEnv checkpoint for the cue's SHAPE and
COLOUR, per timestep, to see what the RSSM latent actually carries.

Motivation: across every training variant the model solves the SHAPE subgoal
(branch) but never the COLOUR subgoal (door) — it walks to a fixed door side.
The decisive question for both "solve it" and "steer it" is whether colour is
*represented but unused* (a policy problem, and directly steerable) or simply
*not carried* by the RSSM (then an auxiliary belief head is the honest fix).

We roll the greedy policy on mixed cues + random doors, collect the actor-input
features (post.features() = deter+stoch) at every step, and fit a linear probe
(dual ridge, held-out split) to decode SHAPE (up/down) and COLOUR (green/blue)
from the latent at each timestep. Plotting accuracy vs. timestep against the
cue-visibility window shows whether each attribute *persists* after the cue
leaves view. SHAPE is the positive control (we know it's used).
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

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

from cogniland.memory_env.jax import (  # noqa: E402
    reset as jreset, step as jstep, build_obs, EnvParams, constants as C,
)
from purejaxwm.dreamerv3.world_model import RSSM, MLPHead  # noqa: E402
import dreamerv3_memory as M  # noqa: E402
import diag_jax as D  # noqa: E402  (reuse _flat / _build_model)

_CUE_TILES = jnp.asarray([C.CUE_GREEN_UP, C.CUE_BLUE_UP, C.CUE_GREEN_DOWN, C.CUE_BLUE_DOWN])
_IS_BLUE = np.asarray(C.CUE_IS_BLUE)   # colour label per cue_type (0=green,1=blue)
_IS_DOWN = np.asarray(C.CUE_IS_DOWN)   # shape  label per cue_type (0=up,1=down)


def _params(cfg, door_random_prob):
    return EnvParams.from_config(
        cue_distribution="factorized",
        max_steps=cfg["max_steps"], view_size=cfg["view_size"],
        center_wall_thickness=cfg["center_wall_thickness"], pre_cue_steps=cfg["pre_cue_steps"],
        pre_branch_corridor_len=cfg["pre_branch_corridor_len"], branch_len=cfg["branch_len"],
        post_branch_corridor_len=cfg["post_branch_corridor_len"], step_penalty=cfg["step_penalty"],
        branch_bonus=cfg["branch_bonus"], success_reward=cfg["success_reward"],
        wrong_door_reward=cfg["wrong_door_reward"], shaping_coef=cfg["shaping_coef"],
        door_random_prob=door_random_prob)


def rollout(cfg, wm, ac, enc, rssm, actor, n, key, door_random_prob):
    """Return feats (T,n,Df), cue_visible (T,n), done_acc (T,n), cue_type (n,)."""
    p = _params(cfg, door_random_prob)
    A, ms = C.NUM_ACTIONS, cfg["max_steps"]
    enc_p, rssm_p, act_p = wm["encoder"], wm["rssm"], ac["actor"]
    keys = jax.random.split(key, n)
    state0 = jax.vmap(lambda k: jreset(k, p))(keys)
    obs0 = D._flat(jax.vmap(lambda s: build_obs(s, p))(state0))
    carry = (state0, obs0, rssm.initial_state((n,)), jnp.zeros((n, A)),
             jnp.ones((n,), bool), jnp.zeros((n,), bool), key)

    def body(carry, _):
        state, obs, rss, la, isf, dacc, key = carry
        embed = enc.apply(enc_p, obs)
        key, sk = jax.random.split(key)
        _, post = rssm.apply(rssm_p, rss, la, embed, isf, rngs={"stoch": sk})
        feats = post.features()
        a = jnp.argmax(actor.apply(act_p, feats), axis=-1)
        key, sk2 = jax.random.split(key)
        sks = jax.random.split(sk2, n)
        ns, r, d, info = jax.vmap(lambda k, s, ai: jstep(k, s, ai, p))(sks, state, a)
        nobs = D._flat(jax.vmap(lambda s: build_obs(s, p))(ns))
        mm = jax.vmap(lambda s: build_obs(s, p)["minimap"])(state)          # (n,V,V)
        cue_vis = jnp.any((mm[..., None] == _CUE_TILES).any(-1), axis=(1, 2)).astype(jnp.float32)
        new = (ns, nobs, post, jax.nn.one_hot(a, A), jnp.zeros((n,), bool), dacc | d, key)
        return new, (feats, cue_vis, dacc.astype(jnp.float32))

    _, (feats, cue_vis, dacc) = jax.lax.scan(body, carry, None, length=ms)
    return (np.asarray(feats), np.asarray(cue_vis), np.asarray(dacc),
            np.asarray(state0.cue_type))


def dual_ridge_acc(X, y, rng, train_frac=0.7, lam=10.0):
    """Held-out accuracy of a linear probe (dual ridge on standardized feats)."""
    n = X.shape[0]
    if n < 40 or y.min() == y.max():
        return np.nan
    perm = rng.permutation(n)
    X, y = X[perm], y[perm].astype(np.float64)
    ntr = int(n * train_frac)
    Xtr, Xte, ytr, yte = X[:ntr], X[ntr:], y[:ntr], y[ntr:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    yb = ytr.mean()
    K = Xtr @ Xtr.T
    alpha = np.linalg.solve(K + lam * np.eye(ntr), ytr - yb)
    pred = Xte @ (Xtr.T @ alpha) + yb
    return float(np.mean((pred > 0.5) == (yte > 0.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=640)
    ap.add_argument("--door-random-prob", type=float, default=1.0)
    ap.add_argument("--tmax", type=int, default=26)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rd = pathlib.Path(a.run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    import orbax.checkpoint as ocp
    ckpt = sorted((rd / "checkpoints").glob("step_*"))[-1]
    payload = ocp.PyTreeCheckpointer().restore(str(ckpt.resolve()))
    wm, ac = payload["wm_params"], payload["ac_params"]
    enc, rssm, actor = D._build_model(cfg)

    feats, cue_vis, dacc, cue_type = rollout(
        cfg, wm, ac, enc, rssm, actor, a.n, jax.random.PRNGKey(0), a.door_random_prob)
    T = min(a.tmax, feats.shape[0])
    color = _IS_BLUE[cue_type]
    shape = _IS_DOWN[cue_type]
    rng = np.random.default_rng(0)

    print(f"== {cfg['cue']} model  ({ckpt.name})  door_random_prob={a.door_random_prob}", flush=True)
    print("   t  live  cue_vis  shape_acc  colour_acc", flush=True)
    ts, sh_acc, co_acc, vis = [], [], [], []
    for t in range(T):
        live = dacc[t] < 0.5
        nlive = int(live.sum())
        X = feats[t][live]
        sa = dual_ridge_acc(X, shape[live], rng)
        ca = dual_ridge_acc(X, color[live], rng)
        v = float(cue_vis[t].mean())
        ts.append(t); sh_acc.append(sa); co_acc.append(ca); vis.append(v)
        print(f"  {t:2d}  {nlive:4d}   {v:4.2f}     {sa:5.2f}      {ca:5.2f}", flush=True)

    out = a.out or str(rd / "probe_shape_colour.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(ts, 0, vis, color="0.85", step="mid", label="cue in view (frac)")
    ax.plot(ts, sh_acc, "-o", color="#1f77b4", label="shape probe acc")
    ax.plot(ts, co_acc, "-o", color="#d62728", label="colour probe acc")
    ax.axhline(0.5, ls="--", color="k", lw=0.8, label="chance")
    ax.set_xlabel("timestep"); ax.set_ylabel("held-out linear-probe accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f"{cfg['cue']} model — what the RSSM latent carries ({ckpt.name})")
    ax.legend(loc="lower left", fontsize=8); fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[probe] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
