#!/usr/bin/env python
"""Steer the SHAPE belief of a PPO+GRU MemoryEnv agent and test belief entanglement.

Intervention: during the window AFTER the cue room and UNTIL the up/down branch
decision is made (agent_x >= x_pre_start AND taken_branch == NONE), add
alpha * (+/-)w_shape to the GRU hidden each step, where w_shape is the linear
shape-probe direction (fit on the activation dataset) and the sign pushes the
belief toward the OPPOSITE of the true cue shape. The intervention ends at the
branch decision; the door choice happens later, unsteered.

Readout per episode: branch taken (flipped?), door colour chosen, success.
Entanglement test: if shape and colour beliefs are ENTANGLED (e.g. the 2cue
model, whose training cues confound shape with colour), flipping shape drags
colour along -> the agent later picks the WRONG-colour door -> task failure.
If DISENTANGLED (4cue control), the branch flips but the door stays correct.
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
from sklearn.linear_model import LogisticRegression

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

from cogniland.memory_env.jax import (  # noqa: E402
    reset as jreset, step as jstep, build_obs, constants as C,
)
import diag_jax as D  # noqa: E402
import train_ppo_memory as P  # noqa: E402

ALL_CUES = ["green_up", "blue_up", "green_down", "blue_down"]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def fit_directions(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    X = d["feat"].astype(np.float64)
    w_shape = unit(LogisticRegression(max_iter=3000, C=1.0).fit(X, d["shape"]).coef_[0])
    w_colour = unit(LogisticRegression(max_iter=3000, C=1.0).fit(X, d["colour"]).coef_[0])
    # class-typical projections on the shape axis, post-cue (belief formed)
    m = d["phase"] >= 2
    proj = X[m] @ w_shape
    tgt_up = float(proj[d["shape"][m] == 0].mean())     # typical value when cue = up
    tgt_down = float(proj[d["shape"][m] == 1].mean())   # typical value when cue = down
    return (jnp.asarray(w_shape, dtype=jnp.float32),
            jnp.asarray(w_colour, dtype=jnp.float32),
            float(abs(w_shape @ w_colour)), tgt_up, tgt_down)


def steered_rollout(cfg, net, params, cue, n, key, w, alpha, sign, mode="add", tgt=0.0):
    """Greedy rollout with an intervention on the GRU carry inside the window.

    mode="add":   h += sign*alpha*w  (activation addition; alpha sets magnitude)
    mode="clamp": h += (tgt - h@w)*w (replace the shape-axis coordinate with the
                  OPPOSITE class's typical value; norm-controlled, no alpha)
    """
    p = D._env_params(cfg, cue)
    ms = cfg["max_steps"]
    keys = jax.random.split(key, n)
    state = jax.vmap(lambda k: jreset(k, p))(keys)
    obs = D._flat(jax.vmap(lambda s: build_obs(s, p))(state))
    hidden = P.ScannedRNN.initialize_carry(n, cfg["gru_hidden"])
    carry = (state, obs, hidden, jnp.zeros((n,), bool), jnp.zeros((n,), bool),
             jnp.zeros((n,)), jnp.zeros((n,), jnp.int32), jnp.zeros((n,), jnp.int32), key)

    def body(carry, _):
        state, obs, hidden, last_done, dacc, succ, tb, sd, key = carry
        # window: after the cue room, until the up/down decision is made
        window = ((state.agent_x > p.x_room_end)
                  & (state.taken_branch == C.BRANCH_NONE) & (~dacc))
        if mode == "clamp":
            delta = (tgt - hidden @ w)[:, None] * w[None, :]
        else:
            delta = (sign * alpha) * w[None, :]
        hidden = hidden + window[:, None] * delta
        new_hidden, logits, _ = net.apply(params, hidden, (obs[None], last_done[None]))
        a = jnp.argmax(logits[0], axis=-1)
        key, sk = jax.random.split(key)
        sks = jax.random.split(sk, n)
        ns, r, dn, info = jax.vmap(lambda k, s, ai: jstep(k, s, ai, p))(sks, state, a)
        nobs = D._flat(jax.vmap(lambda s: build_obs(s, p))(ns))
        newly = dn & (~dacc)
        succ = jnp.where(newly, info["reached_target"].astype(jnp.float32), succ)
        tb = jnp.where(newly, ns.taken_branch, tb)
        sd = jnp.where(newly, ns.selected_door, sd)
        return (ns, nobs, new_hidden, dn, dacc | dn, succ, tb, sd, key), None

    (state, obs, hidden, ld, dacc, succ, tb, sd, key), _ = jax.lax.scan(
        body, carry, None, length=ms)
    return np.asarray(succ), np.asarray(tb), np.asarray(sd), np.asarray(dacc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--mode", choices=["add", "clamp"], default="add")
    a = ap.parse_args()
    rd = pathlib.Path(a.run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    ckpt = sorted((rd / "checkpoints").glob("step_*"))[-1]
    params = ocp.PyTreeCheckpointer().restore(str(ckpt.resolve()))["params"]
    net = P.ActorCriticRNN(action_dim=C.NUM_ACTIONS, view_size=cfg["view_size"],
                           token_dim=cfg["token_dim"], embed_hidden=cfg["embed_hidden"],
                           gru_hidden=cfg["gru_hidden"])
    w_shape, w_colour, cos_sc, tgt_up, tgt_down = fit_directions(rd / "activations.npz")
    trained = D.TRAIN_CUES[cfg["cue"]]
    print(f"== STEERING {cfg['cue']} model ({ckpt.name})  mode={a.mode}", flush=True)
    print(f"   |cos(w_shape, w_colour)| = {cos_sc:.3f}   "
          f"({'ENTANGLED' if cos_sc > 0.5 else 'disentangled'} belief axes)", flush=True)
    print(f"   shape-axis class targets: up={tgt_up:.2f} down={tgt_down:.2f}", flush=True)
    print(f"   window: x > {D._env_params(cfg, trained[0]).x_room_end} (post cue-room) "
          f"until branch decision; steering pushes shape belief to the OPPOSITE side", flush=True)

    key = jax.random.PRNGKey(0)
    hdr = (f"   {'cue':11s} {'alpha':>5s} | {'branch: kept':>12s} {'FLIPPED':>8s} | "
           f"{'door: colour-OK':>15s} {'wrong-col':>9s} | {'success':>7s}")
    for cue in trained:
        print(hdr, flush=True)
        is_down = D.IS_DOWN[cue]
        sign = -1.0 if is_down else 1.0        # push toward the opposite shape
        correct_b = C.BRANCH_DOWN if is_down else C.BRANCH_UP
        flipped_b = C.BRANCH_UP if is_down else C.BRANCH_DOWN
        target_sd = C.SEL_BLUE if "blue" in cue else C.SEL_GREEN
        wrong_sd = C.SEL_GREEN if "blue" in cue else C.SEL_BLUE
        tgt = tgt_up if is_down else tgt_down          # clamp to the OPPOSITE class value
        alphas = [0.0, 1.0] if a.mode == "clamp" else a.alphas   # clamp: 0=off, 1=on
        for alpha in alphas:
            key, k = jax.random.split(key)
            mode = "add" if (a.mode == "clamp" and alpha == 0.0) else a.mode
            eff_alpha = 0.0 if (a.mode == "clamp" and alpha == 0.0) else alpha
            succ, tb, sd, dacc = steered_rollout(cfg, net, params, cue, a.n, k,
                                                 w_shape, eff_alpha, sign, mode=mode, tgt=tgt)
            kept = float((tb == correct_b).mean()); flip = float((tb == flipped_b).mean())
            dok = float((sd == target_sd).mean()); dwr = float((sd == wrong_sd).mean())
            lab = ("off" if alpha == 0.0 else "ON ") if a.mode == "clamp" else f"{alpha:.1f}"
            print(f"   {cue:11s} {lab:>5s} | {kept:12.2f} {flip:8.2f} | "
                  f"{dok:15.2f} {dwr:9.2f} | {float(succ.mean()):7.2f}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
