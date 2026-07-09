#!/usr/bin/env python
"""Build an activation dataset from a solved PPO+GRU MemoryEnv model.

Rolls the greedy policy over many episodes (all four cue types incl. held-out
ones, random 50/50 doors) and records, at every live timestep, two activations:
  feat       — the GRU hidden state (the recurrent memory the actor reads)
  obs_embed  — the encoder output before the GRU (current observation, no memory)
plus labels: cue type, cue direction (stored under the legacy key ``shape``),
cue color (key ``colour``), timestep, maze phase, cue visibility, action, agent
position, and episode success. Saved as one .npz per model — the substrate for
the probing, belief-plane and steering analyses.
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

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

from cogniland.memory_env.jax import (  # noqa: E402
    reset as jreset, step as jstep, build_obs, EnvParams, constants as C,
)
import diag_jax as D  # noqa: E402
import train_ppo_memory as P  # noqa: E402

_CUE_TILES = jnp.asarray([C.CUE_GREEN_UP, C.CUE_BLUE_UP, C.CUE_GREEN_DOWN, C.CUE_BLUE_DOWN])
# phase codes: 0 pre-cue, 1 cue-room, 2 pre-branch, 3 branch, 4 post, 5 door
PHASE_NAMES = ["pre-cue", "cue-room", "pre-branch", "branch", "post", "door"]


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


def _phase(ax, p):
    ph = jnp.where(ax < p.x_room_start, 0,
         jnp.where(ax <= p.x_room_end, 1,
         jnp.where(ax < p.x_branch_start, 2,
         jnp.where(ax <= p.x_branch_end, 3,
         jnp.where(ax < p.x_doorcol, 4, 5)))))
    return ph


def collect(cfg, net, params, n, T, key, door_random_prob):
    """Roll n greedy episodes for T steps; return per-step traces (T, n, ...)."""
    p = _params(cfg, door_random_prob)
    keys = jax.random.split(key, n)
    state0 = jax.vmap(lambda k: jreset(k, p))(keys)
    obs0 = D._flat(jax.vmap(lambda s: build_obs(s, p))(state0))
    hidden0 = P.ScannedRNN.initialize_carry(n, cfg["gru_hidden"])

    def body(carry, _):
        state, obs, hidden, last_done, dacc, key = carry
        (new_hidden, logits, _value), aux = net.apply(
            params, hidden, (obs[None], last_done[None]), mutable=["intermediates"])
        feat = new_hidden                                   # GRU hidden (memory)
        obs_embed = aux["intermediates"]["obs_embed"][0][0]  # encoder out, pre-GRU (no memory)
        a = jnp.argmax(logits[0], axis=-1)
        key, sk = jax.random.split(key)
        sks = jax.random.split(sk, n)
        ns, _r, d, info = jax.vmap(lambda k, s, ai: jstep(k, s, ai, p))(sks, state, a)
        nobs = D._flat(jax.vmap(lambda s: build_obs(s, p))(ns))
        mm = jax.vmap(lambda s: build_obs(s, p)["minimap"])(state)
        cue_vis = jnp.any((mm[..., None] == _CUE_TILES).any(-1), axis=(1, 2))
        out = dict(feat=feat, obs_embed=obs_embed, action=a,
                   agent_x=state.agent_x, agent_y=state.agent_y,
                   phase=_phase(state.agent_x, p), cue_vis=cue_vis,
                   reached=info["reached_target"], done=d, dacc=dacc)
        return (ns, nobs, new_hidden, d, dacc | d, key), out

    carry = (state0, obs0, hidden0, jnp.zeros((n,), bool), jnp.zeros((n,), bool), key)
    _, outs = jax.lax.scan(body, carry, None, length=T)
    return p, state0, jax.tree_util.tree_map(np.asarray, outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--tmax", type=int, default=60)
    ap.add_argument("--door-random-prob", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rd = pathlib.Path(a.run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    ckpt = sorted((rd / "checkpoints").glob("step_*"))[-1]
    params = ocp.PyTreeCheckpointer().restore(str(ckpt.resolve()))["params"]
    net = P.ActorCriticRNN(action_dim=C.NUM_ACTIONS, view_size=cfg["view_size"],
                           token_dim=cfg["token_dim"], embed_hidden=cfg["embed_hidden"],
                           gru_hidden=cfg["gru_hidden"])
    _p, state0, outs = collect(cfg, net, params, a.n, a.tmax, jax.random.PRNGKey(0),
                               a.door_random_prob)

    T, n = outs["feat"].shape[:2]
    cue_type = np.asarray(state0.cue_type)                     # (n,)
    # episode success = reached_target on the step the episode first ended
    ep_success = np.zeros(n)
    reached, dacc = outs["reached"], outs["dacc"]              # dacc = done BEFORE step t
    for t in range(T):
        newly = reached[t] & ~dacc[t].astype(bool)
        ep_success = np.where(newly, 1.0, ep_success)

    # flatten to LIVE steps (before episode ended)
    rows = []
    feats = []
    oembs = []
    for t in range(T):
        live = ~outs["dacc"][t].astype(bool)
        idx = np.where(live)[0]
        if idx.size == 0:
            continue
        feats.append(outs["feat"][t][idx])
        oembs.append(outs["obs_embed"][t][idx])
        rows.append(dict(
            ep=idx, t=np.full(idx.size, t),
            cue_type=cue_type[idx],
            shape=np.asarray(C.CUE_IS_DOWN)[cue_type[idx]],
            colour=np.asarray(C.CUE_IS_BLUE)[cue_type[idx]],
            phase=outs["phase"][t][idx],
            cue_vis=outs["cue_vis"][t][idx].astype(np.int8),
            action=outs["action"][t][idx],
            agent_x=outs["agent_x"][t][idx], agent_y=outs["agent_y"][t][idx],
            ep_success=ep_success[idx]))
    feats = np.concatenate(feats, 0).astype(np.float32)
    oembs = np.concatenate(oembs, 0).astype(np.float32)
    cat = {k: np.concatenate([r[k] for r in rows], 0) for k in rows[0]}

    out = a.out or str(rd / "activations.npz")
    np.savez_compressed(out, feat=feats, obs_embed=oembs, cue=cfg["cue"], **cat)
    print(f"[activations] {cfg['cue']} model: {feats.shape[0]} live steps; "
          f"GRU hidden {feats.shape[1]}d + obs_embed {oembs.shape[1]}d "
          f"from {a.n} episodes -> {out}", flush=True)
    print(f"   success(ep-level)={ep_success.mean():.2f}  phases={np.bincount(cat['phase'], minlength=6).tolist()}",
          flush=True)


if __name__ == "__main__":
    main()
