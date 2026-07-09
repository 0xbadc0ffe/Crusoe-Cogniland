#!/usr/bin/env python
"""Per-cue success + branch_correct for a PPO+GRU MemoryEnv checkpoint (greedy).

Mirror of diag_jax.py but for the recurrent PPO policy (train_ppo_memory.py): the
GRU carries the cue memory, so we run it step-by-step per fixed cue and report,
per cue: success (reached the colour-correct door), branch_correct (shape-correct
branch), reached_end. Doors are randomised (door_random_prob from the run config).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

from cogniland.memory_env.jax import reset as jreset, step as jstep, build_obs, constants as C  # noqa: E402
import diag_jax as D  # noqa: E402  (reuse _flat / _env_params / TRAIN_CUES)
import train_ppo_memory as P  # noqa: E402

ALL_CUES = D.ALL_CUES
IS_DOWN = D.IS_DOWN


def eval_cue(cue, cfg, net, params, n, key):
    p = D._env_params(cfg, cue)
    A, ms = C.NUM_ACTIONS, cfg["max_steps"]
    keys = jax.random.split(key, n)
    state = jax.vmap(lambda k: jreset(k, p))(keys)
    obs = D._flat(jax.vmap(lambda s: build_obs(s, p))(state))
    hidden = P.ScannedRNN.initialize_carry(n, cfg["gru_hidden"])
    carry = (state, obs, hidden, jnp.zeros((n,), bool), jnp.zeros((n,), bool),
             jnp.zeros((n,)), jnp.zeros((n,), jnp.int32), key)

    def body(carry, _):
        state, obs, hidden, last_done, dacc, succ, tb, key = carry
        hidden, logits, _ = net.apply(params, hidden, (obs[None], last_done[None]))
        a = jnp.argmax(logits[0], axis=-1)
        key, sk = jax.random.split(key)
        sks = jax.random.split(sk, n)
        ns, r, d, info = jax.vmap(lambda k, s, ai: jstep(k, s, ai, p))(sks, state, a)
        nobs = D._flat(jax.vmap(lambda s: build_obs(s, p))(ns))
        newly = d & (~dacc)
        succ = jnp.where(newly, info["reached_target"].astype(jnp.float32), succ)
        tb = jnp.where(newly, ns.taken_branch, tb)
        return (ns, nobs, hidden, d, dacc | d, succ, tb, key), None

    (state, obs, hidden, ld, dacc, succ, tb, key), _ = jax.lax.scan(body, carry, None, length=ms)
    correct = C.BRANCH_DOWN if IS_DOWN[cue] else C.BRANCH_UP
    return (float(jnp.mean(succ)), float(jnp.mean((tb == correct).astype(jnp.float32))),
            float(jnp.mean(dacc.astype(jnp.float32))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="outputs/ppo_runs/ppo_<cue>_<tag>")
    ap.add_argument("--n", type=int, default=96)
    a = ap.parse_args()
    rd = pathlib.Path(a.run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    ckpt = sorted((rd / "checkpoints").glob("step_*"))[-1]
    payload = ocp.PyTreeCheckpointer().restore(str(ckpt.resolve()))
    params = payload["params"]
    net = P.ActorCriticRNN(action_dim=C.NUM_ACTIONS, view_size=cfg["view_size"],
                           token_dim=cfg["token_dim"], embed_hidden=cfg["embed_hidden"],
                           gru_hidden=cfg["gru_hidden"])
    trained = set(D.TRAIN_CUES[cfg["cue"]])
    print(f"== PPO {cfg['cue']} model  ({ckpt.name})  trained on: {sorted(trained)}", flush=True)
    key = jax.random.PRNGKey(0)
    for cue in ALL_CUES:
        key, k = jax.random.split(key)
        s, b, dd = eval_cue(cue, cfg, net, params, a.n, k)
        tag = "train  " if cue in trained else "heldout"
        print(f"   {cue:11s} [{tag}] success={s:.2f} branch_correct={b:.2f} reached_end={dd:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
