#!/usr/bin/env python
"""Per-cue success + branch_correct for a JAX Dreamer MemoryEnv checkpoint (greedy).

Runs the trained policy on the pure-JAX MemoryEnv (one fixed cue at a time) and
reports, per cue: success (reached the colour-correct door) and branch_correct
(took the shape-correct branch, read from EnvState.taken_branch). The model +
env params are reconstructed from the run's saved config.json.
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

from cogniland.memory_env.jax import (  # noqa: E402
    reset as jreset, step as jstep, build_obs, EnvParams, constants as C,
)
from purejaxwm.dreamerv3.world_model import RSSM, MLPHead  # noqa: E402
import dreamerv3_memory as M  # noqa: E402  (SymbolicEmbedEncoder etc.; jax-only imports)

ALL_CUES = ["green_up", "blue_up", "green_down", "blue_down"]
IS_DOWN = dict(zip(ALL_CUES, C.CUE_IS_DOWN))
TRAIN_CUES = {"2cue": ["green_up", "blue_down"],
              "3cue": ["green_up", "green_down", "blue_down"],
              "4cue": ALL_CUES}


def _flat(obs):
    oh = jax.nn.one_hot(obs["minimap"].astype(jnp.int32), C.NUM_TILES)
    oh = oh.reshape(*obs["minimap"].shape[:-2], -1)
    return jnp.concatenate([oh, obs["scalars"].astype(jnp.float32)], axis=-1)


def _env_params(cfg, cue):
    return EnvParams.from_config(
        cue_distribution="custom", custom_cues=[cue],
        max_steps=cfg["max_steps"], view_size=cfg["view_size"],
        center_wall_thickness=cfg["center_wall_thickness"], pre_cue_steps=cfg["pre_cue_steps"],
        pre_branch_corridor_len=cfg["pre_branch_corridor_len"], branch_len=cfg["branch_len"],
        post_branch_corridor_len=cfg["post_branch_corridor_len"], step_penalty=cfg["step_penalty"],
        branch_bonus=cfg["branch_bonus"], success_reward=cfg["success_reward"],
        wrong_door_reward=cfg["wrong_door_reward"], shaping_coef=cfg["shaping_coef"],
        door_random_prob=cfg.get("door_random_prob", 1.0))


def _build_model(cfg):
    V = cfg["view_size"]
    if cfg.get("obs_factored", True):
        enc = M.SymbolicFactoredEncoder(view_size=V, n_tiles=C.NUM_TILES,
                                        token_dim=cfg.get("token_dim", 16),
                                        attr_dim=cfg.get("attr_dim", 8),
                                        hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"],
                                        embed_dim=cfg["wm_hidden"])
    elif cfg.get("obs_embed", True):
        enc = M.SymbolicEmbedEncoder(view_size=V, n_tiles=C.NUM_TILES,
                                     token_dim=cfg.get("token_dim", 16),
                                     hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"],
                                     embed_dim=cfg["wm_hidden"])
    else:
        enc = M.BridgeTunnelEncoder(hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"],
                                    embed_dim=cfg["wm_hidden"])
    rssm = RSSM(deter_dim=cfg["deter"], stoch_size=cfg["stoch"], classes=cfg["classes"],
                hidden=cfg["wm_hidden"], unimix=cfg["unimix"], blocks=cfg["blocks"])
    actor = MLPHead(hidden=cfg["ac_hidden"], num_layers=cfg["ac_layers"],
                    out_dim=C.NUM_ACTIONS, outscale=0.01)
    return enc, rssm, actor


def eval_cue(cue, cfg, wm, ac, enc, rssm, actor, n, key):
    p = _env_params(cfg, cue)
    A, ms = C.NUM_ACTIONS, cfg["max_steps"]
    enc_p, rssm_p, act_p = wm["encoder"], wm["rssm"], ac["actor"]
    keys = jax.random.split(key, n)
    state = jax.vmap(lambda k: jreset(k, p))(keys)
    obs = _flat(jax.vmap(lambda s: build_obs(s, p))(state))
    carry = (state, obs, rssm.initial_state((n,)), jnp.zeros((n, A)), jnp.ones((n,), bool),
             jnp.zeros((n,), bool), jnp.zeros((n,)), jnp.zeros((n,), jnp.int32), key)

    def body(carry, _):
        state, obs, rss, la, isf, dacc, succ, tb, key = carry
        embed = enc.apply(enc_p, obs)
        key, sk = jax.random.split(key)
        _, post = rssm.apply(rssm_p, rss, la, embed, isf, rngs={"stoch": sk})
        logits = actor.apply(act_p, post.features())
        a = jnp.argmax(logits, axis=-1)
        key, sk2 = jax.random.split(key)
        sks = jax.random.split(sk2, n)
        ns, r, d, info = jax.vmap(lambda k, s, ai: jstep(k, s, ai, p))(sks, state, a)
        nobs = _flat(jax.vmap(lambda s: build_obs(s, p))(ns))
        newly = d & (~dacc)
        succ = jnp.where(newly, info["reached_target"].astype(jnp.float32), succ)
        tb = jnp.where(newly, ns.taken_branch, tb)
        return (ns, nobs, post, jax.nn.one_hot(a, A), jnp.zeros((n,), bool),
                dacc | d, succ, tb, key), None

    (state, obs, rss, la, isf, dacc, succ, tb, key), _ = jax.lax.scan(body, carry, None, length=ms)
    correct = C.BRANCH_DOWN if IS_DOWN[cue] else C.BRANCH_UP
    return (float(jnp.mean(succ)), float(jnp.mean((tb == correct).astype(jnp.float32))),
            float(jnp.mean(dacc.astype(jnp.float32))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="outputs/dreamer_runs/memjax_<cue>_<tag>")
    ap.add_argument("--n", type=int, default=96)
    a = ap.parse_args()
    rd = pathlib.Path(a.run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    ckpt = sorted((rd / "checkpoints").glob("step_*"))[-1]
    payload = ocp.PyTreeCheckpointer().restore(str(ckpt.resolve()))
    wm, ac = payload["wm_params"], payload["ac_params"]
    enc, rssm, actor = _build_model(cfg)
    trained = set(TRAIN_CUES[cfg["cue"]])
    print(f"== {cfg['cue']} model  ({ckpt.name})  trained on: {sorted(trained)}", flush=True)
    key = jax.random.PRNGKey(0)
    for cue in ALL_CUES:
        key, k = jax.random.split(key)
        s, b, dd = eval_cue(cue, cfg, wm, ac, enc, rssm, actor, a.n, k)
        tag = "train  " if cue in trained else "heldout"
        print(f"   {cue:11s} [{tag}] success={s:.2f} branch_correct={b:.2f} reached_end={dd:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
