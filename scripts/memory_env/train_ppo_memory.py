#!/usr/bin/env python
"""Recurrent PPO (PPO+GRU) on the pure-JAX symbolic MemoryEnv.

A single-file purejaxrl-style trainer, deliberately on the SAME symbolic obs and
env as `dreamerv3_memory.py`, so PPO and DreamerV3 can be compared head-to-head
on the shape->branch + colour->door memory task. The GRU is the agent's memory:
it must carry the cue (shape AND colour) from the cue room to the fork and door.

Unlike Dreamer, PPO optimises the policy directly on real returns (no imagination
middleman) — the point of the comparison is whether that makes it actually USE
the colour it remembers (which Dreamer's imagination-actor never learned to do).

Checkpoint = network params only (orbax), + config.json, matching the Dreamer runs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cogniland.memory_env.jax import EnvParams, MemoryJaxEnv, constants as C  # noqa: E402
from purejaxwm.commons import (  # noqa: E402
    AutoResetEnvWrapper, BatchEnvWrapper, GymnaxWrapper, LogWrapper,
)

SCALAR_DIM = 5


# ── symbolic obs -> flat one-hot (mirror FlattenObsWrapper categorical) ──────
class FlattenObs(GymnaxWrapper):
    def __init__(self, env, view_size):
        super().__init__(env)
        self.view_size = view_size
        self.flat_dim = view_size * view_size * C.NUM_TILES + SCALAR_DIM

    def _flat(self, obs):
        oh = jax.nn.one_hot(obs["minimap"].astype(jnp.int32), C.NUM_TILES)
        mm = oh.reshape(*oh.shape[:-3], -1)
        return jnp.concatenate([mm, obs["scalars"].astype(jnp.float32)], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self._flat(obs), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self._flat(obs), state, reward, done, info


# ── network: tile-embed -> GRU -> actor/critic ───────────────────────────────
class ScannedRNN(nn.Module):
    hidden: int

    @partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0,
             split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        carry = jnp.where(resets[:, None], jnp.zeros_like(carry), carry)
        new_carry, y = nn.GRUCell(features=self.hidden)(carry, ins)
        return new_carry, y

    @staticmethod
    def initialize_carry(batch, hidden):
        return jnp.zeros((batch, hidden))


class ActorCriticRNN(nn.Module):
    action_dim: int
    view_size: int
    token_dim: int = 32
    embed_hidden: int = 256
    gru_hidden: int = 256

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x                       # obs (T,N,flat), dones (T,N)
        V, K = self.view_size, C.NUM_TILES
        vvk = V * V * K
        oh = obs[..., :vvk].reshape(*obs.shape[:-1], V * V, K)
        tok = nn.Dense(self.token_dim, use_bias=False, name="tile_embed")(oh)
        tok = tok.reshape(*tok.shape[:-2], -1)
        h = jnp.concatenate([tok, obs[..., vvk:]], axis=-1)
        h = nn.relu(nn.Dense(self.embed_hidden, kernel_init=orthogonal(np.sqrt(2)))(h))
        self.sow("intermediates", "obs_embed", h)   # per-obs encoding, pre-GRU (no memory)
        hidden, feat = ScannedRNN(self.gru_hidden)(hidden, (h, dones))
        a = nn.relu(nn.Dense(self.embed_hidden, kernel_init=orthogonal(np.sqrt(2)))(feat))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(a)
        c = nn.relu(nn.Dense(self.embed_hidden, kernel_init=orthogonal(np.sqrt(2)))(feat))
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(c)
        return hidden, logits, jnp.squeeze(value, -1)


def categorical_sample(logits, key):
    return jax.random.categorical(key, logits, axis=-1)


def log_prob(logits, action):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, action[..., None], axis=-1)[..., 0]


def entropy(logits):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -(jnp.exp(logp) * logp).sum(-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    reached: jnp.ndarray


def _env_params(cfg):
    return EnvParams.from_config(
        cue_distribution="custom" if cfg["cue_list"] else "factorized",
        custom_cues=cfg["cue_list"] or None,
        max_steps=cfg["max_steps"], view_size=cfg["view_size"],
        center_wall_thickness=cfg["center_wall_thickness"], pre_cue_steps=cfg["pre_cue_steps"],
        pre_branch_corridor_len=cfg["pre_branch_corridor_len"], branch_len=cfg["branch_len"],
        post_branch_corridor_len=cfg["post_branch_corridor_len"], step_penalty=cfg["step_penalty"],
        branch_bonus=cfg["branch_bonus"], wrong_branch_penalty=cfg["wrong_branch_penalty"],
        success_reward=cfg["success_reward"], wrong_door_reward=cfg["wrong_door_reward"],
        shaping_coef=cfg["shaping_coef"], door_random_prob=cfg["door_random_prob"])


def make_train(cfg, log_cb=None):
    env = BatchEnvWrapper(AutoResetEnvWrapper(LogWrapper(
        FlattenObs(MemoryJaxEnv(default_params=_env_params(cfg)), cfg["view_size"]))),
        num_envs=cfg["num_envs"])
    n_updates = cfg["total_timesteps"] // (cfg["num_steps"] * cfg["num_envs"])
    minibatch_envs = cfg["num_envs"] // cfg["num_minibatches"]

    def lr_sched(count):
        frac = 1.0 - (count // (cfg["num_minibatches"] * cfg["update_epochs"])) / n_updates
        return cfg["lr"] * frac

    net = ActorCriticRNN(action_dim=C.NUM_ACTIONS, view_size=cfg["view_size"],
                         token_dim=cfg["token_dim"], embed_hidden=cfg["embed_hidden"],
                         gru_hidden=cfg["gru_hidden"])

    def train(rng):
        rng, k = jax.random.split(rng)
        init_hidden = ScannedRNN.initialize_carry(cfg["num_envs"], cfg["gru_hidden"])
        init_x = (jnp.zeros((1, cfg["num_envs"], env.flat_dim)),
                  jnp.zeros((1, cfg["num_envs"]), bool))
        params = net.init(k, ScannedRNN.initialize_carry(cfg["num_envs"], cfg["gru_hidden"]), init_x)
        tx = optax.chain(optax.clip_by_global_norm(cfg["max_grad_norm"]),
                         optax.adam(lr_sched if cfg["anneal_lr"] else cfg["lr"], eps=1e-5))
        train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        rng, k = jax.random.split(rng)
        obs, env_state = env.reset(k)
        last_done = jnp.zeros(cfg["num_envs"], bool)

        def _update(runner, upd_idx):
            train_state, env_state, last_obs, last_done, hidden, rng = runner
            init_hidden_upd = hidden

            def _step(carry, _):
                train_state, env_state, last_obs, last_done, hidden, rng = carry
                ac_in = (last_obs[None], last_done[None])
                hidden, logits, value = net.apply(train_state.params, hidden, ac_in)
                logits, value = logits[0], value[0]
                rng, ak = jax.random.split(rng)
                action = categorical_sample(logits, ak)
                lp = log_prob(logits, action)
                rng, sk = jax.random.split(rng)
                obs, env_state, reward, done, info = env.step(sk, env_state, action)
                trans = Transition(last_done, action, value, reward, lp, last_obs,
                                   info["reached_target"])
                return (train_state, env_state, obs, done, hidden, rng), (trans, info)

            (train_state, env_state, last_obs, last_done, hidden, rng), (traj, info) = jax.lax.scan(
                _step, (train_state, env_state, last_obs, last_done, hidden, rng),
                None, cfg["num_steps"])

            # bootstrap value
            ac_in = (last_obs[None], last_done[None])
            _, _, last_val = net.apply(train_state.params, hidden, ac_in)
            last_val = last_val[0]

            def _gae(carry, trans):
                gae, next_val, next_done = carry
                delta = trans.reward + cfg["gamma"] * next_val * (1 - next_done) - trans.value
                gae = delta + cfg["gamma"] * cfg["gae_lambda"] * (1 - next_done) * gae
                return (gae, trans.value, trans.done), gae

            _, advantages = jax.lax.scan(
                _gae, (jnp.zeros_like(last_val), last_val, last_done), traj,
                reverse=True, unroll=16)
            targets = advantages + traj.value

            def _epoch(carry, _):
                train_state, rng = carry
                rng, pk = jax.random.split(rng)
                perm = jax.random.permutation(pk, cfg["num_envs"])
                # keep time axis intact; shuffle along env axis into minibatches
                def mb_slice(x):  # x: (T, N, ...) -> (num_mb, T, mb_envs, ...)
                    x = jnp.take(x, perm, axis=1)
                    return x.reshape(x.shape[0], cfg["num_minibatches"], minibatch_envs,
                                     *x.shape[2:]).swapaxes(0, 1)
                h0 = jnp.take(init_hidden_upd, perm, axis=0).reshape(
                    cfg["num_minibatches"], minibatch_envs, cfg["gru_hidden"])
                batch = (h0, jax.tree_util.tree_map(mb_slice, traj),
                         mb_slice(advantages), mb_slice(targets))

                def _mb(train_state, data):
                    h0, traj, adv, tgt = data

                    def loss_fn(params):
                        _, logits, value = net.apply(params, h0, (traj.obs, traj.done))
                        lp = log_prob(logits, traj.action)
                        ratio = jnp.exp(lp - traj.log_prob)
                        a = (adv - adv.mean()) / (adv.std() + 1e-8)
                        l1 = ratio * a
                        l2 = jnp.clip(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * a
                        pg = -jnp.minimum(l1, l2).mean()
                        v_clip = traj.value + (value - traj.value).clip(-cfg["clip_eps"], cfg["clip_eps"])
                        vloss = 0.5 * jnp.maximum(jnp.square(value - tgt),
                                                  jnp.square(v_clip - tgt)).mean()
                        ent = entropy(logits).mean()
                        return pg + cfg["vf_coef"] * vloss - cfg["ent_coef"] * ent, (pg, vloss, ent)

                    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, *aux)

                train_state, metrics = jax.lax.scan(_mb, train_state, batch)
                return (train_state, rng), metrics

            (train_state, rng), _ = jax.lax.scan(
                _epoch, (train_state, rng), None, cfg["update_epochs"])

            # metrics from per-step info: episodes ended this rollout (returned_episode),
            # their mean return, and whether they reached the colour-correct door.
            ep = info["returned_episode"]
            denom = jnp.maximum(ep.sum(), 1.0)
            mean_ret = (info["returned_episode_returns"] * ep).sum() / denom
            succ = (info["reached_target"] & ep).sum() / denom
            length = (info["returned_episode_lengths"] * ep).sum() / denom
            metrics = {"return": mean_ret, "success": succ, "ep_len": length,
                       "n_ep": ep.sum(), "update": upd_idx}
            if log_cb is not None:
                jax.debug.callback(log_cb, metrics)
            return (train_state, env_state, last_obs, last_done, hidden, rng), metrics

        runner = (train_state, env_state, obs, last_done, init_hidden, rng)
        runner, metrics = jax.lax.scan(_update, runner, jnp.arange(n_updates))
        return {"train_state": runner[0], "metrics": metrics}

    return train, n_updates


TRAIN_CUES = {"2cue": ["green_up", "blue_down"],
              "3cue": ["green_up", "green_down", "blue_down"],
              "4cue": ["green_up", "blue_up", "green_down", "blue_down"]}


def default_cfg():
    return dict(
        view_size=5, center_wall_thickness=3, pre_cue_steps=1, pre_branch_corridor_len=5,
        branch_len=4, post_branch_corridor_len=5, max_steps=200,
        step_penalty=0.0, branch_bonus=0.5, wrong_branch_penalty=-1.0, success_reward=0.5,
        wrong_door_reward=0.0, shaping_coef=0.01, door_random_prob=1.0,
        cue_list=None,
        num_envs=256, num_steps=128, total_timesteps=15_000_000,
        update_epochs=4, num_minibatches=8, gamma=0.997, gae_lambda=0.95,
        clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=2.5e-4, anneal_lr=True,
        token_dim=32, embed_hidden=256, gru_hidden=256,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cue", choices=["2cue", "3cue", "4cue"], default="4cue")
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-mode", default="disabled")
    ap.add_argument("--tag", default="ppo")
    ap.add_argument("--set", nargs="*", default=[], help="cfg overrides k=v")
    a = ap.parse_args()

    cfg = default_cfg()
    cfg["cue"] = a.cue
    cfg["cue_list"] = TRAIN_CUES[a.cue]
    cfg["seed"] = a.seed          # persist the seed (was missing from config.json)
    if a.total_timesteps:
        cfg["total_timesteps"] = a.total_timesteps
    if a.num_envs:
        cfg["num_envs"] = a.num_envs
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except Exception:
            pass
        print(f"[cfg override] {k} = {v}", flush=True)
        cfg[k] = v

    run_dir = pathlib.Path("outputs/ppo_runs") / f"ppo_{a.cue}_{a.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    use_wandb = a.wandb_mode != "disabled"
    if use_wandb:
        import wandb
        wandb.init(project="memory_env_jax", name=f"ppo_{a.cue}_{a.tag}",
                   mode=a.wandb_mode, config=cfg,
                   tags=[f"algo=ppo_gru", f"cue={a.cue}", a.tag])

    t0 = time.time()
    state = {"n": 0}

    def log_cb(m):
        state["n"] += 1
        upd = int(m["update"])
        if upd % 20 == 0 or upd < 5:
            fps = (upd + 1) * cfg["num_steps"] * cfg["num_envs"] / max(time.time() - t0, 1e-6)
            print(f"[upd {upd:5d}] return={float(m['return']):.3f} success={float(m['success']):.3f} "
                  f"ep_len={float(m['ep_len']):.1f} n_ep={float(m['n_ep']):.0f} fps={fps:.0f}", flush=True)
        if use_wandb:
            import wandb
            wandb.log({"return/mean": float(m["return"]), "success/mean": float(m["success"]),
                       "rollout/episode_length": float(m["ep_len"])}, step=upd)

    train_fn, n_updates = make_train(cfg, log_cb=log_cb)
    print(f"[ppo] {a.cue} | {n_updates} updates | {cfg['total_timesteps']} steps | "
          f"num_envs={cfg['num_envs']} num_steps={cfg['num_steps']}", flush=True)
    out = jax.jit(train_fn)(jax.random.PRNGKey(a.seed))
    jax.block_until_ready(out)

    params = out["train_state"].params
    ckpt_dir = (run_dir / "checkpoints" / f"step_{cfg['total_timesteps']}").resolve()
    ocp.PyTreeCheckpointer().save(str(ckpt_dir), {"params": params})
    print(f"[ppo] done in {time.time()-t0:.0f}s; saved {ckpt_dir}", flush=True)
    if use_wandb:
        import wandb; wandb.finish()


if __name__ == "__main__":
    main()
