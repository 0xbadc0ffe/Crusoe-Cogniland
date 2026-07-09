"""DreamerV3 on the bridge_tunnel NATURAL-maps task — single-file, pure-JAX.

Trains the in-tree ``purejaxwm`` DreamerV3 on EXACTLY the task the PyTorch
PPO+GRU ``natural_agent`` was trained on (``released_models/bridge_tunnel/natural_agent.yaml``):
open procedural 32×64 terrain, water/rock obstacles the agent can bridge / mine
or walk around, a central goal door (goal_half=4). The env is the pure-JAX
``cogniland.bridge_tunnel.jax`` port, proven bit-for-bit equivalent to the PyTorch
``BridgeTunnelEnv`` in ``tests/test_bridge_tunnel_jax_parity.py``.

* Obs: dict ``{minimap: (V,V) int8, scalars: (5,) float32}`` flattened to
  ``(V*V + 5,)`` float32. Scalars = ``[facing one-hot (4), step/max]``.
* Encoder/decoder: paper-aligned 4-block MLP (the minimap is symbolic).
* Shared W&B schema with PPO: ``success/mean``, ``return/mean``,
  ``rollout/episode_length`` etc. under tags ``algo=dreamerv3`` / ``size=...``
  so it lands on the same comparison charts as the PPO natural_agent.

The model-size presets follow DreamerV3 paper Table 3 (same as
``dreamerv3_crafter_in_cogniland.py``).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

import flashbax as fbx
import orbax.checkpoint as ocp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # so `purejaxwm` resolves

from cogniland.memory_env.jax import (  # noqa: E402
    EnvParams,
    MemoryJaxEnv,
    constants as C,
)

from purejaxwm.commons import (  # noqa: E402
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    GymnaxWrapper,
    LogWrapper,
    resolve_dtype,
)
from purejaxwm.dreamerv3.laprop import laprop
from purejaxwm.dreamerv3 import behavior as ac
from purejaxwm.dreamerv3 import world_model as wm_losses
from purejaxwm.dreamerv3.distributions import TwoHotDist
from purejaxwm.dreamerv3.world_model import MLPHead, RSSM, State
from purejaxwm.dreamerv3.behavior import DreamerTrainState, RetNorm


# ── model size presets (paper Table 3) ────────────────────────────────
SIZE_PRESETS = {
    "12M":  dict(d=256,  deter=1024,  cnn_d=16, codes=16),
    "25M":  dict(d=384,  deter=3072,  cnn_d=24, codes=24),
    "50M":  dict(d=512,  deter=4096,  cnn_d=32, codes=32),
    "100M": dict(d=768,  deter=6144,  cnn_d=48, codes=48),
    "200M": dict(d=1024, deter=8192,  cnn_d=64, codes=64),
    "400M": dict(d=1536, deter=12288, cnn_d=96, codes=96),
}

SCALAR_DIM = 5   # [facing one-hot (4), step/max]


# ─────────────────────────────────────────────────────────────────────
# FlattenObsWrapper — dict obs → (FLAT_OBS_DIM,) float32
# ─────────────────────────────────────────────────────────────────────


class FlattenObsWrapper(GymnaxWrapper):
    """Flattens ``{minimap, scalars}`` into one float32 vector."""

    def __init__(self, env, view_size: int, decoder: str = "mse", scalar_dim: int = SCALAR_DIM):
        super().__init__(env)
        self.view_size = view_size
        self.decoder = decoder
        if decoder == "categorical":
            self.flat_dim = view_size * view_size * C.NUM_TILES + scalar_dim
        else:
            self.flat_dim = view_size * view_size + scalar_dim

    def _flatten(self, obs: dict) -> jax.Array:
        if self.decoder == "categorical":
            oh = jax.nn.one_hot(obs["minimap"].astype(jnp.int32), C.NUM_TILES)
            mm = oh.reshape(*oh.shape[:-3], -1)        # (...,V,V,K) → (...,V*V*K)
        else:
            mm = (obs["minimap"].astype(jnp.float32) / float(C.NUM_TILES))
            mm = mm.reshape(*mm.shape[:-2], -1)
        return jnp.concatenate([
            mm,
            obs["scalars"].astype(jnp.float32),
        ], axis=-1)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self._flatten(obs), state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self._flatten(obs), state, reward, done, info


# ─────────────────────────────────────────────────────────────────────
# MLP encoder / decoder on the flat obs vector
# ─────────────────────────────────────────────────────────────────────


class BridgeTunnelEncoder(nn.Module):
    hidden: int
    num_layers: int
    embed_dim: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden, use_bias=False,
                         dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = jax.nn.silu(x)
        x = nn.Dense(self.embed_dim, use_bias=False,
                     dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return jax.nn.silu(x)


class BridgeTunnelDecoder(nn.Module):
    hidden: int
    num_layers: int
    out_dim: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden, use_bias=False,
                         dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = jax.nn.silu(x)
        x = nn.Dense(self.out_dim, use_bias=True,
                     dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return x.astype(jnp.float32)


class SymbolicEmbedEncoder(nn.Module):
    """BT-Dreamer trunk on a LEARNED per-tile embedding of the symbolic minimap.

    The obs is the categorical (one-hot) flat vector ``[V*V*K one-hot, scalars]``.
    We reshape the one-hot to ``(V, V, K)`` and apply a single shared
    ``Dense(K -> token_dim)`` per cell — i.e. a learned tile-embedding table
    (one-hot @ W = row lookup), shared across all cells so a tile means the same
    thing everywhere — then flatten, append the scalars, and run the MLP trunk.
    Requires the categorical (one-hot) obs.
    """
    view_size: int
    n_tiles: int
    token_dim: int
    hidden: int
    num_layers: int
    embed_dim: int              # output width (== wm_hidden)
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        V, K = self.view_size, self.n_tiles
        vvk = V * V * K
        oh = x[..., :vvk].reshape(*x.shape[:-1], V * V, K)          # (..., V*V, K)
        tok = nn.Dense(self.token_dim, use_bias=False, name="tile_embed",
                       dtype=self.dtype, param_dtype=self.param_dtype)(oh)
        tok = tok.reshape(*tok.shape[:-2], -1)                       # (..., V*V*token_dim)
        h = jnp.concatenate([tok, x[..., vvk:]], axis=-1)           # + scalars
        for _ in range(self.num_layers):
            h = nn.Dense(self.hidden, use_bias=False,
                         dtype=self.dtype, param_dtype=self.param_dtype)(h)
            h = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(h)
            h = jax.nn.silu(h)
        h = nn.Dense(self.embed_dim, use_bias=False,
                     dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(h)
        return jax.nn.silu(h)


class SymbolicFactoredEncoder(nn.Module):
    """Factored tile encoder: learned per-tile embedding PLUS shared learned
    COLOUR and SHAPE attribute embeddings (from fixed tile->colour / tile->shape
    lookups). Cue-green and door-green share the SAME colour embedding, so
    colour->door is trivial matching; the cue orientation is a separate shape
    factor. Gives clean, separable shape/colour latents (for steering) and makes
    both mappings equally learnable. Requires the categorical (one-hot) obs.
    """
    view_size: int
    n_tiles: int
    token_dim: int
    attr_dim: int                 # colour / shape embedding width
    hidden: int
    num_layers: int
    embed_dim: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        V, K = self.view_size, self.n_tiles
        vvk = V * V * K
        oh = x[..., :vvk].reshape(*x.shape[:-1], V * V, K)            # (..., V*V, K)
        Mc = jax.nn.one_hot(jnp.asarray(C.TILE_COLOR), C.N_COLOR).astype(self.dtype)  # (K, 3)
        Ms = jax.nn.one_hot(jnp.asarray(C.TILE_SHAPE), C.N_SHAPE).astype(self.dtype)  # (K, 3)
        col_oh = oh @ Mc                                             # (..., V*V, N_COLOR)
        shp_oh = oh @ Ms                                             # (..., V*V, N_SHAPE)
        tile_e = nn.Dense(self.token_dim, use_bias=False, name="tile_embed",
                          dtype=self.dtype, param_dtype=self.param_dtype)(oh)
        col_e = nn.Dense(self.attr_dim, use_bias=False, name="color_embed",
                         dtype=self.dtype, param_dtype=self.param_dtype)(col_oh)
        shp_e = nn.Dense(self.attr_dim, use_bias=False, name="shape_embed",
                         dtype=self.dtype, param_dtype=self.param_dtype)(shp_oh)
        cell = jnp.concatenate([tile_e, col_e, shp_e], axis=-1)      # (..., V*V, token+2*attr)
        cell = cell.reshape(*cell.shape[:-2], -1)
        h = jnp.concatenate([cell, x[..., vvk:]], axis=-1)          # + scalars
        for _ in range(self.num_layers):
            h = nn.Dense(self.hidden, use_bias=False,
                         dtype=self.dtype, param_dtype=self.param_dtype)(h)
            h = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(h)
            h = jax.nn.silu(h)
        h = nn.Dense(self.embed_dim, use_bias=False,
                     dtype=self.dtype, param_dtype=self.param_dtype)(h)
        h = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(h)
        return jax.nn.silu(h)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    is_first: jnp.ndarray
    is_last: jnp.ndarray
    is_terminal: jnp.ndarray
    reached: jnp.ndarray   # extra info channel for success-rate logging
    belief: jnp.ndarray    # () int32 map-category target (btc; zeros for bt)


# cue subsets per task (mirrors scripts/memory_env/datasets.py TRAIN_CUES)
_CUE_SUBSETS = {
    "2cue": ["green_up", "blue_down"],
    "3cue": ["green_up", "green_down", "blue_down"],
    "4cue": ["green_up", "blue_up", "green_down", "blue_down"],
}


def _make_env_params(cfg) -> EnvParams:
    """MemoryEnv geometry is deterministic from config (no map dataset)."""
    cue = cfg.get("cue", "4cue")
    return EnvParams.from_config(
        cue_distribution="custom", custom_cues=_CUE_SUBSETS[cue],
        max_steps=cfg["max_steps"], view_size=cfg["view_size"],
        center_wall_thickness=cfg["center_wall_thickness"],
        pre_cue_steps=cfg["pre_cue_steps"],
        pre_branch_corridor_len=cfg["pre_branch_corridor_len"],
        branch_len=cfg["branch_len"],
        post_branch_corridor_len=cfg["post_branch_corridor_len"],
        step_penalty=cfg["step_penalty"], branch_bonus=cfg["branch_bonus"],
        wrong_branch_penalty=cfg.get("wrong_branch_penalty", 0.0),
        success_reward=cfg["success_reward"], wrong_door_reward=cfg["wrong_door_reward"],
        shaping_coef=cfg["shaping_coef"],
        door_random_prob=cfg.get("door_random_prob", 1.0),
    )


def make_train(cfg, log_cb=None):
    env_params = _make_env_params(cfg)
    decoder_mode = cfg.get("decoder", "mse")
    V = cfg["view_size"]
    K = C.NUM_TILES
    scalar_dim = 5   # [facing one-hot (4), step/max]
    if decoder_mode == "categorical":
        flat_dim = V * V * K + scalar_dim
    else:
        flat_dim = V * V + scalar_dim
    base_env = MemoryJaxEnv(default_params=env_params)
    env = BatchEnvWrapper(
        AutoResetEnvWrapper(LogWrapper(
            FlattenObsWrapper(base_env, cfg["view_size"], decoder=decoder_mode,
                              scalar_dim=scalar_dim))),
        num_envs=cfg["num_envs"],
    )
    action_dim = C.NUM_ACTIONS
    obs_shape = (flat_dim,)

    compute_dtype = resolve_dtype(cfg.get("compute_dtype", "float32"))
    param_dtype = jnp.float32

    if cfg.get("obs_factored", True):
        assert decoder_mode == "categorical", "obs_factored requires the categorical (one-hot) obs"
        encoder = SymbolicFactoredEncoder(
            view_size=V, n_tiles=K, token_dim=cfg.get("token_dim", 16),
            attr_dim=cfg.get("attr_dim", 8),
            hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"],
            embed_dim=cfg["wm_hidden"], dtype=compute_dtype, param_dtype=param_dtype,
        )
    elif cfg.get("obs_embed", True):
        assert decoder_mode == "categorical", "obs_embed requires the categorical (one-hot) obs"
        encoder = SymbolicEmbedEncoder(
            view_size=V, n_tiles=K, token_dim=cfg.get("token_dim", 16),
            hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"],
            embed_dim=cfg["wm_hidden"], dtype=compute_dtype, param_dtype=param_dtype,
        )
    else:
        encoder = BridgeTunnelEncoder(
            hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"],
            embed_dim=cfg["wm_hidden"],
            dtype=compute_dtype, param_dtype=param_dtype,
        )
    decoder = BridgeTunnelDecoder(
        hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"], out_dim=flat_dim,
        dtype=compute_dtype, param_dtype=param_dtype,
    )
    rssm = RSSM(
        deter_dim=cfg["deter"], stoch_size=cfg["stoch"], classes=cfg["classes"],
        hidden=cfg["wm_hidden"], unimix=cfg["unimix"], blocks=cfg["blocks"],
        dtype=compute_dtype, param_dtype=param_dtype,
    )
    reward_head = MLPHead(
        hidden=cfg["wm_hidden"], num_layers=1, out_dim=cfg["num_reward_bins"],
        outscale=0.0, dtype=compute_dtype, param_dtype=param_dtype,
    )
    cont_head = MLPHead(
        hidden=cfg["wm_hidden"], num_layers=1, out_dim=1,
        outscale=1.0, dtype=compute_dtype, param_dtype=param_dtype,
    )
    actor_head = MLPHead(
        hidden=cfg["ac_hidden"], num_layers=cfg["ac_layers"], out_dim=action_dim,
        outscale=0.01, dtype=compute_dtype, param_dtype=param_dtype,
    )
    critic_head = MLPHead(
        hidden=cfg["ac_hidden"], num_layers=cfg["ac_layers"], out_dim=cfg["num_reward_bins"],
        outscale=0.0, dtype=compute_dtype, param_dtype=param_dtype,
    )
    # auxiliary belief head: classifies map category (rocky/balanced/lakes) from the
    # RSSM model state; CE aux loss (loss_belief) shapes the latent. btc-only.
    use_belief = False   # MemoryEnv has no map-category belief target
    belief_head = (MLPHead(
        hidden=cfg["wm_hidden"], num_layers=1, out_dim=3,
        outscale=0.0, dtype=compute_dtype, param_dtype=param_dtype,
    ) if use_belief else None)

    dummy_transition = Transition(
        obs=jnp.zeros(obs_shape, dtype=jnp.float32),
        action=jnp.zeros(action_dim, dtype=jnp.float32),
        reward=jnp.zeros((), dtype=jnp.float32),
        is_first=jnp.zeros((), dtype=bool),
        is_last=jnp.zeros((), dtype=bool),
        is_terminal=jnp.zeros((), dtype=bool),
        reached=jnp.zeros((), dtype=bool),
        belief=jnp.zeros((), dtype=jnp.int32),
    )
    buffer = fbx.make_trajectory_buffer(
        add_batch_size=cfg["num_envs"],
        sample_batch_size=cfg["batch_size"],
        sample_sequence_length=cfg["seq_len"],
        period=1,
        min_length_time_axis=max(cfg["buffer_min_size"] // cfg["num_envs"], cfg["seq_len"]),
        max_length_time_axis=max(cfg["buffer_capacity"] // cfg["num_envs"], cfg["seq_len"]),
    )

    def _rec_loss_categorical(pred, target):    # both (N, V*V*K + SCALAR_DIM)
        vvK = V * V * K
        mlogit = pred[:, :vvK].reshape(-1, V * V, K)
        mtarget = target[:, :vvK].reshape(-1, V * V, K)        # one-hot
        logp = jax.nn.log_softmax(mlogit, axis=-1)
        ce = -(mtarget * logp).sum(axis=-1).sum(axis=-1)       # (N,) sum over cells
        s_mse = 0.5 * jnp.square(pred[:, vvK:] - target[:, vvK:]).sum(axis=-1)
        return (ce + s_mse).mean()

    rec_loss_fn = _rec_loss_categorical if decoder_mode == "categorical" else None

    num_updates = cfg["total_env_steps"] // cfg["num_envs"]
    ckpt_interval = cfg.get("ckpt_interval_updates") or num_updates
    chunk_updates = max(1, min(int(ckpt_interval), num_updates))

    wm_tx = laprop(lr=cfg["lr_wm"], agc=cfg["max_grad_norm"], eps=cfg["opt_eps"])
    ac_tx = laprop(lr=cfg["lr_ac"], agc=cfg["max_grad_norm"], eps=cfg["opt_eps"])

    def init_carry_fn(rng):
        rng, sub = jax.random.split(rng)
        dummy_obs = jnp.zeros((1,) + obs_shape)
        enc_params = encoder.init(sub, dummy_obs)
        dummy_embed = encoder.apply(enc_params, dummy_obs)

        rng, sub = jax.random.split(rng)
        init_rssm_state = rssm.initial_state((1,))
        dummy_action = jnp.zeros((1, action_dim))
        dummy_is_first = jnp.zeros((1,), dtype=bool)
        rssm_params = rssm.init(
            {"params": sub, "stoch": sub},
            init_rssm_state, dummy_action, dummy_embed, dummy_is_first,
        )
        _, _post = rssm.apply(
            rssm_params, init_rssm_state, dummy_action, dummy_embed, dummy_is_first,
            rngs={"stoch": sub},
        )
        dummy_feat = _post.features()

        rng, s_dec, s_rew, s_cont, s_act, s_crit = jax.random.split(rng, 6)
        dec_params = decoder.init(s_dec, dummy_feat)
        rew_params = reward_head.init(s_rew, dummy_feat)
        cont_params = cont_head.init(s_cont, dummy_feat)
        actor_params = actor_head.init(s_act, dummy_feat)
        critic_params = critic_head.init(s_crit, dummy_feat)

        wm_params = {
            "encoder": enc_params, "rssm": rssm_params, "decoder": dec_params,
            "reward": rew_params, "cont": cont_params,
        }
        if use_belief:
            rng, s_bel = jax.random.split(rng)
            wm_params["belief"] = belief_head.init(s_bel, dummy_feat)
        ac_params = {"actor": actor_params, "critic": critic_params}
        slow_critic_params = critic_params

        wm_opt_state = wm_tx.init(wm_params)
        ac_opt_state = ac_tx.init(ac_params)

        buffer_state = buffer.init(dummy_transition)

        rng, sub = jax.random.split(rng)
        obs0, env_state = env.reset(sub, None)
        rssm_state_act = rssm.initial_state((cfg["num_envs"],))

        last_action = jnp.zeros((cfg["num_envs"], action_dim))
        last_reward = jnp.zeros((cfg["num_envs"],))
        last_done = jnp.zeros((cfg["num_envs"],), dtype=bool)
        last_terminal = jnp.zeros((cfg["num_envs"],), dtype=bool)
        last_is_first = jnp.ones((cfg["num_envs"],), dtype=bool)
        last_reached = jnp.zeros((cfg["num_envs"],), dtype=bool)
        last_belief = jnp.zeros((cfg["num_envs"],), dtype=jnp.int32)

        train_state = DreamerTrainState(
            wm_params=wm_params,
            ac_params=ac_params,
            slow_critic_params=slow_critic_params,
            opt_state={"wm": wm_opt_state, "ac": ac_opt_state},
            retnorm=RetNorm.initial(),
            step=jnp.array(0),
            train_step=jnp.array(0),
        )

        init_carry = (
            train_state, env_state, buffer_state, obs0,
            last_action, last_reward, last_done, last_terminal, last_is_first,
            last_reached, last_belief, rssm_state_act, jnp.float32(0.0), rng,
        )
        return init_carry

    def _rollout_step(carry, _):
        (train_state, env_state, buffer_state, obs,
         last_action, last_reward, last_done, last_terminal, last_is_first,
         last_reached, last_belief, rssm_state_act, rng) = carry

        action_masked = jnp.where(
            last_is_first[..., None], jnp.zeros_like(last_action), last_action
        )
        reward_stored = jnp.where(last_is_first, 0.0, last_reward)

        transition = Transition(
            obs=obs, action=action_masked, reward=reward_stored,
            is_first=last_is_first, is_last=last_done,
            is_terminal=last_terminal, reached=last_reached, belief=last_belief,
        )
        buffer_state = buffer.add(
            buffer_state,
            jax.tree_util.tree_map(lambda x: x[:, None, ...], transition),
        )

        rng, s_stoch, s_pol = jax.random.split(rng, 3)
        embed = encoder.apply(train_state.wm_params["encoder"], obs)
        _, posterior = rssm.apply(
            train_state.wm_params["rssm"],
            rssm_state_act, action_masked, embed, last_is_first,
            rngs={"stoch": s_stoch},
        )
        rssm_state_act = posterior
        feat = posterior.features()
        logits = ac.unimix_logits(
            actor_head.apply(train_state.ac_params["actor"], feat)
        )
        action_idx = jax.random.categorical(s_pol, logits)
        action_oh = jax.nn.one_hot(action_idx, action_dim)

        rng, s_step = jax.random.split(rng)
        next_obs, env_state, reward_next, done_next, info = env.step(
            s_step, env_state, action_idx, None
        )
        terminal_next = done_next
        reached_next = info["reached_target"]
        belief_next = info["category"].astype(jnp.int32)   # map category of next_obs

        new_carry = (
            train_state, env_state, buffer_state, next_obs,
            action_oh, reward_next, done_next, terminal_next, done_next,
            reached_next, belief_next, rssm_state_act, rng,
        )
        completed = info["returned_episode"].astype(jnp.float32)
        n_completed = jnp.maximum(completed.sum(), 1.0)
        success_episodes = (info["reached_target"] & info["returned_episode"]).astype(jnp.float32)
        # Path-efficiency: map width is the left→right span the agent must cover.
        min_steps = jnp.float32(cfg["map_width"])
        per_ep_ratio = min_steps / jnp.maximum(
            info["returned_episode_lengths"].astype(jnp.float32), 1.0,
        )
        metrics = {
            "rollout/reward_step_mean": reward_next.mean(),
            "rollout/done_frac": done_next.mean(),
            "return/mean": (info["returned_episode_returns"] * completed).sum() / n_completed,
            "return/min_over_steps": (per_ep_ratio * completed).sum() / n_completed,
            "rollout/returned_episode_count": completed.sum(),
            "success/mean": success_episodes.sum() / n_completed,
            "rollout/episode_length": (info["returned_episode_lengths"] * completed).sum() / n_completed,
        }
        return new_carry, metrics

    def _train_step(train_state: DreamerTrainState, buffer_state, rng):
        rng, s_sample, s_wm_rng, s_ac_rng = jax.random.split(rng, 4)
        batch = buffer.sample(buffer_state, s_sample).experience

        def swap_TB(x):
            return jnp.swapaxes(x, 0, 1)

        obs_b = swap_TB(batch.obs)
        action_b = swap_TB(batch.action)
        reward_b = swap_TB(batch.reward)
        is_first_b = swap_TB(batch.is_first)
        is_terminal_b = swap_TB(batch.is_terminal)
        belief_b = swap_TB(batch.belief)

        def _wm_loss_fn(wm_params, rng):
            init_state = rssm.initial_state((cfg["batch_size"],))
            enc_apply = lambda p, o: encoder.apply(p, o)
            dec_apply = lambda p, f: decoder.apply(p, f)
            rew_apply = lambda p, f: TwoHotDist(reward_head.apply(p, f))
            cont_apply = lambda p, f: cont_head.apply(p, f).squeeze(-1)
            wm_dict = {
                "encoder": wm_params["encoder"], "rssm": wm_params["rssm"],
                "decoder": wm_params["decoder"], "reward": wm_params["reward"],
                "cont": wm_params["cont"],
            }
            belief_apply = belief_target = None
            belief_scale = 0.0
            if use_belief:
                wm_dict["belief"] = wm_params["belief"]
                belief_apply = lambda p, f: belief_head.apply(p, f)
                belief_target = belief_b
                belief_scale = cfg["loss_belief"]
            total, aux = wm_losses.wm_loss(
                wm_dict,
                encoder_apply=enc_apply, rssm=rssm,
                decoder_apply=dec_apply, reward_apply=rew_apply,
                cont_apply=cont_apply,
                obs=obs_b, action=action_b, reward=reward_b,
                is_first=is_first_b, is_terminal=is_terminal_b,
                init_rssm_state=init_state, rng=rng,
                loss_scales={
                    "rec": cfg["loss_rec"], "rew": cfg["loss_rew"],
                    "cont": cfg["loss_con"], "dyn": cfg["loss_dyn"],
                    "rep": cfg["loss_rep"],
                },
                free_nats=cfg["free_nats"],
                rec_loss_fn=rec_loss_fn,
                belief_apply=belief_apply, belief_target=belief_target,
                belief_scale=belief_scale,
            )
            return total, aux

        (wm_total, wm_aux), wm_grads = jax.value_and_grad(
            _wm_loss_fn, has_aux=True
        )(train_state.wm_params, s_wm_rng)
        wm_updates, new_wm_opt = wm_tx.update(
            wm_grads, train_state.opt_state["wm"], train_state.wm_params
        )
        new_wm_params = optax.apply_updates(train_state.wm_params, wm_updates)

        post = wm_aux.post

        def flat_state(s: State) -> State:
            Tt, Bb = s.deter.shape[0], s.deter.shape[1]
            return State(
                deter=s.deter.reshape(Tt * Bb, -1),
                stoch=s.stoch.reshape(Tt * Bb, *s.stoch.shape[2:]),
                logits=s.logits.reshape(Tt * Bb, *s.logits.shape[2:]),
            )

        post_flat = jax.tree_util.tree_map(jax.lax.stop_gradient, flat_state(post))

        def _ac_loss_fn(ac_params, rng):
            reward_apply = lambda f: TwoHotDist(reward_head.apply(new_wm_params["reward"], f))
            cont_apply = lambda f: cont_head.apply(new_wm_params["cont"], f).squeeze(-1)
            imag_total, (imag_aux, new_retnorm) = ac.imag_loss(
                ac_params,
                slow_critic_params=train_state.slow_critic_params,
                rssm=rssm,
                rssm_params=new_wm_params["rssm"],
                actor_head=actor_head, critic_head=critic_head,
                init_state=post_flat,
                reward_head_apply=reward_apply,
                cont_head_apply=cont_apply,
                retnorm=train_state.retnorm,
                action_dim=action_dim,
                horizon=cfg["imag_horizon"],
                gamma=cfg["gamma"],
                gae_lambda=cfg["gae_lambda"],
                entropy_coef=cfg["entropy_coef"],
                slow_reg_coef=cfg["slow_reg_coef"],
                percentile_lo=cfg["advantage_pct_lo"],
                percentile_hi=cfg["advantage_pct_hi"],
                retnorm_rate=cfg["retnorm_rate"],
                contdisc=cfg["contdisc"],
                slowtar=cfg["slowtar"],
                rng=rng,
            )
            Trep, Brep = obs_b.shape[0], obs_b.shape[1]
            bootstrap_targets = imag_aux.returns_start.reshape(Trep, Brep)
            feats_sg = jax.lax.stop_gradient(post.features())
            rloss, raux = ac.repl_loss(
                ac_params,
                slow_critic_params=train_state.slow_critic_params,
                critic_head=critic_head,
                replay_features_sg=feats_sg,
                replay_rewards=reward_b,
                replay_is_terminal=is_terminal_b,
                bootstrap_values_sg=bootstrap_targets,
                gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"],
                slow_reg_coef=cfg["slow_reg_coef"],
            )
            total = (
                cfg["loss_actor"] * imag_aux.actor_loss
                + cfg["loss_critic"] * imag_aux.critic_loss
                + cfg["loss_repval"] * rloss
            )
            return total, (imag_aux, new_retnorm, raux)

        (ac_total, (imag_aux, new_retnorm, repl_aux)), ac_grads = jax.value_and_grad(
            _ac_loss_fn, has_aux=True
        )(train_state.ac_params, s_ac_rng)
        ac_updates, new_ac_opt = ac_tx.update(
            ac_grads, train_state.opt_state["ac"], train_state.ac_params
        )
        new_ac_params = optax.apply_updates(train_state.ac_params, ac_updates)

        new_slow_critic = ac.slow_critic_update(
            train_state.slow_critic_params, new_ac_params["critic"],
            ema_rate=cfg["slow_ema_rate"],
        )
        new_state = DreamerTrainState(
            wm_params=new_wm_params, ac_params=new_ac_params,
            slow_critic_params=new_slow_critic,
            opt_state={"wm": new_wm_opt, "ac": new_ac_opt},
            retnorm=new_retnorm, step=train_state.step,
            train_step=train_state.train_step + 1,
        )
        tm = {
            "loss/wm_total": wm_total,
            "loss/rec": wm_aux.rec, "loss/reward": wm_aux.rew,
            "loss/cont": wm_aux.cont, "loss/dyn": wm_aux.dyn, "loss/rep": wm_aux.rep,
            "loss/belief": wm_aux.belief, "belief/acc": wm_aux.belief_acc,
            "loss/policy": imag_aux.actor_loss, "loss/value": imag_aux.critic_loss,
            "loss/entropy": imag_aux.entropy,
            "loss/repval": repl_aux.repl_loss,
            "retnorm/low": new_retnorm.low, "retnorm/high": new_retnorm.high,
            "imag/ret": imag_aux.return_mean, "imag/val": imag_aux.value_mean,
            "imag/rew": imag_aux.reward_mean,
            "imag/adv_std": imag_aux.advantage_std,
        }
        return new_state, tm

    rate_per_outer = (
        cfg["train_ratio"] / (cfg["batch_size"] * cfg["seq_len"]) * cfg["num_envs"]
    )
    MAX_PER_STEP = math.ceil(rate_per_outer) + 1

    _zero = {
        k: jnp.float32(0.0) for k in [
            "loss/wm_total", "loss/rec", "loss/reward", "loss/cont", "loss/dyn",
            "loss/rep", "loss/belief", "belief/acc",
            "loss/policy", "loss/value", "loss/entropy",
            "loss/repval", "retnorm/low", "retnorm/high", "imag/ret",
            "imag/val", "imag/rew", "imag/adv_std",
        ]
    }

    def _outer_step(carry, _):
        (train_state, env_state, buffer_state, obs,
         last_action, last_reward, last_done, last_terminal, last_is_first,
         last_reached, last_belief, rssm_state_act, train_debt, rng) = carry

        inner_carry = (
            train_state, env_state, buffer_state, obs,
            last_action, last_reward, last_done, last_terminal, last_is_first,
            last_reached, last_belief, rssm_state_act, rng,
        )
        new_inner, rollout_metrics = _rollout_step(inner_carry, None)
        (train_state, env_state, buffer_state, obs,
         last_action, last_reward, last_done, last_terminal, last_is_first,
         last_reached, last_belief, rssm_state_act, rng) = new_inner

        train_state = train_state._replace(step=train_state.step + cfg["num_envs"])

        can_sample = buffer.can_sample(buffer_state)
        past_warmup = train_state.step >= cfg["warmup_steps"]
        can_train = jnp.logical_and(can_sample, past_warmup)
        train_debt = jnp.where(can_train, train_debt + rate_per_outer, jnp.float32(0.0))
        n_upd = jnp.where(
            can_train, jnp.floor(train_debt).astype(jnp.int32), jnp.int32(0)
        )
        n_upd = jnp.minimum(n_upd, jnp.int32(MAX_PER_STEP))

        def _train_body(ts_rng, i):
            ts, rng = ts_rng
            should = i < n_upd
            rng, sub = jax.random.split(rng)

            def _do(ts_, sub_):
                return _train_step(ts_, buffer_state, sub_)

            def _skip(ts_, sub_):
                return ts_, _zero

            new_ts, tm = jax.lax.cond(should, _do, _skip, ts, sub)
            return (new_ts, rng), tm

        (train_state, rng), metrics_stack = jax.lax.scan(
            _train_body, (train_state, rng), jnp.arange(MAX_PER_STEP)
        )
        train_debt = train_debt - n_upd.astype(jnp.float32)
        count_f = jnp.maximum(n_upd.astype(jnp.float32), 1.0)
        train_metrics = jax.tree_util.tree_map(
            lambda x: x.sum(axis=0) / count_f, metrics_stack
        )

        all_metrics = {
            **rollout_metrics, **train_metrics,
            "schedule/train_steps": train_state.train_step.astype(jnp.float32),
        }
        if log_cb is not None:
            jax.debug.callback(log_cb, all_metrics, train_state.step, ordered=True)

        new_carry = (
            train_state, env_state, buffer_state, obs,
            last_action, last_reward, last_done, last_terminal, last_is_first,
            last_reached, last_belief, rssm_state_act, train_debt, rng,
        )
        return new_carry, all_metrics

    def run_chunk_fn(carry):
        final_carry, metrics_stacked = jax.lax.scan(
            _outer_step, carry, None, chunk_updates
        )
        return final_carry, metrics_stacked

    return init_carry_fn, run_chunk_fn, chunk_updates


def _save_final_checkpoint(final_state, run_dir: Path, env_step: int) -> None:
    ckpt_dir = (run_dir / "checkpoints" / f"step_{env_step}").resolve()
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wm_params": final_state.wm_params,
        "ac_params": final_state.ac_params,
        "slow_critic_params": final_state.slow_critic_params,
        "retnorm": {
            "low": final_state.retnorm.low,
            "high": final_state.retnorm.high,
            "count": final_state.retnorm.count,
        },
    }
    payload = jax.tree_util.tree_map(np.asarray, payload)
    ocp.PyTreeCheckpointer().save(str(ckpt_dir), payload)


def _default_cfg() -> dict:
    """Paper-aligned Dreamer defaults; env block = modified MemoryEnv."""
    return {
        # env (MemoryEnv geometry/reward — mirrors MemoryEnvConfig defaults)
        "cue": "4cue",            # "2cue" | "3cue" | "4cue"
        "env_id": "memory_jax",
        "view_size": 5,
        "max_steps": 200,
        "map_width": 24,          # ~door column; used only for the path-eff metric
        "center_wall_thickness": 3,
        "pre_cue_steps": 1,
        "pre_branch_corridor_len": 5,
        "branch_len": 4,
        "post_branch_corridor_len": 5,
        # decoder: "categorical" (per-cell softmax CE on the one-hot tile minimap,
        # natural for discrete tiles) or "mse" (normalized-id MSE).
        "decoder": "categorical",
        "obs_factored": True,     # factored encoder: shared colour + shape attribute embeddings
        "obs_embed": True,        # (fallback) plain learned per-tile embedding encoder
        "token_dim": 16,          # per-tile embedding width
        "attr_dim": 8,            # shared colour / shape attribute embedding width
        "step_penalty": 0.0,
        "branch_bonus": 0.5,
        "wrong_branch_penalty": 0.0,   # penalty on entering the shape-wrong branch (makes it a real subgoal)
        "door_random_prob": 1.0,       # curriculum: 1.0=random doors (target); 0.0=fixed (unconditional)
        "success_reward": 0.5,
        "wrong_door_reward": 0.0,
        "shaping_coef": 0.01,
        # train budget (match the PyTorch tr64 baseline for a fair fps comparison)
        "num_envs": 16,
        "total_env_steps": 2_000_000,
        "train_ratio": 64,
        "warmup_steps": 1024,
        "seed": 0,
        # buffer
        "buffer_capacity": 200_000,
        "buffer_min_size": 1024,
        "batch_size": 16,
        "seq_len": 64,
        # world model (filled by --size)
        "deter": 3072,
        "stoch": 32,
        "classes": 24,
        "blocks": 8,
        "wm_hidden": 384,
        "unimix": 0.01,
        "free_nats": 1.0,
        "num_reward_bins": 255,
        # encoder
        "enc_hidden": 384,
        "enc_layers": 4,
        # actor-critic
        "ac_hidden": 384,
        "ac_layers": 3,
        "imag_horizon": 15,
        "gamma": 0.997,
        "gae_lambda": 0.95,
        "entropy_coef": 3e-4,
        "slow_ema_rate": 0.02,
        "slow_reg_coef": 1.0,
        "contdisc": True,
        "slowtar": True,
        "retnorm_rate": 0.01,
        "advantage_pct_lo": 5.0,
        "advantage_pct_hi": 95.0,
        # optim
        "lr_wm": 4e-5,
        "lr_ac": 4e-5,
        "opt_eps": 1e-20,
        "max_grad_norm": 0.3,
        # losses
        "loss_rec": 1.0, "loss_rew": 1.0, "loss_con": 1.0,
        "loss_dyn": 1.0, "loss_rep": 0.1,
        "loss_belief": 0.0,   # btc-only aux map-category CE weight (0=off); enable via --set
        "loss_actor": 1.0, "loss_critic": 1.0, "loss_repval": 0.3,
        # mixed-precision
        "compute_dtype": "bfloat16",
        # checkpoint
        "ckpt_interval_updates": None,
        # wandb
        "wandb_project": "memory_env_jax",
        "wandb_mode": "online",
        "wandb_log_interval": 50,
    }


def _apply_size(cfg: dict, size: str) -> dict:
    preset = SIZE_PRESETS[size]
    cfg = dict(cfg)
    cfg["wm_hidden"] = preset["d"]
    cfg["enc_hidden"] = preset["d"]
    cfg["ac_hidden"] = preset["d"]
    cfg["deter"] = preset["deter"]
    cfg["classes"] = preset["codes"]
    cfg["stoch"] = preset["codes"]
    return cfg


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="25M", choices=list(SIZE_PRESETS))
    p.add_argument("--cue", default="4cue", choices=("2cue", "3cue", "4cue"),
                   help="cue subset to train on")
    p.add_argument("--decoder", default=None, choices=("mse", "categorical"),
                   help="obs reconstruction mode (default categorical)")
    p.add_argument("--total-env-steps", type=int, default=None)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--train-ratio", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-mode", default=None, choices=("online", "offline", "disabled"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--run-dir", default="outputs/dreamer_runs")
    p.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VAL",
                   help="override any cfg key, e.g. --set entropy_coef=0.01 num_envs=32")
    args = p.parse_args()

    cfg = _apply_size(_default_cfg(), args.size)
    for k in ("decoder", "total_env_steps", "num_envs", "train_ratio",
              "wandb_project", "wandb_mode"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    cfg["cue"] = args.cue
    cfg["env_id"] = f"memory_jax_{args.cue}"
    cfg["seed"] = args.seed
    # persisted so the viz can reconstruct the decoder head + obs layout.
    cfg["num_tiles"] = int(C.NUM_TILES)
    cfg.setdefault("view_size", _default_cfg()["view_size"])
    # generic cfg overrides (--set key=value), type-matched to the existing default
    for kv in args.overrides:
        key, _, val = kv.partition("=")
        if key in cfg and isinstance(cfg[key], bool):
            cfg[key] = val.lower() in ("1", "true", "yes")
        elif key in cfg and isinstance(cfg[key], int):
            cfg[key] = int(float(val))
        elif key in cfg and isinstance(cfg[key], float):
            cfg[key] = float(val)
        else:
            try:
                cfg[key] = json.loads(val)
            except Exception:
                cfg[key] = val
        print(f"[cfg override] {key} = {cfg[key]}", flush=True)

    run_id = args.run_name or f"dreamerv3_{cfg['env_id']}_size{args.size}_seed{cfg['seed']}_{int(time.time())}"
    run_dir = Path(args.run_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    wandb_active = cfg["wandb_mode"] != "disabled"
    if wandb_active:
        wandb.init(
            project=cfg["wandb_project"],
            name=run_id,
            mode=cfg["wandb_mode"],
            config=cfg,
            group=cfg["cue"],
            tags=[
                f"size={args.size}",
                "algo=dreamerv3",
                f"env={cfg['env_id']}",
                f"cue={cfg['cue']}",
                f"decoder={cfg['decoder']}",
            ],
            settings=wandb.Settings(_disable_stats=True),
        )
        print(f"[wandb] run id={wandb.run.id}  url={wandb.run.url}", flush=True)

    _last = {"step": 0, "time": time.time()}
    _success_history: list[tuple[int, float]] = []
    _return_history: list[tuple[int, float]] = []

    def log_cb(metrics_dict, env_step):
        n_completed = int(metrics_dict.get("rollout/returned_episode_count", 0))
        if n_completed > 0:
            _success_history.append((n_completed, float(metrics_dict["success/mean"])))
            _return_history.append((n_completed, float(metrics_dict["return/mean"])))

        step = int(env_step)
        if step % cfg["wandb_log_interval"] != 0:
            return
        payload = {}
        for k, v in metrics_dict.items():
            try:
                payload[k] = float(v)
            except (TypeError, ValueError):
                continue
        if _success_history:
            recent = _success_history[-200:]
            tot_n = sum(n for n, _ in recent)
            wsum = sum(n * r for n, r in recent)
            if tot_n > 0:
                payload["success/rolling100"] = wsum / tot_n
        if _return_history:
            recent = _return_history[-200:]
            tot_n = sum(n for n, _ in recent)
            wsum = sum(n * r for n, r in recent)
            if tot_n > 0:
                payload["return/rolling100"] = wsum / tot_n

        now = time.time()
        dt = now - _last["time"]
        ds = step - _last["step"]
        if dt > 0 and ds > 0:
            payload["perf/fps"] = ds / dt
        _last["step"] = step; _last["time"] = now
        wandb.log(payload, step=step)

    live_cb = log_cb if wandb_active else None
    init_carry_fn, run_chunk_fn, chunk_updates = make_train(cfg, log_cb=live_cb)
    num_updates = cfg["total_env_steps"] // cfg["num_envs"]
    num_chunks = math.ceil(num_updates / chunk_updates)
    print(f"chunk_updates={chunk_updates}; num_chunks={num_chunks}; "
          f"total_env_steps≈{num_chunks * chunk_updates * cfg['num_envs']}", flush=True)

    rng = jax.random.PRNGKey(cfg["seed"])
    jit_init = jax.jit(init_carry_fn)
    jit_chunk = jax.jit(run_chunk_fn)

    print("init carry...", flush=True)
    t0 = time.time()
    carry = jit_init(rng)
    jax.block_until_ready(carry[0].wm_params)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    t_total = time.time()
    for chunk_idx in range(num_chunks):
        t_chunk = time.time()
        carry, chunk_metrics = jit_chunk(carry)
        jax.block_until_ready(chunk_metrics["loss/rec"])
        dt = time.time() - t_chunk
        env_steps_done = (chunk_idx + 1) * chunk_updates * cfg["num_envs"]
        rec = np.asarray(chunk_metrics["loss/rec"])
        nz = rec[rec != 0.0]
        rec_str = (f"  loss/rec {nz[0]:.3f}→{nz[-1]:.3f}" if nz.size else "")
        bacc = np.asarray(chunk_metrics.get("belief/acc", np.zeros(1)))
        bnz = bacc[bacc != 0.0]
        b_str = (f"  belief_acc {bnz[0]:.2f}→{bnz[-1]:.2f}" if bnz.size else "")
        print(f"chunk {chunk_idx + 1}/{num_chunks}: {dt:.1f}s "
              f"(~{env_steps_done} env steps){rec_str}{b_str}", flush=True)

    print(f"train complete in {time.time() - t_total:.1f}s", flush=True)
    _save_final_checkpoint(carry[0], run_dir, cfg["total_env_steps"])
    print(f"final checkpoint → {run_dir}/checkpoints/step_{cfg['total_env_steps']}", flush=True)

    if wandb_active:
        wandb.finish()


if __name__ == "__main__":
    main()
