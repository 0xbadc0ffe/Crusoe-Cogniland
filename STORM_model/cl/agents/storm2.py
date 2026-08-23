"""STORM v2 -- faithful port of the original STORM training scheme
(weipu-zhang/STORM, Zhang et al. 2023).

The first-pass STORM agent in this repo (cl/agents/storm.py) deviates from the
paper in a way that silently removes ALL temporal memory:

  * it calls the transformer on 1-token sequences with no KV cache, so the
    prior is Markov in (z_{t-1}, a_{t-1});
  * the policy features are the posterior z_t alone, and the posterior is a
    function of the CURRENT observation only -- the actor is purely reactive.

That is fine for reactive tasks (it reaches 100% on Navix-Empty-8x8) but makes
memory tasks (bridge_tunnel fork_wall) unsolvable in principle.

This agent restores the original semantics:

  * TRAINING: the transformer processes the whole replay sequence in parallel
    under a causal (+ episode-segment) mask -- token t = stem(z_t, a_t), the
    output at position t-1 is the history summary h_t used for the prior over
    z_t, the reward r_t and the continuation c_t (reward/continuation heads
    read h, as in the original, so they can be belief-dependent).
  * FEATURES: the actor-critic input is concat(z_t, h_t) -- z from the current
    observation, h from attention over up to `batch_length` past tokens.
  * ACTION SELECTION: a rolling window of the last `env_context` (z, a) tokens
    is kept per env; the transformer is re-run over the window each step
    (windows are tiny -- this is cheap and avoids KV-cache plumbing).
  * IMAGINATION: as in the original (ImagineContextLength / ImagineBatchLength),
    rollouts are primed with a context of real posterior latents before the
    policy takes over, so imagined trajectories inherit real history/belief.

Everything else (encoder/decoder, DistHead, MLP policy, LaProp, normalizers,
lambda-return actor-critic loss, replay buffer, trainer integration) is reused
from the existing infrastructure.
"""

from typing import Any, Dict, Tuple, Optional
from collections import defaultdict

import chex
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from cl.agents.base import ContinualAgent
from cl.agents.registry import register_agent
from cl.agents.commons import ReservoirReplayBuffer
from cl.agents.commons.distributions import OneHotDist
from cl.agents.commons.networks.mlp import MLPHead
from cl.agents.commons.normalizers import init_normalizer
from cl.agents.commons.optimizer import laprop
from cl.agents.policy.actor_critic import imag_loss
from cl.agents.policy.mlp import MLPPolicy
from cl.agents.state import (
    AgentState, PolicyParams, RuntimeState, WorldModelParams,
)
from cl.agents.utils import RatioTracker, sg
from cl.agents.world_models.dreamerv3.encoder import Encoder
from cl.agents.world_models.dreamerv3.decoder import Decoder
from cl.agents.world_models.storm.state import StormParams, StormTrainState
from cl.agents.world_models.storm.transformer import StochasticTransformer
from cl.agents.world_models.storm.world_model import DistHead


@chex.dataclass
class Storm2State:
    """Per-env rolling context window for action selection.

    z_ctx:   [B, W, K*C] flattened posterior latents of the last W steps
    a_ctx:   [B, W]      action indices taken at those steps
    ctx_len: [B]         number of valid tokens (0 right after reset)
    """
    z_ctx: jnp.ndarray
    a_ctx: jnp.ndarray
    ctx_len: jnp.ndarray


@register_agent('storm2')
class STORM2(ContinualAgent):
    """STORM with the original paper's memory semantics restored."""

    def __init__(self, config: OmegaConf, obs_space: Dict, act_space: int):
        super().__init__(config, obs_space, act_space)

        self.cfg = config.agent
        self.obs_space = obs_space
        self.obs_modalities = list(obs_space.keys())
        self.action_space = act_space
        self.num_actions = act_space

        num_envs = config.env.get('num_parallel_envs', 16)

        self.batch_size = int(self.cfg.get('batch_size', 16))
        self.batch_length = int(self.cfg.get('batch_length', 64))

        self.buffer = ReservoirReplayBuffer(
            capacity=int(self.cfg.replay.get('capacity', 1e6)),
            obs_shapes=obs_space,
            action_dim=act_space,
            batch_size=self.batch_size,
            batch_length=self.batch_length,
            num_envs=num_envs,
        )

        dyn = self.cfg.model.dyn
        self.stoch_dim = int(dyn.get('stoch_dim', 32))
        self.classes = int(dyn.get('classes', 32))
        self.unimix = float(dyn.get('unimix', 0.01))
        self.free_nats = float(dyn.get('free_nats', 1.0))
        self.stoch_flat = self.stoch_dim * self.classes

        tcfg = OmegaConf.to_container(dyn.transformer, resolve=True)
        self.feat_dim = int(tcfg['feat_dim'])
        self.transformer = StochasticTransformer(
            stoch_dim=self.stoch_flat,
            action_dim=act_space,
            feat_dim=self.feat_dim,
            num_layers=int(tcfg['num_layers']),
            num_heads=int(tcfg['num_heads']),
            max_length=int(tcfg['max_length']),
            dropout=float(tcfg.get('dropout', 0.1)),
        )
        self.post_head = DistHead(stoch_dim=self.stoch_dim, classes=self.classes,
                                  unimix=self.unimix)
        self.prior_head = DistHead(stoch_dim=self.stoch_dim, classes=self.classes,
                                   unimix=self.unimix)

        self.encoder = Encoder(**OmegaConf.to_container(self.cfg.model.enc.simple, resolve=True))
        self.decoder = Decoder(obs_shapes=obs_space,
                               **OmegaConf.to_container(self.cfg.model.dec.simple, resolve=True))
        self.reward_head = MLPHead(output_shape=(1,),
                                   **OmegaConf.to_container(self.cfg.model.rewhead, resolve=True))
        self.cont_head = MLPHead(output_shape=(1,),
                                 **OmegaConf.to_container(self.cfg.model.conthead, resolve=True))

        self.policy = MLPPolicy(
            action_space=act_space,
            actor_config=OmegaConf.to_container(self.cfg.model.actor, resolve=True),
            critic_config=OmegaConf.to_container(self.cfg.model.critic, resolve=True),
        )

        self.optimizer = laprop(**OmegaConf.to_container(self.cfg.model.opt, resolve=True))

        batch_steps = self.batch_size * self.batch_length
        self._train_ratio_tracker = RatioTracker(
            self.cfg.get('train_ratio', 32) / batch_steps
        )

        # Imagination recipe (original STORM: context 8, rollout 16).
        self.imag_context = int(self.cfg.model.get('imag_context', 8))
        self.imag_horizon = int(self.cfg.model.get('imag_length', 16))
        stride = int(self.cfg.model.get('imag_window_stride', 8))
        # window starts: token t uses stored_action[t+1], so s + imag_context <= L-1
        max_start = self.batch_length - 1 - self.imag_context
        self.imag_starts = tuple(range(0, max_start + 1, stride))

        # Env-time rolling context window (>= corridor memory requirement).
        self.env_context = int(self.cfg.model.get('env_context', 32))

        self.lambda_ = self.cfg.model.imag_loss.get('lam', 0.95)
        self.slow_critic_rate = self.cfg.model.slowvalue.get('rate', 0.02)
        self.grad_checkpoint = self.cfg.model.get('grad_checkpoint', True)
        self.loss_scales = OmegaConf.to_container(self.cfg.model.loss_scales, resolve=True)

        self._select_action_jit_compiled = {
            'train': jax.jit(self._select_action_jit, static_argnums=(5,)),
            'eval': jax.jit(self._select_action_jit, static_argnums=(5,)),
        }
        self._train_step_jit = jax.jit(self._train_step_core)

    # ── init ─────────────────────────────────────────────────────────────

    def initial_wm_state(self, batch_size: int) -> Storm2State:
        W = self.env_context
        return Storm2State(
            z_ctx=jnp.zeros((batch_size, W, self.stoch_flat), dtype=jnp.float32),
            a_ctx=jnp.zeros((batch_size, W), dtype=jnp.int32),
            ctx_len=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

    def init(self, rng: jax.random.PRNGKey) -> AgentState:
        rng, r_enc, r_dec, r_tr, r_post, r_prior, r_rew, r_cont, r_pol, r_buf, r_run = \
            jax.random.split(rng, 11)

        buffer_state = self.buffer.init(r_buf)

        dummy_obs = {}
        for key, shape in self.obs_space.items():
            dt = jnp.uint8 if len(shape) == 3 else jnp.float32
            dummy_obs[key] = jnp.zeros((1,) + tuple(shape), dtype=dt)

        enc_params = self.encoder.init(r_enc, dummy_obs)
        embed = self.encoder.apply(enc_params, dummy_obs)
        embed_dim = embed.shape[-1]

        post_params = self.post_head.init(r_post, jnp.zeros((1, embed_dim)))
        prior_params = self.prior_head.init(r_prior, jnp.zeros((1, self.feat_dim)))

        dummy_samples = jnp.zeros((1, 2, self.stoch_flat))
        dummy_actions = jnp.zeros((1, 2), dtype=jnp.int32)
        dummy_mask = jnp.tril(jnp.ones((2, 2), dtype=bool))[None]
        tr_params = self.transformer.init(
            r_tr, dummy_samples, dummy_actions, mask=dummy_mask, training=False)

        dec_params = self.decoder.init(r_dec, {
            'deter': jnp.zeros((1, self.stoch_flat)),
            'stoch': jnp.zeros((1, self.stoch_dim, self.classes)),
        })
        rew_params = self.reward_head.init(r_rew, jnp.zeros((1, self.feat_dim)), training=False)
        cont_params = self.cont_head.init(r_cont, jnp.zeros((1, self.feat_dim)), training=False)

        wm_params = WorldModelParams(
            encoder=enc_params, decoder=dec_params,
            dynamics={'transformer': tr_params, 'post_head': post_params,
                      'prior_head': prior_params},
            reward=rew_params, continuation=cont_params,
        )

        pol_feat_dim = self.stoch_flat + self.feat_dim
        policy_params_init = self.policy.init_params(r_pol, pol_feat_dim, self.action_space)

        normalizers = {
            'return': init_normalizer(**OmegaConf.to_container(self.cfg.model.retnorm, resolve=True)),
            'value': init_normalizer(**OmegaConf.to_container(self.cfg.model.valnorm, resolve=True)),
            'advantage': init_normalizer(**OmegaConf.to_container(self.cfg.model.advnorm, resolve=True)),
        }

        params = StormParams(
            wm=wm_params,
            policy=PolicyParams(actor=policy_params_init.actor,
                                critic=policy_params_init.critic,
                                slow_critic=None, normalizers={}),
        )
        train_state = StormTrainState.create(
            apply_fn=None, params=params, tx=self.optimizer,
            slow_critic=policy_params_init.slow_critic,
            normalizers=normalizers,
            slow_critic_rate=self.slow_critic_rate,
        )
        runtime = RuntimeState(
            buffer_state=buffer_state,
            wm_state=self.initial_wm_state(1),
            step=jnp.array(0, dtype=jnp.int32),
            train_steps=jnp.array(0, dtype=jnp.int32),
            rng=r_run,
        )
        return AgentState(train_state=train_state, runtime=runtime)

    # ── world-model helpers ──────────────────────────────────────────────

    def _encode(self, wm_params, obs_flat):
        return self.encoder.apply(wm_params.encoder, obs_flat)

    def _post_logits(self, wm_params, embed):
        return self.post_head.apply(wm_params.dynamics['post_head'], embed)

    def _prior_logits(self, wm_params, h):
        return self.prior_head.apply(wm_params.dynamics['prior_head'], h)

    def _transformer_fwd(self, wm_params, samples, actions, mask):
        return self.transformer.apply(
            wm_params.dynamics['transformer'], samples, actions,
            mask=mask, training=False)

    def _sample_z(self, logits, rng=None):
        """Straight-through sample (or mode if rng is None) from K*C logits."""
        dist = OneHotDist(logits.reshape(*logits.shape[:-1], self.stoch_dim, self.classes),
                          unimix=self.unimix)
        z = dist.sample(seed=rng) if rng is not None else dist.mode()
        z = z.astype(jnp.float32)
        return z, z.reshape(*z.shape[:-2], self.stoch_flat)

    # ── loss ─────────────────────────────────────────────────────────────

    def _loss_fn(self, params: StormParams, slow_critic, normalizers, batch, rng):
        wm_params = params.wm
        policy_params = PolicyParams(
            actor=params.policy.actor, critic=params.policy.critic,
            slow_critic=slow_critic, normalizers=normalizers,
        )
        rng_post, rng_imag = jax.random.split(rng)

        obs_dict = {}
        for key in batch.keys():
            if key.startswith('obs_'):
                obs_dict[key[4:]] = batch[key]

        actions_oh = batch['action']            # [B, L, A] stored one-hot (PREV action)
        rewards = batch['reward']               # [B, L]
        is_first = batch['is_first']            # [B, L]
        is_terminal = batch['is_terminal']      # [B, L]

        B, L = rewards.shape
        losses = {}
        metrics = {}

        # 1) posterior latents for every position
        obs_flat = {}
        for key, value in obs_dict.items():
            v = value.reshape(B * L, *value.shape[2:])
            if key in self.obs_space and len(self.obs_space[key]) == 3 and v.dtype == jnp.uint8:
                v = v.astype(jnp.float32) / 255.0
            obs_flat[key] = v
        embed = self._encode(wm_params, obs_flat)                      # [B*L, E]
        post_logits = self._post_logits(wm_params, embed)              # [B*L, K*C]
        z, z_flat = self._sample_z(post_logits, rng_post)              # [B*L, K, C], [B*L, K*C]
        post_logits = post_logits.reshape(B, L, -1)
        z = z.reshape(B, L, self.stoch_dim, self.classes)
        z_flat = z_flat.reshape(B, L, self.stoch_flat)

        # 2) parallel causal transformer over tokens t = (z_t, a_t) where a_t is
        #    the action TAKEN at t = stored action at t+1.
        action_idx = jnp.argmax(actions_oh, axis=-1).astype(jnp.int32)  # [B, L]
        tok_z = z_flat[:, :-1]                                          # [B, L-1, S]
        tok_a = action_idx[:, 1:]                                       # [B, L-1]

        # causal + same-episode-segment mask: query j (predicting t=j+1) may
        # attend token k iff k<=j and token k is in the same episode as t=j+1.
        seg = jnp.cumsum(is_first.astype(jnp.int32), axis=1)            # [B, L]
        Lm1 = L - 1
        causal = jnp.tril(jnp.ones((Lm1, Lm1), dtype=bool))             # [L-1, L-1]
        same_seg = seg[:, None, 1:] == seg[:, :Lm1, None]               # [B, k, j]
        same_seg = jnp.swapaxes(same_seg, 1, 2)                         # [B, j, k]
        mask = causal[None] & same_seg                                  # [B, L-1, L-1]

        dist_feat = self._transformer_fwd(wm_params, tok_z, tok_a, mask)  # [B, L-1, D]
        # h_t for t=1..L-1; invalid where t is an episode start (no history)
        h_valid = ~is_first[:, 1:]                                      # [B, L-1]
        dist_feat = jnp.where(h_valid[..., None], dist_feat, 0.0)

        # 3) prior / KL (positions t=1..L-1, masked at episode starts)
        prior_logits = self._prior_logits(wm_params, dist_feat)         # [B, L-1, K*C]
        post_t = post_logits[:, 1:].reshape(B, Lm1, self.stoch_dim, self.classes)
        prior_t = prior_logits.reshape(B, Lm1, self.stoch_dim, self.classes)

        post_dist_sg = OneHotDist(sg(post_t), unimix=self.unimix)
        prior_dist = OneHotDist(prior_t, unimix=self.unimix)
        post_dist = OneHotDist(post_t, unimix=self.unimix)
        prior_dist_sg = OneHotDist(sg(prior_t), unimix=self.unimix)
        kl_dyn = post_dist_sg.kl_divergence(prior_dist)                 # [B, L-1]
        kl_rep = post_dist.kl_divergence(prior_dist_sg)
        if self.free_nats > 0:
            kl_dyn = jnp.maximum(kl_dyn, self.free_nats)
            kl_rep = jnp.maximum(kl_rep, self.free_nats)
        wmask = h_valid.astype(jnp.float32)
        denom = jnp.maximum(wmask.sum(), 1.0)
        losses['dyn'] = (kl_dyn * wmask).sum() / denom
        losses['rep'] = (kl_rep * wmask).sum() / denom

        # 4) reconstruction from z (all positions)
        recons = self.decoder.apply(wm_params.decoder, {
            'deter': z_flat.reshape(B * L, -1),
            'stoch': z.reshape(B * L, self.stoch_dim, self.classes),
        })
        rec_terms = []
        for key in obs_dict.keys():
            if key in recons:
                rec_terms.append(-recons[key].log_prob(sg(obs_flat[key])))
        rec = sum(rec_terms) if rec_terms else jnp.zeros((B * L,))
        losses['rec'] = rec.mean()

        # 5) reward / continuation from h (positions t=1..L-1)
        h_flat = dist_feat.reshape(B * Lm1, -1)
        rew_dist = self.reward_head.apply(wm_params.reward, h_flat, training=False)
        rew_nll = rew_dist.loss(sg(rewards[:, 1:].reshape(-1))).reshape(B, Lm1)
        losses['rew'] = (rew_nll * wmask).sum() / denom

        cont_target = 1.0 - is_terminal[:, 1:].astype(jnp.float32)
        if self.cfg.model.get('contdisc', True):
            cont_target = cont_target * (1 - 1 / self.cfg.model.get('horizon', 333))
        cont_dist = self.cont_head.apply(wm_params.continuation, h_flat, training=False)
        cont_nll = -cont_dist.log_prob(sg(cont_target.reshape(-1))).reshape(B, Lm1)
        losses['con'] = (cont_nll * wmask).sum() / denom

        # 6) imagination: context-primed rollouts (original STORM recipe)
        C, H = self.imag_context, self.imag_horizon
        starts = self.imag_starts
        Wn = len(starts)
        z_ctx = jnp.stack([sg(z_flat[:, s:s + C]) for s in starts], axis=1)      # [B, Wn, C, S]
        a_ctx = jnp.stack([tok_a[:, s:s + C] for s in starts], axis=1)           # [B, Wn, C]
        N = B * Wn
        buf_z = jnp.zeros((N, C + H + 1, self.stoch_flat))
        buf_a = jnp.zeros((N, C + H + 1), dtype=jnp.int32)
        buf_z = buf_z.at[:, :C].set(z_ctx.reshape(N, C, self.stoch_flat))
        buf_a = buf_a.at[:, :C].set(a_ctx.reshape(N, C))
        causal_imag = jnp.tril(jnp.ones((C + H + 1, C + H + 1), dtype=bool))[None]

        def imag_step(carry, inp):
            bz, ba = carry
            i, rng_i = inp
            r1, r2 = jax.random.split(rng_i)
            feats_all = self._transformer_fwd(wm_params, bz, ba, causal_imag)
            h = sg(feats_all[:, C - 1 + i])                                     # [N, D]
            pl = self._prior_logits(wm_params, h)
            _, zf = self._sample_z(sg(pl), r1)                                  # [N, S]
            feat = jnp.concatenate([zf, h], axis=-1)
            a_dist = self.policy.apply_actor(policy_params.actor, feat, training=True)
            a_idx = a_dist.sample(seed=r2)
            a_oh = jax.nn.one_hot(a_idx, self.action_space)
            r_hat = self.reward_head.apply(wm_params.reward, h, training=False).mode()
            c_hat = self.cont_head.apply(wm_params.continuation, h, training=False).prob(1.0)
            bz = jax.lax.dynamic_update_slice(bz, zf[:, None, :], (0, C + i, 0))
            ba = jax.lax.dynamic_update_slice(ba, a_idx[:, None].astype(jnp.int32), (0, C + i))
            outs = (feat, a_oh, sg(r_hat), sg(c_hat))
            return (bz, ba), outs

        rngs = jax.random.split(rng_imag, H + 1)
        idxs = jnp.arange(H + 1)
        step_fn = jax.checkpoint(imag_step) if self.grad_checkpoint else imag_step
        _, (im_feat, im_act, im_rew, im_cont) = jax.lax.scan(
            step_fn, (buf_z, buf_a), (idxs, rngs))
        # shapes: [H+1, N, ...] -- exactly what imag_loss expects.

        imag_config = {
            'horizon': self.cfg.model.get('horizon', 333),
            'lambda': self.lambda_,
            'entropy_coef': self.cfg.model.imag_loss.get('actent', 3e-4),
            'slow_reg': self.cfg.model.imag_loss.get('slowreg', 1.0),
            'slowtar': self.cfg.model.imag_loss.get('slowtar', False),
            'contdisc': self.cfg.model.get('contdisc', True),
            'update_normalizers': True,
        }
        actor_loss, critic_loss, imag_outputs, imag_metrics = imag_loss(
            policy=self.policy, policy_params=policy_params,
            features=im_feat, actions=im_act, rewards=im_rew,
            continuations=im_cont, config=imag_config,
        )
        losses['policy'] = actor_loss
        losses['value'] = critic_loss

        total_loss = sum(self.loss_scales.get(name, 1.0) * jnp.asarray(l).mean()
                         for name, l in losses.items())

        for name, l in losses.items():
            metrics[f'loss/{name}'] = jnp.asarray(l).mean()
        metrics.update(imag_metrics)

        final_normalizers = {
            'return': imag_outputs['retnorm_state'],
            'value': imag_outputs['valnorm_state'],
            'advantage': imag_outputs['advnorm_state'],
        }
        return total_loss, {'losses': losses, 'normalizers': final_normalizers,
                            'metrics': metrics}

    # ── train step ───────────────────────────────────────────────────────

    def _train_step_core(self, train_state, batch, rng):
        (total_loss, aux), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(
            train_state.params, train_state.slow_critic, train_state.normalizers,
            batch, rng)
        new_train_state = train_state.apply_gradients(grads=grads)
        new_train_state = new_train_state.replace(normalizers=aux['normalizers'])
        metrics = {'total_loss': total_loss, **aux.get('metrics', {})}
        return new_train_state, metrics

    def train_step(self, agent_state, batch):
        train_state = agent_state.train_state
        runtime = agent_state.runtime
        rng, rng_train = jax.random.split(runtime.rng)
        new_train_state, metrics = self._train_step_jit(train_state, batch, rng_train)
        new_runtime = RuntimeState(
            buffer_state=runtime.buffer_state, wm_state=runtime.wm_state,
            step=runtime.step, train_steps=runtime.train_steps + 1, rng=rng,
        )
        return AgentState(train_state=new_train_state, runtime=new_runtime), metrics

    # ── action selection ─────────────────────────────────────────────────

    def _select_action_jit(self, params, wm_state, obs_dict, prev_action,
                           is_first, training, rng):
        wm_params = params.wm
        batch_size = next(iter(obs_dict.values())).shape[0]
        W = self.env_context

        # reset context where a new episode starts
        m = is_first
        wm_state = Storm2State(
            z_ctx=jnp.where(m[:, None, None], 0.0, wm_state.z_ctx),
            a_ctx=jnp.where(m[:, None], 0, wm_state.a_ctx),
            ctx_len=jnp.where(m, 0, wm_state.ctx_len),
        )

        obs_in = {}
        for key, v in obs_dict.items():
            if key in self.obs_space and len(self.obs_space[key]) == 3 and v.dtype == jnp.uint8:
                v = v.astype(jnp.float32) / 255.0
            obs_in[key] = v
        embed = self._encode(wm_params, obs_in)
        post_logits = self._post_logits(wm_params, embed)
        rng, r_z, r_a = jax.random.split(rng, 3)
        _, z_flat = self._sample_z(post_logits, r_z if training else None)

        # h from the rolling window: query j attends token k iff k<=j and k<len
        ctx_len = wm_state.ctx_len                                       # [B]
        causal = jnp.tril(jnp.ones((W, W), dtype=bool))                  # [W, W]
        key_valid = jnp.arange(W)[None, None, :] < ctx_len[:, None, None]  # [B, 1, W]
        mask = causal[None] & key_valid                                  # [B, W, W]
        feats = self._transformer_fwd(wm_params, wm_state.z_ctx, wm_state.a_ctx, mask)
        last_idx = jnp.maximum(ctx_len - 1, 0)
        h = jnp.take_along_axis(feats, last_idx[:, None, None].repeat(feats.shape[-1], -1),
                                axis=1)[:, 0]                            # [B, D]
        h = jnp.where((ctx_len > 0)[:, None], h, 0.0)

        feat = jnp.concatenate([z_flat, h], axis=-1)
        a_dist = self.policy.apply_actor(params.policy.actor, feat, training=training)
        action_idx = a_dist.sample(seed=r_a) if training else a_dist.mode()
        action_idx = jnp.reshape(action_idx, (batch_size,)).astype(jnp.int32)

        # append (z_t, a_t): shift window left when full
        full = ctx_len >= W
        z_shift = jnp.where(full[:, None, None], jnp.roll(wm_state.z_ctx, -1, axis=1),
                            wm_state.z_ctx)
        a_shift = jnp.where(full[:, None], jnp.roll(wm_state.a_ctx, -1, axis=1),
                            wm_state.a_ctx)
        write_idx = jnp.where(full, W - 1, ctx_len)                      # [B]
        onehot_w = jax.nn.one_hot(write_idx, W)                          # [B, W]
        z_new = z_shift * (1 - onehot_w[..., None]) + z_flat[:, None, :] * onehot_w[..., None]
        a_new = jnp.where(onehot_w.astype(bool), action_idx[:, None], a_shift)
        new_state = Storm2State(
            z_ctx=z_new, a_ctx=a_new,
            ctx_len=jnp.minimum(ctx_len + 1, W),
        )
        return action_idx, new_state, rng

    def act(self, agent_state, obs_dict, prev_action, is_first, training=False):
        params = agent_state.train_state.params
        runtime = agent_state.runtime
        wm_state = runtime.wm_state
        batch_size = next(iter(obs_dict.values())).shape[0]
        if wm_state is None or jax.tree.leaves(wm_state)[0].shape[0] != batch_size:
            wm_state = self.initial_wm_state(batch_size)
        mode = 'train' if training else 'eval'
        action_idx, new_wm_state, new_rng = self._select_action_jit_compiled[mode](
            params, wm_state, obs_dict, prev_action, is_first, training, runtime.rng)
        new_runtime = RuntimeState(
            buffer_state=runtime.buffer_state, wm_state=new_wm_state,
            step=runtime.step, train_steps=runtime.train_steps, rng=new_rng,
        )
        return action_idx, AgentState(train_state=agent_state.train_state,
                                      runtime=new_runtime)

    def select_action(self, state, obs, rng, is_first=None, prev_action=None,
                      training=False):
        batch_size = next(iter(obs.values())).shape[0]
        if prev_action is None:
            prev_action = jnp.zeros((batch_size, self.action_space), dtype=jnp.float32)
        if is_first is None:
            is_first = jnp.zeros(batch_size, dtype=bool)
        else:
            is_first = jnp.squeeze(is_first)
        return self.act(state, obs, prev_action, is_first, training=training)

    # ── env loops (same structure as cl/agents/storm.py) ─────────────────

    def train(self, state, env, rng, num_train_frames, progress_bar=None,
              checkpoint_callback=None):
        metrics = defaultdict(list)
        metrics['episode_info'] = {
            'returned_episode_returns': [], 'returned_episode_lengths': [],
            'returned_episode': [], 'timestep': [],
        }
        frames_collected = 0

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, env.num_envs)
        env_state = env.reset(reset_rngs)
        prev_actions = jnp.zeros((env.num_envs, self.action_space))

        while frames_collected < num_train_frames:
            obs = env_state.env_state.observation
            reward = env_state.env_state.reward
            done = env_state.env_state.is_done()
            terminal = env_state.env_state.is_termination()
            is_first_flags = (env_state.env_state.t == 0) & (~done)
            reward = jnp.where(is_first_flags, 0.0, reward)

            replay_transition = {}
            for modality in self.obs_space.keys():
                obs_data = obs[modality]
                if len(self.obs_space[modality]) == 3 and obs_data.dtype != jnp.uint8:
                    obs_data = (obs_data * 255).astype(jnp.uint8)
                replay_transition[f'obs_{modality}'] = obs_data
            masked_prev_actions = jnp.where(
                is_first_flags[..., None], jnp.zeros_like(prev_actions), prev_actions)
            replay_transition.update({
                'action': masked_prev_actions, 'reward': reward,
                'is_first': is_first_flags, 'is_last': done, 'is_terminal': terminal,
            })
            new_buffer_state = self.buffer.add_batch(state.runtime.buffer_state,
                                                     replay_transition)
            state = AgentState(
                train_state=state.train_state,
                runtime=RuntimeState(
                    buffer_state=new_buffer_state, wm_state=state.runtime.wm_state,
                    step=state.runtime.step + env.num_envs,
                    train_steps=state.runtime.train_steps, rng=state.runtime.rng,
                ),
            )
            frames_collected += env.num_envs
            if progress_bar is not None:
                progress_bar.update(env.num_envs)

            rng, action_rng = jax.random.split(rng)
            action_indices, state = self.select_action(
                state, obs, action_rng, is_first=is_first_flags,
                prev_action=prev_actions, training=True)
            actions_onehot = jax.nn.one_hot(action_indices, self.action_space)
            env_state = env.step(env_state, action_indices)

            done_next = env_state.env_state.is_done()
            if jnp.any(done_next):
                for idx in jnp.where(done_next)[0]:
                    metrics['episode_info']['returned_episode_returns'].append(
                        float(env_state.returned_episode_returns[idx]))
                    metrics['episode_info']['returned_episode_lengths'].append(
                        int(env_state.returned_episode_lengths[idx]))
                    metrics['episode_info']['returned_episode'].append(True)
                    metrics['episode_info']['timestep'].append(int(env_state.timestep[idx]))
            prev_actions = actions_onehot

            buf_stats = self.buffer.stats(state.runtime.buffer_state)
            if state.runtime.step >= self.cfg.get('pretrain', 100) and \
               buf_stats.get('valid_timesteps', 0) >= self.batch_length:
                num_updates = self._train_ratio_tracker(state.runtime.step)
                for _ in range(num_updates):
                    rng, sample_rng = jax.random.split(rng)
                    batch = self.buffer.sample(state.runtime.buffer_state, sample_rng)
                    state, step_metrics = self.train_step(state, batch)
                    for key, value in step_metrics.items():
                        if key != 'episode_info':
                            metrics[key].append(float(value))
                    if checkpoint_callback is not None:
                        checkpoint_callback.on_train_step_end(
                            agent_state=state, step=int(state.runtime.train_steps),
                            metrics=None)

        state = AgentState(
            train_state=state.train_state,
            runtime=RuntimeState(
                buffer_state=state.runtime.buffer_state,
                wm_state=state.runtime.wm_state,
                step=state.runtime.step, train_steps=state.runtime.train_steps,
                rng=rng,
            ),
        )
        buf_stats = self.buffer.stats(state.runtime.buffer_state)
        metrics['buffer_size'] = buf_stats['size']
        metrics['buffer_total_steps'] = buf_stats['total_steps']

        metrics_aggregated = {}
        for key, value in metrics.items():
            if key == 'episode_info':
                continue
            elif isinstance(value, list):
                if len(value) > 0:
                    metrics_aggregated[key] = float(np.mean(value))
            else:
                metrics_aggregated[key] = value
        metrics_aggregated['episode_info'] = self._format_episode_info(
            metrics['episode_info'])
        return state, metrics_aggregated

    def evaluate(self, state, env, rng, num_eval_frames, progress_bar=None):
        metrics = {
            'episode_info': {
                'returned_episode_returns': [], 'returned_episode_lengths': [],
                'returned_episode': [], 'timestep': [],
            },
            'frames': 0,
        }
        frames_evaluated = 0
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, env.num_envs)
        env_state = env.reset(reset_rngs)
        prev_actions = jnp.zeros((env.num_envs, self.action_space))

        step_count = 0
        PROGRESS_UPDATE_INTERVAL = 100
        while frames_evaluated < num_eval_frames:
            obs = env_state.env_state.observation
            done = env_state.env_state.is_done()
            is_first_flags = (env_state.env_state.t == 0) & (~done)
            rng, action_rng = jax.random.split(rng)
            action_indices, state = self.select_action(
                state, obs, action_rng, is_first=is_first_flags,
                prev_action=prev_actions, training=False)
            actions_onehot = jax.nn.one_hot(action_indices, self.action_space)
            env_state = env.step(env_state, action_indices)
            frames_evaluated += env.num_envs
            step_count += 1
            if progress_bar is not None and step_count % PROGRESS_UPDATE_INTERVAL == 0:
                progress_bar.n = frames_evaluated
                progress_bar.refresh()
            done_next = env_state.env_state.is_done()
            if jnp.any(done_next):
                for idx in jnp.where(done_next)[0]:
                    metrics['episode_info']['returned_episode_returns'].append(
                        float(env_state.returned_episode_returns[idx]))
                    metrics['episode_info']['returned_episode_lengths'].append(
                        int(env_state.returned_episode_lengths[idx]))
                    metrics['episode_info']['returned_episode'].append(True)
                    metrics['episode_info']['timestep'].append(int(env_state.timestep[idx]))
            prev_actions = actions_onehot

        if progress_bar is not None:
            progress_bar.n = frames_evaluated
            progress_bar.refresh()
        metrics['frames'] = frames_evaluated
        metrics['episode_info'] = self._format_episode_info(metrics['episode_info'])
        return metrics

    @staticmethod
    def _format_episode_info(episode_info):
        if len(episode_info['returned_episode_returns']) > 0:
            n = len(episode_info['returned_episode_returns'])
            return {
                'returned_episode_returns': np.array(
                    episode_info['returned_episode_returns']).reshape((1, n, 1)),
                'returned_episode_lengths': np.array(
                    episode_info['returned_episode_lengths']).reshape((1, n, 1)),
                'returned_episode': np.array(
                    episode_info['returned_episode']).reshape((1, n, 1)),
                'timestep': np.array(episode_info['timestep']).reshape((1, n, 1)),
            }
        return None

    # ── checkpointing / reset ────────────────────────────────────────────

    def state_from_checkpoint(self, checkpoint_data, runtime_state):
        train_state_dict = checkpoint_data['train_state']
        params_dict = train_state_dict['params']
        wm_params = WorldModelParams(**params_dict['wm'])
        policy_params = PolicyParams(**params_dict['policy'])
        params = StormParams(wm=wm_params, policy=policy_params)
        train_state = StormTrainState(
            step=train_state_dict['step'], apply_fn=None, params=params,
            tx=self.optimizer, opt_state=train_state_dict['opt_state'],
            slow_critic=train_state_dict['slow_critic'],
            normalizers=train_state_dict['normalizers'],
            slow_critic_rate=train_state_dict.get('slow_critic_rate', 0.02),
        )
        return AgentState(train_state=train_state, runtime=runtime_state)

    def reset(self, state, rng):
        new_train_state = StormTrainState.create(
            apply_fn=None, params=state.train_state.params, tx=self.optimizer,
            slow_critic=state.train_state.slow_critic,
            normalizers=state.train_state.normalizers,
            slow_critic_rate=self.slow_critic_rate,
        )
        return AgentState(
            train_state=new_train_state,
            runtime=RuntimeState(
                buffer_state=state.runtime.buffer_state, wm_state=None,
                step=state.runtime.step, train_steps=state.runtime.train_steps,
                rng=rng,
            ),
        )


__all__ = ['STORM2']
