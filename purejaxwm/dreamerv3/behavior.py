"""Imagination rollout, actor-critic losses, return computation, return normalization,
slow-critic EMA, and training state.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from purejaxwm.dreamerv3.distributions import TwoHotDist
from purejaxwm.dreamerv3.world_model import RSSM, State, imagine_scan


def unimix_logits(logits: jnp.ndarray, unimix: float = 0.01) -> jnp.ndarray:
    """Mix ``unimix`` fraction of uniform probability into softmax(logits).

    Returns re-derived logits = log(mixed_probs). This is a probability-space mix (not
    logit-space), matching Hafner's reference implementation.
    """
    if unimix <= 0:
        return logits
    probs = jax.nn.softmax(logits, axis=-1)
    num_classes = logits.shape[-1]
    uniform = jnp.ones_like(probs) / num_classes
    probs = (1.0 - unimix) * probs + unimix * uniform
    return jnp.log(probs + 1e-12)


def lambda_returns(
    rewards: jnp.ndarray,      # (T, B) reward at each imagined step
    values: jnp.ndarray,       # (T+1, B) critic values, values[-1] bootstraps
    continues: jnp.ndarray,    # (T, B) gamma * cont at each step (cont ∈ [0,1])
    lam: float = 0.95,
) -> jnp.ndarray:
    """Compute \lambda-returns of length T.

    Implementation:
      R_T = values[-1]
      R_t = rewards[t] + continues[t] * ((1-\lambda) * values[t+1] + \lambda * R_{t+1})
    """

    def body(next_ret, t_inp):
        r, cont, v_next = t_inp
        ret = r + cont * ((1 - lam) * v_next + lam * next_ret)
        return ret, ret

    _, rets = jax.lax.scan(
        body,
        values[-1],                          # initial "next_ret" = bootstrap
        (rewards, continues, values[1:]),    # (T, B) inputs
        reverse=True,
    )
    return rets  # (T, B)


def lambda_return_repl(
    last: jnp.ndarray,     # (T, B) episode boundary flags (is_terminal)
    term: jnp.ndarray,     # (T, B) terminal flags (is_terminal)
    rew: jnp.ndarray,      # (T, B) real rewards
    boot: jnp.ndarray,     # (T, B) bootstrap values from imagination
    disc: float,            # discount factor (gamma)
    lam: float,             # lambda parameter
) -> jnp.ndarray:
    """TD(\lambda) returns with episode boundary handling for replay data.

    Uses ``rew[1:]`` so the arrival reward at s_0 is skipped.  At episode
    boundaries (``last=True``), the lambda trace resets and bootstraps
    from ``boot``.  At terminal states (``term=True``), discount is zero.

    Returns (T-1, B).
    """
    live = (1 - term[1:].astype(jnp.float32)) * disc      # (T-1, B)
    cont = (1 - last[1:].astype(jnp.float32)) * lam       # (T-1, B)
    interm = rew[1:] + (1 - cont) * live * boot[1:]        # (T-1, B)

    def body(next_ret, inp):
        i, l, c = inp
        ret = i + l * c * next_ret
        return ret, ret

    _, rets = jax.lax.scan(
        body,
        boot[-1],                            # bootstrap from last timestep
        (interm, live, cont),                # (T-1, B) inputs
        reverse=True,
    )
    return rets  # (T-1, B)


class RetNorm(NamedTuple):
    """Percentile-based return normalizer with debiasing.

    Tracks EMAs of the 5th and 95th percentiles of returns. Used to scale advantages
    before they drive the actor loss. See Hafner 2023 §3.1.

    Debiasing corrects for the zero-initialization bias so the scale reflects the true
    return range after the very first update, rather than converging over ~100 steps.
    """
    low: jnp.ndarray    # ()  EMA of 5th percentile
    high: jnp.ndarray   # ()  EMA of 95th percentile
    count: jnp.ndarray  # ()  EMA counter for debiasing

    @classmethod
    def initial(cls) -> "RetNorm":
        return cls(low=jnp.array(0.0), high=jnp.array(0.0), count=jnp.array(0.0))

    def scale(self, rate: float = 0.01) -> jnp.ndarray:
        corr = 1.0 / jnp.maximum(rate, self.count)
        lo = self.low * corr
        hi = self.high * corr
        return jnp.maximum(hi - lo, 1.0)


class DreamerTrainState(NamedTuple):
    """Full training carry for one seed's training loop.

    All fields are pytree-compatible and may be vmapped over seeds when ``train`` is
    invoked as ``jax.vmap(train, in_axes=(0, None))(seed_keys, cfg)``.
    """
    wm_params: Any
    ac_params: Any
    slow_critic_params: Any
    opt_state: Any
    retnorm: RetNorm
    step: jnp.ndarray            # env-step counter
    train_step: jnp.ndarray      # gradient-step counter


class ImaginedTrajectory(NamedTuple):
    states: State                 # (T, B, ...) imagined states (each is one RSSM step forward)
    actions: jnp.ndarray          # (T, B, A) one-hot actions
    log_probs: jnp.ndarray        # (T, B) log π(a | s)
    entropies: jnp.ndarray        # (T, B) H(π(·|s))


def _align_current_states(init_state: State, next_states: State) -> State:
    """Build the actor/critic time grid aligned with the sampled actions.

    ``imagine_scan`` emits ``next_states[t] = s_{t+1}`` while the actor's log-probs at
    index ``t`` were computed from ``s_t``. For return targets and replay bootstrap, we
    therefore need the sequence ``(s_0, s_1, ..., s_{T-1})``, which is exactly
    ``init_state`` prepended to ``next_states[:-1]``.
    """
    return State(
        deter=jnp.concatenate([init_state.deter[None], next_states.deter[:-1]], axis=0),
        stoch=jnp.concatenate([init_state.stoch[None], next_states.stoch[:-1]], axis=0),
        logits=jnp.concatenate([init_state.logits[None], next_states.logits[:-1]], axis=0),
    )


def imagine_trajectory(
    rssm: RSSM,
    rssm_params,
    actor_head,             # MLPHead: apply(params, feat) → raw logits
    actor_params,
    init_state: State,      # (B, ...) starting state (no time axis)
    action_dim: int,
    horizon: int,
    rng: jax.Array,
    unimix: float = 0.01,
) -> ImaginedTrajectory:
    """Roll out ``horizon`` imagined steps using the actor head and RSSM prior."""
    def policy_fn(policy_params, state: State, sub):
        feat = state.features()
        logits = unimix_logits(actor_head.apply(policy_params, feat), unimix)
        probs = jax.nn.softmax(logits, axis=-1)
        action_idx = jax.random.categorical(sub, logits)
        action_oh = jax.nn.one_hot(action_idx, action_dim)
        action = jax.lax.stop_gradient(action_oh)
        log_prob = jnp.log(jnp.sum(action_oh * probs, axis=-1) + 1e-10)
        entropy = -(probs * jax.nn.log_softmax(logits, axis=-1)).sum(axis=-1)
        return action, (log_prob, entropy)

    states, actions, extras = imagine_scan(
        rssm, rssm_params, init_state, policy_fn, actor_params, horizon, rng
    )
    log_probs, entropies = extras
    return ImaginedTrajectory(states=states, actions=actions,
                              log_probs=log_probs, entropies=entropies)


class ACLossAux(NamedTuple):
    actor_loss: jnp.ndarray
    critic_loss: jnp.ndarray
    entropy: jnp.ndarray
    advantage_mean: jnp.ndarray
    advantage_std: jnp.ndarray
    return_mean: jnp.ndarray
    ret_low: jnp.ndarray
    ret_high: jnp.ndarray
    value_mean: jnp.ndarray
    slow_value_mean: jnp.ndarray
    reward_mean: jnp.ndarray
    scale: jnp.ndarray
    returns_start: jnp.ndarray


def imag_loss(
    ac_params,                       # {'actor': ..., 'critic': ...}
    slow_critic_params,              # slow critic (EMA) params
    rssm: RSSM,
    rssm_params,
    actor_head,                      # MLPHead: apply(params, feat) → raw logits
    critic_head,                     # MLPHead: apply(params, feat) → TwoHot logits
    init_state: State,               # (B*, ...) — caller should stop_gradient before
    reward_head_apply,               # fn(features) → (..., ) predicted reward (scalar mean)
    cont_head_apply,                 # fn(features) → (..., ) predicted cont logits
    retnorm: RetNorm,
    *,
    action_dim: int,
    horizon: int,
    gamma: float,
    gae_lambda: float,
    entropy_coef: float,
    slow_reg_coef: float,
    percentile_lo: float,
    percentile_hi: float,
    retnorm_rate: float = 0.01,
    contdisc: bool = True,
    slowtar: bool = True,
    unimix: float = 0.01,
    rng: jax.Array,
):
    """Compute the imagination-time actor and critic losses.

    Returns (actor_loss + critic_loss) as a scalar total, plus an ACLossAux struct of
    diagnostics, and an updated RetNorm.
    """
    rng, sub = jax.random.split(rng)
    traj = imagine_trajectory(
        rssm, rssm_params, actor_head, ac_params["actor"], init_state,
        action_dim, horizon, sub, unimix=unimix,
    )
    current_states = _align_current_states(init_state, traj.states)
    feats = current_states.features()                  # (T, B, F) for s_t
    bootstrap_feat = traj.states.features()[-1]       # (B, F) for s_T

    rewards_pred = reward_head_apply(feats)
    if hasattr(rewards_pred, "mean"):
        rewards = rewards_pred.mean()                # (T, B)
    else:
        rewards = rewards_pred
    cont_logits = cont_head_apply(feats)
    continues = jax.nn.sigmoid(cont_logits)          # (T, B) ∈ [0, 1]

    value_dist = TwoHotDist(critic_head.apply(ac_params["critic"], feats))
    values = value_dist.mean()                        # (T, B)

    slow_value_dist = TwoHotDist(critic_head.apply(slow_critic_params, feats))
    slow_values = jax.lax.stop_gradient(slow_value_dist.mean())

    bootstrap_value = TwoHotDist(
        critic_head.apply(ac_params["critic"], bootstrap_feat)
    ).mean()
    bootstrap_slow_value = jax.lax.stop_gradient(
        TwoHotDist(critic_head.apply(slow_critic_params, bootstrap_feat)).mean()
    )

    tarval = slow_values if slowtar else values
    bootstrap = bootstrap_slow_value if slowtar else bootstrap_value
    tarval_ext = jnp.concatenate([tarval, bootstrap[None]], axis=0)

    disc_lambda = gamma * continues                  # (T, B)
    returns = lambda_returns(rewards, tarval_ext, disc_lambda, lam=gae_lambda)  # (T, B)

    # update retnorm (EMA of 5th/95th percentile of returns)
    lo_now = jnp.percentile(returns, percentile_lo)
    hi_now = jnp.percentile(returns, percentile_hi)
    new_retnorm = RetNorm(
        low=(1 - retnorm_rate) * retnorm.low + retnorm_rate * lo_now,
        high=(1 - retnorm_rate) * retnorm.high + retnorm_rate * hi_now,
        count=(1 - retnorm_rate) * retnorm.count + retnorm_rate,
    )
    scale = new_retnorm.scale(rate=retnorm_rate)

    advantages = (returns - tarval) / scale
    advantages_sg = jax.lax.stop_gradient(advantages)

    disc_w = 1.0 if contdisc else gamma
    causal_weight = jax.lax.stop_gradient(
        jnp.cumprod(disc_w * continues, axis=0) / disc_w
    )
    pg_loss = -(traj.log_probs * advantages_sg * causal_weight).mean()
    entropy_bonus = (traj.entropies * causal_weight).mean()
    actor_loss = pg_loss - entropy_coef * entropy_bonus

    # critic loss: TwoHot cross-entropy on sg(returns), plus slow-target regularizer
    returns_sg = jax.lax.stop_gradient(returns)
    logp_returns = value_dist.log_prob(returns_sg)                         # (T, B)
    logp_slow = value_dist.log_prob(jax.lax.stop_gradient(slow_values))    # (T, B)
    critic_loss = -(logp_returns * causal_weight).mean() + \
                   slow_reg_coef * -(logp_slow * causal_weight).mean()

    aux = ACLossAux(
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        entropy=entropy_bonus,
        advantage_mean=advantages.mean(),
        advantage_std=advantages.std(),
        return_mean=returns.mean(),
        ret_low=new_retnorm.low,
        ret_high=new_retnorm.high,
        value_mean=values.mean(),
        slow_value_mean=slow_values.mean(),
        reward_mean=rewards.mean(),
        scale=scale,
        returns_start=jax.lax.stop_gradient(returns[0]),   # (B,)
    )
    total = actor_loss + critic_loss
    return total, (aux, new_retnorm)


class ReplLossAux(NamedTuple):
    repl_loss: jnp.ndarray
    target_mean: jnp.ndarray
    target_std: jnp.ndarray
    value_mean: jnp.ndarray
    slow_value_mean: jnp.ndarray


def repl_loss(
    ac_params,
    slow_critic_params,
    critic_head,                                     # MLPHead: apply(params, feat) → TwoHot logits
    replay_features_sg: jnp.ndarray,     # (T, B, F) stop-gradient'd posterior features
    replay_rewards: jnp.ndarray,         # (T, B) real rewards from replay buffer
    replay_is_terminal: jnp.ndarray,     # (T, B) terminal flags from replay buffer
    bootstrap_values_sg: jnp.ndarray,    # (T, B) imagination returns[0] per replay step
    *,
    gamma: float = 0.997,
    gae_lambda: float = 0.95,
    slow_reg_coef: float = 1.0,
):
    """Critic loss on real replay trajectories with lambda-return targets."""
    T, B = replay_rewards.shape

    feat_flat = replay_features_sg.reshape(T * B, -1)
    value_dist = TwoHotDist(critic_head.apply(ac_params["critic"], feat_flat))
    values = value_dist.mean().reshape(T, B)

    slow_dist = TwoHotDist(critic_head.apply(slow_critic_params, feat_flat))
    slow_values = jax.lax.stop_gradient(slow_dist.mean().reshape(T, B))

    rets = lambda_return_repl(
        last=replay_is_terminal,
        term=replay_is_terminal,
        rew=replay_rewards,
        boot=bootstrap_values_sg,
        disc=gamma,
        lam=gae_lambda,
    )  # (T-1, B)

    weight = (~replay_is_terminal[:T - 1]).astype(jnp.float32)  # (T-1, B)

    rets_padded = jnp.concatenate(
        [jax.lax.stop_gradient(rets), jnp.zeros_like(rets[:1])], axis=0,
    )  # (T, B)

    logp_returns = value_dist.log_prob(rets_padded.reshape(T * B)).reshape(T, B)
    logp_slow = value_dist.log_prob(
        jax.lax.stop_gradient(slow_values.reshape(T * B))
    ).reshape(T, B)

    critic_loss_per = (-logp_returns + slow_reg_coef * -logp_slow)[:T - 1]
    loss = (weight * critic_loss_per).mean()

    aux = ReplLossAux(
        repl_loss=loss,
        target_mean=rets.mean(),
        target_std=rets.std(),
        value_mean=values.mean(),
        slow_value_mean=slow_values.mean(),
    )
    return loss, aux


def slow_critic_update(slow_params, live_params, ema_rate: float = 0.02):
    return jax.tree_util.tree_map(
        lambda s, l: (1 - ema_rate) * s + ema_rate * l,
        slow_params, live_params,
    )
