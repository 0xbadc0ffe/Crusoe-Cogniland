"""Imagination-based actor-critic for world model agents.

This module provides actor-critic training on top of world models. The key idea is
to use the world model to imagine trajectories, then train the actor and critic
on these imagined experiences rather than real environment interactions.

Key functions:
    - imagine_trajectory(): Roll out imagined trajectories using the world model
    - imag_loss(): Compute actor-critic loss on imagined trajectories
    - repl_loss(): Compute critic loss on replay buffer data (optional)

Design principles:
    1. World-model agnostic: Works with any WorldModel implementation
    2. Functional: Pure functions that don't maintain state
    3. Composable: Can be used by different agents (DreamerV3, STORM, etc.)
    4. JIT-friendly: Designed for efficient compilation

Usage pattern:
    # 1. Imagine trajectories from the world model
    imag_states, imag_actions, imag_rewards, imag_conts = imagine_trajectory(
        wm=world_model,
        wm_params=state.wm_params,
        initial_state=wm_state,
        policy_fn=actor_fn,
        horizon=15,
        rng=rng,
    )

    # 2. Compute actor-critic loss on imagined data
    actor_loss, critic_loss, metrics = imag_loss(
        actor_params=state.policy_params.actor,
        critic_params=state.policy_params.critic,
        features=imag_features,
        actions=imag_actions,
        rewards=imag_rewards,
        continuations=imag_conts,
        config=config,
    )

    # 3. Update actor and critic
    grads = jax.grad(lambda p: actor_loss + critic_loss)(params)
    new_params = optimizer.apply_updates(params, grads)
"""

from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from cogniland.agents.commons.returns import lambda_return
from cogniland.agents.commons.normalizers import update_normalizer, get_normalizer_stats
from cogniland.agents.world_models.base import WorldModel, WorldModelParams, WorldModelState
from cogniland.agents.policy.base import Policy
from cogniland.agents.utils import sg

# Type alias for policy functions
# Policy takes (features, rng) and returns (action, new_rng)
# This allows proper RNG threading and supports stochastic policies
PolicyFn = Callable[[jnp.ndarray, jax.random.PRNGKey], Tuple[jnp.ndarray, jax.random.PRNGKey]]


def imagine_trajectory(
    wm: WorldModel,
    wm_params: WorldModelParams,
    initial_state: WorldModelState,
    policy_fn: PolicyFn,
    horizon: int,
    rng: jax.random.PRNGKey,
    grad_checkpoint: bool = True,
    ac_grads: bool = False,
) -> Tuple[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Roll out imagined trajectories using the world model.

    IMPORTANT: This function is designed to be world-model agnostic! It should work with
    any WorldModel implementation (RSSM, Transformer, etc.) by only calling
    the WorldModel protocol methods (imagine, get_feat, predict_reward, predict_continuation).

    The function uses the world model's dynamics to simulate future trajectories
    starting from the given initial state. At each step:
        1. Extract features from current state
        2. Sample action from policy (with RNG threading)
        3. Predict next state using world model
        4. Predict reward and continuation

    Args:
        wm: World model instance (any implementation of WorldModel protocol)
        wm_params: World model parameters
        initial_state: Initial world model state [B, ...]
        policy_fn: Policy function with signature:
                  (features: [B, F], rng: PRNGKey) -> (action: [B, A], new_rng: PRNGKey)
                  This allows the policy to handle stochastic sampling and proper RNG threading.
        horizon: Number of steps to imagine
        rng: Random key for action sampling and world model stochasticity
        grad_checkpoint: Whether to use gradient checkpointing (default True)
        ac_grads: Whether to allow actor-critic gradients to flow into world model (default False)

    Returns:
        imag_features: Imagined features [H+1, B, F]
                      Length H+1 because we prepend initial timestep (1 real + H imagined)
        imag_actions: Imagined actions [H+1, B, A]
        imag_rewards: Imagined rewards [H+1, B]
        imag_conts: Imagined continuations (1 - done) [H+1, B]

    Example usage:
        # Create a policy wrapper that matches PolicyFn signature
        def policy_fn(feat, rng):
            logits = actor.apply(actor_params, feat)
            rng, sample_rng = jax.random.split(rng)
            action = sample_from_logits(logits, sample_rng)
            return action, rng

        # Imagine trajectories
        states, actions, rewards, conts = imagine_trajectory(
            wm, wm_params, initial_state, policy_fn, horizon=15, rng=rng
        )
    """
    def scan_fn(carry, rng_t):
        """Single imagination step."""
        state, policy_rng = carry

        # Extract features using WorldModel protocol
        feat = wm.get_feat(wm_params, state)

        # Stop gradient on features unless ac_grads is True
        # This prevents actor-critic from "hacking" the world model
        feat = feat if ac_grads else sg(feat)

        # Sample action from policy
        action, policy_rng = policy_fn(feat, policy_rng)

        # Stop gradient on action (policy learns from values, not WM gradients)
        action = sg(action)

        # Predict next state using world model
        next_state, _ = wm.imagine(
            wm_params=wm_params,
            state=state,
            action=action,
            training=False,
            rng=rng_t,
        )

        # Predict reward and continuation from current features
        reward_dist = wm.predict_reward(wm_params, feat)
        reward = reward_dist.mode()  # Use mode for both MSEDist and TwoHotDist

        cont_dist = wm.predict_continuation(wm_params, feat)
        continuation = cont_dist.prob(1.0)  # P(continue) = P(not done)

        # Collect outputs (feat, action, reward, continuation)
        outputs = (feat, action, reward, continuation)

        return (next_state, policy_rng), outputs

    # Generate RNGs for each imagination step
    rngs = jax.random.split(rng, horizon + 1)
    policy_rng = rngs[0]
    step_rngs = rngs[1:]

    # Run scan to generate H imagined steps with optional gradient checkpointing
    # Checkpointing reduces VRAM usage at the cost of recomputing during backprop
    if grad_checkpoint:
        scan_fn_checkpointed = jax.checkpoint(scan_fn)
        _, trajectory_data = jax.lax.scan(
            scan_fn_checkpointed,
            (initial_state, policy_rng),
            step_rngs,
        )
    else:
        _, trajectory_data = jax.lax.scan(
            scan_fn,
            (initial_state, policy_rng),
            step_rngs,
        )

    # Unpack trajectory data (H imagined steps)
    imag_features, imag_actions, imag_rewards, imag_conts = trajectory_data

    # === CRITICAL: Prepend initial timestep (following DreamerV3) ===
    # Compute predictions on initial_state features (the "real" timestep)
    # This gives us H+1 total timesteps (1 real + H imagined)

    # Extract features from initial state
    batch_size = jax.tree_util.tree_leaves(initial_state)[0].shape[0]
    real_feat = wm.get_feat(wm_params, initial_state)

    # Stop gradient on real features unless ac_grads is True
    real_feat = real_feat if ac_grads else sg(real_feat)

    # Sample action from policy on real features
    real_action, _ = policy_fn(real_feat, policy_rng)

    # Predict reward and continuation from real features
    real_reward_dist = wm.predict_reward(wm_params, real_feat)
    real_reward = real_reward_dist.mode()

    real_cont_dist = wm.predict_continuation(wm_params, real_feat)
    real_continuation = real_cont_dist.prob(1.0)

    # Transpose imagined data from [H, B, ...] to [B, H, ...]
    imag_features_t = jnp.transpose(imag_features, (1, 0, 2))  # [B, H, F]
    imag_actions_t = jnp.transpose(imag_actions, (1, 0, 2))    # [B, H, A]
    imag_rewards_t = jnp.transpose(imag_rewards, (1, 0))        # [B, H]
    imag_conts_t = jnp.transpose(imag_conts, (1, 0))            # [B, H]

    # Prepend real timestep: [B, 1, ...] + [B, H, ...] = [B, H+1, ...]
    features_full = jnp.concatenate([real_feat[:, None, :], imag_features_t], axis=1)
    actions_full = jnp.concatenate([real_action[:, None, :], imag_actions_t], axis=1)
    rewards_full = jnp.concatenate([real_reward[:, None], imag_rewards_t], axis=1)
    conts_full = jnp.concatenate([real_continuation[:, None], imag_conts_t], axis=1)

    # Transpose back to [H+1, B, ...] format
    features_full = jnp.transpose(features_full, (1, 0, 2))  # [H+1, B, F]
    actions_full = jnp.transpose(actions_full, (1, 0, 2))    # [H+1, B, A]
    rewards_full = jnp.transpose(rewards_full, (1, 0))        # [H+1, B]
    conts_full = jnp.transpose(conts_full, (1, 0))            # [H+1, B]

    return features_full, actions_full, rewards_full, conts_full


def imag_loss(
    policy: Any,  # Policy instance
    policy_params: Any,  # PolicyParams (actor, critic, slow_critic, normalizers)
    features: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    continuations: jnp.ndarray,
    config: Dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """Compute actor-critic loss on imagined trajectories.

    Trains the actor to maximize value estimates (policy gradient) and
    trains the critic to predict TD(λ) returns accurately.

    This follows the DreamerV3 design:
        - Policy gradient with baseline (advantage)
        - Entropy regularization
        - Distribution loss for critic
        - Slow critic as target (EMA for stability)
        - Return/value/advantage normalization

    Args:
        policy: Policy instance (e.g., MLPPolicy)
        policy_params: PolicyParams with actor, critic, slow_critic, normalizers
        features: Imagined features [H+1, B, F] from world model (1 real + H imagined)
        actions: Imagined actions [H+1, B, A] (one-hot encoded)
        rewards: Imagined rewards [H+1, B]
        continuations: Imagined continuations [H+1, B]
        config: Configuration dictionary with:
                - horizon: Discount horizon (e.g., 333)
                - lambda: Lambda for TD(λ) (e.g., 0.95)
                - entropy_coef: Entropy regularization coefficient (e.g., 3e-4)
                - slow_reg: Slow critic regularization coefficient (e.g., 1.0)
                - slowtar: Whether to use slow critic as target (default True)
                - contdisc: Whether to use continuation discount (default True)
                - update_normalizers: Whether to update normalizer stats (default True)

    Returns:
        actor_loss: Scalar actor loss
        critic_loss: Scalar critic loss
        metrics: Dictionary with:
                - Logging metrics (advantage, reward, return, etc.)
                - 'normalizers': Updated normalizers dict

    Note: Returns updated normalizers in metrics['normalizers'] for the agent
          to update PolicyParams.normalizers.
    """

    # Extract config
    horizon = config.get('horizon', 333)
    lam = config.get('lambda', 0.95)
    actent = config.get('entropy_coef', 3e-4)
    slowreg = config.get('slow_reg', 1.0)
    slowtar = config.get('slowtar', True)
    contdisc = config.get('contdisc', True)
    update_normalizers_flag = config.get('update_normalizers', True)

    # Get normalizers
    retnorm = policy_params.normalizers['return']
    valnorm = policy_params.normalizers['value']
    advnorm = policy_params.normalizers['advantage']

    # Transpose from [H, B, ...] to [B, H, ...]
    H, B = rewards.shape
    features = jnp.transpose(features, (1, 0, 2))  # [B, H, F]
    actions = jnp.transpose(actions, (1, 0, 2))    # [B, H, A]
    rewards = jnp.transpose(rewards, (1, 0))        # [B, H]
    continuations = jnp.transpose(continuations, (1, 0))  # [B, H]

    # Compute values, log_probs, entropies from features
    # Flatten for vectorized application
    features_flat = features.reshape(B * H, -1)

    # Get value estimates (normalized space)
    value_dist = policy.apply_critic(policy_params.critic, features_flat, training=True)
    values_normalized = value_dist.mean().reshape(B, H)

    slow_value_dist = policy.apply_critic(policy_params.slow_critic, features_flat, training=False)
    slow_values_normalized = slow_value_dist.mean().reshape(B, H)

    # Get policy log_probs and entropies
    actor_dist = policy.apply_actor(policy_params.actor, features_flat, training=True)

    # actions are one-hot, get the action indices
    actions_flat = actions.reshape(B * H, -1)
    action_indices = jnp.argmax(actions_flat, axis=-1)

    log_probs = actor_dist.log_prob(action_indices).reshape(B, H)
    entropies = actor_dist.entropy().reshape(B, H)

    # === Value Normalization Scaling ===
    # Critic outputs are in NORMALIZED space, must scale to real space for TD(λ)
    voffset, vscale = get_normalizer_stats(valnorm)

    values = values_normalized * vscale + voffset
    slow_values = slow_values_normalized * vscale + voffset
    tarval = slow_values if slowtar else values

    # Compute discount and weight
    disc = 1 if contdisc else 1 - 1 / horizon
    weight = jnp.cumprod(disc * continuations, axis=1) / disc

    # Prepare for lambda returns
    last = jnp.zeros_like(continuations)
    term = 1 - continuations

    # Compute lambda returns in real space
    ret = lambda_return(
        last=last,
        term=term,
        rew=rewards,
        val=tarval,
        boot=tarval,
        disc=disc,
        lam=lam
    )

    # Update return normalizer
    if update_normalizers_flag:
        retnorm_new = update_normalizer(retnorm, ret)
    else:
        retnorm_new = retnorm
    roffset, rscale = get_normalizer_stats(retnorm_new)

    # Compute advantages (ret is [B, H-1], tarval is [B, H])
    adv = (ret - tarval[:, :-1]) / rscale

    # Update advantage normalizer
    if update_normalizers_flag:
        advnorm_new = update_normalizer(advnorm, adv)
    else:
        advnorm_new = advnorm
    aoffset, ascale = get_normalizer_stats(advnorm_new)

    # Normalize advantages
    adv_normed = (adv - aoffset) / ascale

    # Policy loss (REINFORCE with baseline)
    # Slice to match ret shape (H-1)
    policy_loss = -sg(weight[:, :-1]) * (
        log_probs[:, :-1] * sg(adv_normed) +
        actent * entropies[:, :-1]
    )

    # === Value Loss ===
    # Update value normalizer with returns
    if update_normalizers_flag:
        valnorm_new = update_normalizer(valnorm, ret)
    else:
        valnorm_new = valnorm
    voffset_new, vscale_new = get_normalizer_stats(valnorm_new)

    # Normalize target returns
    tar_normed = (ret - voffset_new) / vscale_new

    # Pad target to match trajectory length (H timesteps)
    # ret has shape [B, H-1], pad with zeros to get [B, H]
    tar_padded = jnp.concatenate([tar_normed, jnp.zeros_like(tar_normed[:, -1:])], axis=1)

    # Compute value loss using distribution.loss() (cross-entropy for TwoHot, MSE for MSE)
    tar_padded_flat = tar_padded.reshape(B * H)
    slow_values_flat = slow_values_normalized.reshape(B * H)

    # Distribution loss
    value_xe = value_dist.loss(sg(tar_padded_flat))
    slow_value_xe = value_dist.loss(sg(slow_values_flat))

    # Reshape back
    value_xe = value_xe.reshape(B, H)
    slow_value_xe = slow_value_xe.reshape(B, H)

    # Apply weights and slice to first H-1 timesteps
    critic_loss = sg(weight[:, :-1]) * (
        value_xe + slowreg * slow_value_xe
    )[:, :-1]

    # Aggregate losses
    actor_loss_scalar = policy_loss.mean()
    critic_loss_scalar = critic_loss.mean()

    # Outputs for downstream use (non-loggable data)
    outputs = {
        'returns': ret,  # [B, H] in real space for bootstrapping repl_loss
        'advantages': adv,
        'adv_normed': adv_normed,
        'weight': weight,
        'retnorm_state': retnorm_new,
        'valnorm_state': valnorm_new,
        'advnorm_state': advnorm_new,
    }

    # Metrics for logging (ONLY scalars)
    ret_normed = (ret - roffset) / rscale
    metrics = {
        # Advantage metrics
        'adv': adv.mean(),
        'adv_std': adv.std(),
        'adv_mag': jnp.abs(adv).mean(),

        # Trajectory metrics
        'rew': rewards.mean(),
        'con': continuations.mean(),
        'ret': ret_normed.mean(),
        'ret_min': ret_normed.min(),
        'ret_max': ret_normed.max(),
        'ret_rate': (jnp.abs(ret_normed) >= 1.0).mean(),

        # Value metrics
        'val': values.mean(),
        'tar': tar_normed.mean(),
        'slowval': slow_values.mean(),

        # Policy metrics
        'entropy': entropies.mean(),
        'weight': weight.mean(),
    }

    # Only log normalizer stats if normalizers are enabled (not 'none')
    if valnorm_new.impl != 'none':
        metrics['valnorm_scale'] = vscale_new
        metrics['valnorm_offset'] = voffset_new

    if retnorm_new.impl != 'none':
        metrics['retnorm_scale'] = rscale
        metrics['retnorm_offset'] = roffset

    if advnorm_new.impl != 'none':
        metrics['advnorm_scale'] = ascale
        metrics['advnorm_offset'] = aoffset

    return actor_loss_scalar, critic_loss_scalar, outputs, metrics


def repl_loss(
    policy: Policy,
    policy_params: Any,  # PolicyParams
    features: jnp.ndarray,
    rewards: jnp.ndarray,
    continuations: jnp.ndarray,
    last: jnp.ndarray,
    boot: jnp.ndarray,
    config: Dict,
) -> Tuple[jnp.ndarray, Dict]:
    """Compute critic loss on replay buffer data (optional).

    Some world model agents (like DreamerV3) train the critic on both:
        1. Imagined trajectories (via imag_loss)
        2. Real replay buffer data (via repl_loss)

    This improves sample efficiency by using real data to calibrate value estimates.

    Args:
        policy: Policy instance (e.g., MLPPolicy)
        policy_params: PolicyParams with critic, slow_critic, normalizers
        features: Features from replay buffer [B, T, F]
        rewards: Real rewards from replay buffer [B, T]
        continuations: Real continuations [B, T]
        last: Episode last flags [B, T]
        boot: Bootstrap values from imagination [B]
        config: Configuration dictionary with:
                - horizon: Discount horizon
                - lambda: Lambda for TD(λ)
                - slow_reg: Slow critic regularization
                - slowtar: Whether to use slow critic as target
                - update_normalizers: Whether to update normalizers

    Returns:
        critic_loss: Scalar critic loss
        metrics: Dictionary with:
                - Logging metrics
                - 'normalizers': Updated normalizers dict

    Note: This is optional and may not be used by all agents.
          DreamerV3 uses it for training critic on real data.
    """
    # Extract config
    horizon = config.get('horizon', 333)
    lam = config.get('lambda', 0.95)
    slowreg = config.get('slow_reg', 1.0)
    slowtar = config.get('slowtar', True)
    update_normalizers_flag = config.get('update_normalizers', True)

    # Get value normalizer
    valnorm = policy_params.normalizers['value']

    batch_size, time_steps = rewards.shape

    # === Value Normalization Scaling ===
    # Critic outputs are NORMALIZED, must scale to real space before TD(λ)
    voffset, vscale = get_normalizer_stats(valnorm)

    # Flatten features for vectorized critic application
    features_flat = features.reshape(batch_size * time_steps, -1)

    # Get value predictions in NORMALIZED space
    value_dist = policy.apply_critic(policy_params.critic, features_flat, training=True)
    values_normalized = value_dist.mean().reshape(batch_size, time_steps)

    slow_value_dist = policy.apply_critic(policy_params.slow_critic, features_flat, training=False)
    slow_values_normalized = slow_value_dist.mean().reshape(batch_size, time_steps)

    # Scale predictions from normalized -> real space
    values = values_normalized * vscale + voffset
    slow_values = slow_values_normalized * vscale + voffset
    tarval = slow_values if slowtar else values

    # Compute discount
    disc = 1 - 1 / horizon

    # Weight (mask out last steps)
    weight = jnp.float32(~last)

    # Compute termination flags
    term = 1 - continuations

    # Compute lambda returns in REAL space
    # Note: boot comes from imag_loss and is already in real space
    ret = lambda_return(
        last=last,
        term=term,
        rew=rewards,
        val=tarval,
        boot=boot,  # Already shaped [B, T]
        disc=disc,
        lam=lam
    )

    # Update value normalizer with real returns
    if update_normalizers_flag:
        valnorm_new = update_normalizer(valnorm, ret)
    else:
        valnorm_new = valnorm
    voffset_new, vscale_new = get_normalizer_stats(valnorm_new)

    # Normalize returns
    ret_normed = (ret - voffset_new) / vscale_new

    # Pad ret_normed to match value shape (lambda_return returns T-1 timesteps)
    ret_padded = jnp.concatenate([ret_normed, jnp.zeros_like(ret_normed[:, -1:])], axis=1)

    # Compute value loss using distribution.loss() (cross-entropy for TwoHot, MSE for MSE)
    ret_padded_flat = ret_padded.reshape(batch_size * time_steps)

    # Distribution loss
    value_xe = value_dist.loss(sg(ret_padded_flat))
    slow_value_xe = value_dist.loss(sg(slow_values_normalized.reshape(-1)))

    # Reshape back to (batch, time)
    value_xe = value_xe.reshape(batch_size, time_steps)
    slow_value_xe = slow_value_xe.reshape(batch_size, time_steps)

    # Compute value loss (exclude last timestep with [:, :-1])
    critic_loss = weight[:, :-1] * (
        value_xe + slowreg * slow_value_xe
    )[:, :-1]

    # Aggregate loss
    critic_loss_scalar = critic_loss.mean()

    # Outputs for downstream use (non-loggable data)
    outputs = {
        'returns': ret,
        'ret_normed': ret_normed,
        'valnorm_state': valnorm_new,
    }

    # Metrics for logging (ONLY scalars, matching old DreamerV3 exactly with repval/ prefix)
    metrics = {
        'repval/ret': ret.mean(),
        'repval/val': values.mean(),
        'repval/slowval': slow_values.mean(),
    }

    # Only log normalizer stats if valnorm is enabled (not 'none')
    if valnorm_new.impl != 'none':
        metrics['repval/valnorm_scale'] = vscale_new
        metrics['repval/valnorm_offset'] = voffset_new

    return critic_loss_scalar, outputs, metrics


# Export public API
__all__ = [
    'imagine_trajectory',
    'imag_loss',
    'repl_loss',
]
