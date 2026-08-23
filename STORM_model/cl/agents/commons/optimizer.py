"""Custom optimizers"""
from typing import Any
import re

import jax
import jax.numpy as jnp
import optax


def clip_by_agc(clip: float = 0.3, pmin: float = 1e-3) -> optax.GradientTransformation:
    """
    Adaptive gradient clipping from DreamerV3.

    Clips gradients based on the ratio of gradient norm to parameter norm.
    This is more stable than global gradient clipping for large models.

    Args:
        clip: Clipping threshold (default 0.3)
        pmin: Minimum parameter norm to avoid division by zero

    Returns:
        Optax gradient transformation
    """
    def init_fn(params):
        return ()

    def update_fn(updates, state, params=None):
        def fn(param, update):
            unorm = jnp.linalg.norm(update.flatten(), 2)
            pnorm = jnp.linalg.norm(param.flatten(), 2)
            upper = clip * jnp.maximum(pmin, pnorm)
            return update * (1 / jnp.maximum(1.0, unorm / upper))

        updates = jax.tree.map(fn, params, updates) if (clip and params is not None) else updates
        return updates, ()

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta: float = 0.999, eps: float = 1e-8) -> optax.GradientTransformation:
    """
    Scale updates by RMS (like RMSprop) with bias correction.

    Args:
        beta: Decay rate for RMS (default 0.999)
        eps: Epsilon for numerical stability

    Returns:
        Optax gradient transformation
    """
    def init_fn(params):
        nu = jax.tree.map(lambda t: jnp.zeros_like(t, jnp.float32), params)
        step = jnp.zeros((), jnp.int32)
        return (step, nu)

    def update_fn(updates, state, params=None):
        step, nu = state
        step = optax.safe_int32_increment(step)
        nu = jax.tree.map(
            lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates
        )
        nu_hat = optax.bias_correction(nu, beta, step)
        updates = jax.tree.map(
            lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat
        )
        return updates, (step, nu)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta: float = 0.9, nesterov: bool = False) -> optax.GradientTransformation:
    """
    Scale updates by momentum with bias correction.

    Args:
        beta: Momentum coefficient (default 0.9)
        nesterov: Whether to use Nesterov momentum

    Returns:
        Optax gradient transformation
    """
    def init_fn(params):
        mu = jax.tree.map(lambda t: jnp.zeros_like(t, jnp.float32), params)
        step = jnp.zeros((), jnp.int32)
        return (step, mu)

    def update_fn(updates, state, params=None):
        step, mu = state
        step = optax.safe_int32_increment(step)
        mu = optax.update_moment(updates, mu, beta, 1)
        if nesterov:
            mu_nesterov = optax.update_moment(updates, mu, beta, 1)
            mu_hat = optax.bias_correction(mu_nesterov, beta, step)
        else:
            mu_hat = optax.bias_correction(mu, beta, step)
        return mu_hat, (step, mu)

    return optax.GradientTransformation(init_fn, update_fn)


def laprop(
    lr: float = 4e-5,
    agc: float = 0.3,
    eps: float = 1e-20,
    beta1: float = 0.9,
    beta2: float = 0.999,
    momentum: bool = True,
    nesterov: bool = False,
    wd: float = 0.0,
    wdregex: str = r'/kernel$',
    schedule: str = 'const',
    warmup: int = 1000,
    anneal: int = 0,
) -> optax.GradientTransformation:
    """
    LaProp optimizer from DreamerV3.

    This matches the original DreamerV3 implementation by chaining:
    1. Adaptive gradient clipping
    2. RMS scaling (like RMSprop)
    3. Momentum scaling (optional)
    4. Weight decay (optional)
    5. Learning rate schedule

    Args:
        lr: Learning rate
        agc: Adaptive gradient clipping threshold
        eps: Epsilon for numerical stability
        beta1: Momentum coefficient
        beta2: RMS decay rate
        momentum: Whether to use momentum
        nesterov: Whether to use Nesterov momentum
        wd: Weight decay coefficient
        wdregex: Regex pattern for weight decay (e.g., r'/kernel$' for kernels only)
        schedule: Learning rate schedule ('const', 'linear', 'cosine')
        warmup: Warmup steps
        anneal: Annealing steps

    Returns:
        Optax gradient transformation
    """
    chain = []

    # Adaptive gradient clipping (from embodied.jax.opt.clip_by_agc)
    if agc > 0:
        chain.append(clip_by_agc(agc))

    # RMS scaling (like RMSprop but with bias correction)
    chain.append(scale_by_rms(beta2, eps))

    # Momentum scaling
    if momentum:
        chain.append(scale_by_momentum(beta1, nesterov))

    # Weight decay (only on matching parameters like kernels)
    if wd > 0:
        assert not wdregex[0].isnumeric(), wdregex
        pattern = re.compile(wdregex)
        wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
        chain.append(optax.add_decayed_weights(wd, wdmask))

    # Learning rate schedule
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
        sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
        sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
        sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
        raise NotImplementedError(schedule)

    if warmup:
        ramp = optax.linear_schedule(0.0, lr, warmup)
        sched = optax.join_schedules([ramp, sched], [warmup])

    # Use scale_by_learning_rate for schedule (positive learning rate)
    chain.append(optax.scale_by_learning_rate(sched))

    return optax.chain(*chain)

def ema_update(
    slow_params: Any,
    fast_params: Any,
    rate: float = 0.02,
) -> Any:
    """
    Exponential moving average update for slow critic.

    Args:
        slow_params: Current slow network parameters
        fast_params: Fast network parameters to incorporate
        rate: Update rate (1.0 = copy fast params)

    Returns:
        Updated slow parameters
    """
    return jax.tree.map(
        lambda s, f: (1 - rate) * s + rate * f,
        slow_params, fast_params
    )