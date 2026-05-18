"""LaProp + AGC — DreamerV3's reference optimizer.
"""
from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import optax


def clip_by_agc(clip: float = 0.3, pmin: float = 1e-3) -> optax.GradientTransformation:
    """Adaptive gradient clipping (Brock et al. 2021).

    Clips each update by the ratio of its norm to the corresponding parameter's norm:
        update_clipped = update * min(1, clip * max(pmin, ||param||) / ||update||)
    """

    def init_fn(params):
        del params
        return ()

    def update_fn(updates, state, params=None):
        del state

        def fn(param, update):
            unorm = jnp.linalg.norm(update.flatten(), 2)
            pnorm = jnp.linalg.norm(param.flatten(), 2)
            upper = clip * jnp.maximum(pmin, pnorm)
            return update * (1.0 / jnp.maximum(1.0, unorm / upper))

        if clip and params is not None:
            updates = jax.tree.map(fn, params, updates)
        return updates, ()

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta: float = 0.999, eps: float = 1e-8) -> optax.GradientTransformation:
    """RMS scaling with bias correction (the RMSprop piece of LaProp)."""

    def init_fn(params):
        nu = jax.tree.map(lambda t: jnp.zeros_like(t, jnp.float32), params)
        step = jnp.zeros((), jnp.int32)
        return (step, nu)

    def update_fn(updates, state, params=None):
        del params
        step, nu = state
        step = optax.safe_int32_increment(step)
        nu = jax.tree.map(lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
        nu_hat = optax.bias_correction(nu, beta, step)
        updates = jax.tree.map(lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
        return updates, (step, nu)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(
    beta: float = 0.9, nesterov: bool = False
) -> optax.GradientTransformation:
    """Momentum scaling with bias correction."""

    def init_fn(params):
        mu = jax.tree.map(lambda t: jnp.zeros_like(t, jnp.float32), params)
        step = jnp.zeros((), jnp.int32)
        return (step, mu)

    def update_fn(updates, state, params=None):
        del params
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
    wdregex: str = r"/kernel$",
    schedule: str = "const",
    warmup: int = 1000,
    anneal: int = 0,
) -> optax.GradientTransformation:
    """LaProp (Liu et al. 2020)
    Chain: [AGC, RMS scaling, momentum, weight decay (optional), LR schedule].
    """
    chain: list[optax.GradientTransformation] = []
    if agc > 0:
        chain.append(clip_by_agc(agc))
    chain.append(scale_by_rms(beta2, eps))
    if momentum:
        chain.append(scale_by_momentum(beta1, nesterov))
    if wd > 0:
        assert not wdregex[0].isnumeric(), wdregex
        pattern = re.compile(wdregex)
        wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
        chain.append(optax.add_decayed_weights(wd, wdmask))

    assert anneal > 0 or schedule == "const"
    if schedule == "const":
        sched = optax.constant_schedule(lr)
    elif schedule == "linear":
        sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == "cosine":
        sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
        raise NotImplementedError(schedule)
    if warmup:
        ramp = optax.linear_schedule(0.0, lr, warmup)
        sched = optax.join_schedules([ramp, sched], [warmup])

    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)
