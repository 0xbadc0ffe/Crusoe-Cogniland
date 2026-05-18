"""Distribution and transform utilities for DreamerV3.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


class TwoHotDist:
    """Two-hot categorical distribution over a fixed symlog-spaced support.

    Used by DreamerV3's reward and critic heads. Given `logits` of shape (..., num_bins),
    models a real-valued distribution whose support is `symexp(bin_centers)`. Sampling
    returns the (continuous) expected value. `log_prob(x)` places two-hot weight on the
    bins straddling `symlog(x)` and returns the cross-entropy with `logits`.
    """

    def __init__(self, logits: jnp.ndarray, low: float = -20.0, high: float = 20.0):
        self.logits = logits
        self.num_bins = logits.shape[-1]
        # bin centers in symlog space, symexp'd into real space
        self.bin_centers_symlog = jnp.linspace(low, high, self.num_bins)
        self.bin_centers = symexp(self.bin_centers_symlog)
        self.probs = jax.nn.softmax(logits, axis=-1)

    def mean(self) -> jnp.ndarray:
        n = self.num_bins
        if n % 2 == 1:
            m = (n - 1) // 2
            p_lo = self.probs[..., :m]
            p_mid = self.probs[..., m:m + 1]
            p_hi = self.probs[..., m + 1:]
            b_lo = self.bin_centers[:m]
            b_mid = self.bin_centers[m:m + 1]
            b_hi = self.bin_centers[m + 1:]
            return (p_mid * b_mid).sum(axis=-1) + (
                (p_lo * b_lo)[..., ::-1] + p_hi * b_hi
            ).sum(axis=-1)
        else:
            m = n // 2
            return (
                (self.probs[..., :m] * self.bin_centers[:m])[..., ::-1]
                + self.probs[..., m:] * self.bin_centers[m:]
            ).sum(axis=-1)

    # alias used in some DreamerV3 code
    @property
    def mode(self) -> jnp.ndarray:
        return self.mean()

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        # map x → symlog → two-hot target
        x_symlog = symlog(x)
        low = self.bin_centers_symlog[0]
        high = self.bin_centers_symlog[-1]
        # normalized bin position
        pos = jnp.clip(
            (x_symlog - low) / (high - low) * (self.num_bins - 1), 0.0, self.num_bins - 1
        )
        below = jnp.floor(pos).astype(jnp.int32)
        above = jnp.ceil(pos).astype(jnp.int32)
        weight_above = pos - below.astype(pos.dtype)
        weight_below = 1.0 - weight_above
        target = (
            jax.nn.one_hot(below, self.num_bins) * weight_below[..., None]
            + jax.nn.one_hot(above, self.num_bins) * weight_above[..., None]
        )
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        return (target * log_probs).sum(axis=-1)


class OneHotCategoricalSTE:
    """Categorical distribution with straight-through gradient estimator.

    Forward: discrete one-hot sample.
    Backward: gradient flows through the softmax probabilities.
    """

    def __init__(self, logits: jnp.ndarray, unimix: float = 0.0):
        self.raw_logits = logits
        self.num_classes = logits.shape[-1]
        probs = jax.nn.softmax(logits, axis=-1)
        if unimix > 0:
            uniform = jnp.ones_like(probs) / self.num_classes
            probs = (1 - unimix) * probs + unimix * uniform
            # re-derive logits for entropy / log_prob in a uniform-mixed space
            logits = jnp.log(probs + 1e-12)
        self.logits = logits
        self.probs = probs

    def sample(self, seed) -> jnp.ndarray:
        idx = jax.random.categorical(seed, self.logits, axis=-1)
        hard = jax.nn.one_hot(idx, self.num_classes)
        # straight-through: forward is hard, backward flows through probs
        return jax.lax.stop_gradient(hard - self.probs) + self.probs

    @property
    def mode(self) -> jnp.ndarray:
        idx = jnp.argmax(self.logits, axis=-1)
        hard = jax.nn.one_hot(idx, self.num_classes)
        return jax.lax.stop_gradient(hard - self.probs) + self.probs

    def entropy(self) -> jnp.ndarray:
        # H = -sum p log p (per categorical axis, sum over classes)
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        return -(self.probs * log_probs).sum(axis=-1)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        return (value * log_probs).sum(axis=-1)

