"""Symlog and twohot encoding utilities for DreamerV3."""
import jax.numpy as jnp


def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """
    Symlog transformation: sign(x) * log(|x| + 1).

    Compresses large positive and negative values while preserving sign.
    Used for reward and value function normalization.

    Args:
        x: Input array

    Returns:
        Symlog-transformed array
    """
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """
    Inverse symlog transformation: sign(x) * (exp(|x|) - 1).

    Args:
        x: Symlog-transformed array

    Returns:
        Original scale array
    """
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)