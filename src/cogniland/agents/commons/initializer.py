"""
This module contains initialization functions and other utilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, Callable


def dreamerv3_trunc_normal(
    scale: float = 1.0,
    mode: str = 'fan_in',
    dtype: jnp.dtype = jnp.float32
) -> Callable:
    """
    Reproduction of Danijar Hafner's DreamerV3 truncated-normal initializer.
    Source: https://github.com/danijar/dreamerv3/blob/main/embodied/jax/nets.py # line 144

    Draws samples from a truncated normal N(0, 1) clipped to [-2, 2],
    applies variance correction (with a multiplicative factor of 1.1368), and scales by sqrt(scale / fan),
    where fan is 'fan_in' or 'fan_out' depending on `mode`.

    Args:
        scale: overall scaling factor (default 1.0).
        mode: one of {"fan_in", "fan_out", "fan_avg"} to select fan type.
        dtype: output dtype (default float32).

    Returns:
        A callable initializer(key, shape, dtype=None) → jnp.ndarray.
    """
    truncation_correction = 1.1368683772161603  # sqrt(1 / (1 - 2 * P(|X| > 2))) for X ~ N(0,1)

    def init(
        key: jax.random.PRNGKey,
        shape: Tuple[int, ...],
        dtype_in: Optional[jnp.dtype] = None
    ) -> jnp.ndarray:
        """Initializer function."""
        dtype_use = dtype_in or dtype

        if len(shape) < 2:
            # For 1D shapes (bias), just use fan_in = 1
            fan = 1.0
        else:
            fan_in, fan_out = compute_fans(shape)
            if mode == 'fan_in':
                fan = fan_in
            elif mode == 'fan_out':
                fan = fan_out
            elif mode == 'fan_avg':
                fan = (fan_in + fan_out) / 2.0
            else:
                raise ValueError(
                    f"Invalid mode {mode}, expected one of "
                    f"['fan_in', 'fan_out', 'fan_avg']"
                )

        stddev = truncation_correction * np.sqrt(scale / fan)
        return jax.random.truncated_normal(key, -2, 2, shape, dtype_use) * stddev

    return init


def compute_fans(shape: Tuple[int, ...]) -> Tuple[float, float]:
    """
    Compute fan-in and fan-out for a given shape.

    This matches the computation in the original DreamerV3 implementation.

    Args:
        shape: Shape of the weight tensor

    Returns:
        Tuple of (fan_in, fan_out)
    """
    if len(shape) == 0:
        return (1.0, 1.0)
    elif len(shape) == 1:
        return (1.0, float(shape[0]))
    elif len(shape) == 2:
        return (float(shape[0]), float(shape[1]))
    else:
        # For convolutions: (kernel_h, kernel_w, in_channels, out_channels)
        # fan_in = in_channels * kernel_size
        # fan_out = out_channels * kernel_size
        receptive_field_size = np.prod(shape[:-2])
        fan_in = float(shape[-2]) * receptive_field_size
        fan_out = float(shape[-1]) * receptive_field_size
        return (fan_in, fan_out)


class Initializer:
    """
    Helper class for weight initialization that mirrors the original DreamerV3 Initializer.

    This class provides a unified interface for different initialization methods
    and automatically handles the fan mode.
    """

    def __init__(
        self,
        dist: str = 'trunc_normal',
        fan: str = 'in',
        scale: float = 1.0,
        dtype: jnp.dtype = jnp.float32
    ):
        """
        Initialize the Initializer.

        Args:
            dist: Distribution type ('trunc_normal', 'zeros', 'uniform', 'normal')
            fan: Fan mode ('in', 'out', 'avg')
            scale: Scaling factor
            dtype: Output dtype
        """
        self.dist = dist
        self.fan = fan
        self.scale = scale
        self.dtype = dtype

    def __call__(
        self,
        key: jax.random.PRNGKey,
        shape: Tuple[int, ...],
        dtype: Optional[jnp.dtype] = None
    ) -> jnp.ndarray:
        """
        Initialize weights.

        Args:
            key: JAX random key
            shape: Shape of the weight tensor
            dtype: Optional dtype override

        Returns:
            Initialized weight tensor
        """
        dtype_use = dtype or self.dtype

        if self.dist == 'zeros':
            return jnp.zeros(shape, dtype_use)

        elif self.dist == 'trunc_normal':
            init_fn = dreamerv3_trunc_normal(
                scale=self.scale,
                mode=f'fan_{self.fan}',
                dtype=dtype_use
            )
            return init_fn(key, shape)

        elif self.dist == 'uniform':
            fan_in, fan_out = compute_fans(shape) if len(shape) >= 2 else (1.0, 1.0)
            fan = {
                'in': fan_in,
                'out': fan_out,
                'avg': (fan_in + fan_out) / 2.0
            }[self.fan]
            limit = np.sqrt(1 / fan) * self.scale
            return jax.random.uniform(key, shape, dtype_use, -limit, limit)

        elif self.dist == 'normal':
            fan_in, fan_out = compute_fans(shape) if len(shape) >= 2 else (1.0, 1.0)
            fan = {
                'in': fan_in,
                'out': fan_out,
                'avg': (fan_in + fan_out) / 2.0
            }[self.fan]
            stddev = np.sqrt(1 / fan) * self.scale
            return jax.random.normal(key, shape, dtype_use) * stddev

        else:
            raise ValueError(f"Unknown distribution: {self.dist}")

    def __repr__(self) -> str:
        return f"Initializer(dist={self.dist}, fan={self.fan}, scale={self.scale})"
