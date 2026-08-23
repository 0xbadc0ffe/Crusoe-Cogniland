"""JAX configuration for continual-rl to enable bfloat16 compute dtype."""

import jax
import jax.numpy as jnp


# Global compute dtype (default: bfloat16 for memory efficiency)
COMPUTE_DTYPE = jnp.bfloat16


def enable_x64(enable=True):
    """Enable/disable 64-bit precision."""
    jax.config.update("jax_enable_x64", enable)


def set_compute_dtype(dtype='bfloat16'):
    """
    Set the global compute dtype for JAX computations.

    Args:
        dtype: Either 'bfloat16' or 'float32'
            - bfloat16: Half precision (16-bit), saves 50% memory
            - float32: Full precision (32-bit), more accurate
    """
    global COMPUTE_DTYPE

    if dtype == 'bfloat16':
        COMPUTE_DTYPE = jnp.bfloat16
    elif dtype == 'float32':
        COMPUTE_DTYPE = jnp.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Use 'bfloat16' or 'float32'.")

    print(f"JAX compute dtype set to: {COMPUTE_DTYPE}")


def cast_to_compute(x):
    """
    Cast tensors to compute dtype (bfloat16 or float32).

    Floating point arrays are cast to COMPUTE_DTYPE.
    Other dtypes (int, bool, uint8) are left unchanged.
    """
    if jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(COMPUTE_DTYPE)
    return x


def cast_pytree_to_compute(pytree):
    """Cast all floating point arrays in a pytree to compute dtype."""
    return jax.tree.map(cast_to_compute, pytree)


# Initialize with bfloat16 by default (for memory efficiency)
set_compute_dtype('bfloat16')
