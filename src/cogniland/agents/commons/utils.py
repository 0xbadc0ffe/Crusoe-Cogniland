"""Common utility functions for RL agents"""

import jax
import jax.numpy as jnp
import numpy as np


def orthogonal_init(gain=1.0, column_axis=-1):
    """
    JIT-safe orthogonal initializer.
    Produces weights whose columns along `column_axis` are orthonormal.
    Works for Dense and Conv kernels (flattens all but the column axis).

    Args:
        gain: Multiplicative factor to apply to the orthogonal matrix
        column_axis: The axis to use as columns (default: -1)

    Returns:
        An initializer function that produces orthogonal weights
    """
    def init(key, shape, dtype=jnp.float32):
        shape = tuple(int(s) for s in shape)
        axis = column_axis % len(shape)

        n_rows = int(np.prod(shape[:axis])) or 1
        n_cols = int(np.prod(shape[axis:])) or 1

        # Work on a 2D matrix of shape (n_rows, n_cols)
        a = jax.random.normal(key, (n_rows, n_cols), dtype)

        # If we have more columns than rows, QR on the transpose and transpose back
        # to guarantee enough orthonormal columns.
        transposed = n_rows < n_cols
        a_qr = a.T if transposed else a

        q, r = jnp.linalg.qr(a_qr, mode="reduced")
        q = q * jnp.sign(jnp.diag(r))

        q = q.T if transposed else q  # final shape (n_rows, n_cols)

        w = (gain * q).astype(dtype)

        # Reshape back to original shape, keeping the last axis as the column axis
        pre = shape[:axis]
        post = shape[axis:]
        return w.reshape(pre + post)

    return init

# Common initializers with different gains
ortho_init = orthogonal_init(jnp.sqrt(2.0))
ortho_init_small = orthogonal_init(0.01)
ortho_init_one = orthogonal_init(1.0)
