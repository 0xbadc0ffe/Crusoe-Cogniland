"""Stub for replay buffer state — replaced by Agent 4 with full implementation."""

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import chex


@chex.dataclass
class ReplayBufferState:
    data: Dict[str, np.ndarray]
    write_idx: jnp.ndarray
    valid_size: jnp.ndarray
    total_steps: jnp.ndarray
    rng: jax.random.PRNGKey
    seq_step: jnp.ndarray
    seq_target: jnp.ndarray
