"""Functional normalizers for value, return, and advantage normalization.

PyTree-based implementation where static config (impl, rate, etc.) lives in aux_data
and dynamic state (mean, sqrs, count) are traced children.
"""
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.tree_util

# Type aliases
f32 = jnp.float32
sg = lambda x: jax.lax.stop_gradient(x)


@jax.tree_util.register_pytree_node_class
@dataclass
class Normalizer:
    """
    Normalizer with static config and dynamic state.

    Static fields (aux_data): impl, rate, limit, perclo, perchi, debias
    Dynamic fields (children): mean, sqrs, lo, hi, count
    """
    # Static config (not traced by JAX)
    impl: str
    rate: float
    limit: float
    perclo: float
    perchi: float
    debias: bool

    # Dynamic state (traced by JAX) - stored as dict for flexibility
    state: Dict[str, jnp.ndarray]

    def tree_flatten(self):
        """Flatten to (children, aux_data) for JAX PyTree."""
        # Children are the dynamic arrays that JAX traces
        children = (self.state,)
        # Aux data is static config (not traced, used for cache key)
        aux_data = (self.impl, self.rate, self.limit, self.perclo, self.perchi, self.debias)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten from (aux_data, children) to reconstruct Normalizer."""
        impl, rate, limit, perclo, perchi, debias = aux_data
        state, = children
        return cls(impl, rate, limit, perclo, perchi, debias, state)

    def replace(self, **kwargs):
        """Functional update (like dataclass replace)."""
        # Update state dict
        if 'state' in kwargs:
            return Normalizer(
                self.impl, self.rate, self.limit,
                self.perclo, self.perchi, self.debias,
                kwargs['state']
            )
        # Update individual state fields
        new_state = dict(self.state)
        for k, v in kwargs.items():
            if k in new_state:
                new_state[k] = v
        return Normalizer(
            self.impl, self.rate, self.limit,
            self.perclo, self.perchi, self.debias,
            new_state
        )


def init_normalizer(impl: str = 'meanstd', rate: float = 0.01,
                    limit: float = 1e-8, perclo: float = 5.0,
                    perchi: float = 95.0, debias: bool = True) -> Normalizer:
    """
    Initialize normalizer with config and state.

    Returns a Normalizer PyTree where config is static aux_data
    and state arrays are dynamic children.

    Args:
        impl: Implementation type ('none', 'meanstd', 'perc')
        rate: Update rate for running statistics
        limit: Minimum standard deviation
        perclo: Lower percentile (for 'perc' implementation)
        perchi: Upper percentile (for 'perc' implementation)
        debias: Whether to debias the running statistics

    Returns:
        Normalizer instance (JAX PyTree)
    """
    state = {}

    if impl == 'none':
        pass
    elif impl == 'meanstd':
        state['mean'] = jnp.array(0.0, dtype=f32)
        state['sqrs'] = jnp.array(0.0, dtype=f32)
        if debias:
            state['count'] = jnp.array(0.0, dtype=f32)
    elif impl == 'perc':
        state['lo'] = jnp.array(0.0, dtype=f32)
        state['hi'] = jnp.array(0.0, dtype=f32)
        if debias:
            state['count'] = jnp.array(0.0, dtype=f32)
    else:
        raise NotImplementedError(f"Unknown normalizer implementation: {impl}")

    return Normalizer(impl, rate, limit, perclo, perchi, debias, state)


def update_normalizer(norm: Normalizer, x: jnp.ndarray) -> Normalizer:
    """
    Update normalizer statistics with new values (pure function).

    Args:
        norm: Normalizer instance
        x: Values to update statistics with

    Returns:
        Updated Normalizer (new instance, does not mutate input)
    """
    # Stop gradient and convert to float32
    x = sg(jnp.asarray(x, dtype=f32))

    if norm.impl == 'none':
        return norm

    # Create new state dict (functional update)
    new_state = dict(norm.state)

    if norm.impl == 'meanstd':
        # Update running mean and mean of squares
        x_mean = jnp.mean(x)
        x_sqrs = jnp.mean(jnp.square(x))

        new_state['mean'] = (1 - norm.rate) * norm.state['mean'] + norm.rate * x_mean
        new_state['sqrs'] = (1 - norm.rate) * norm.state['sqrs'] + norm.rate * x_sqrs

        if norm.debias:
            new_state['count'] = (1 - norm.rate) * norm.state['count'] + norm.rate

    elif norm.impl == 'perc':
        # Update running percentiles
        lo = jnp.percentile(x, norm.perclo)
        hi = jnp.percentile(x, norm.perchi)

        new_state['lo'] = (1 - norm.rate) * norm.state['lo'] + norm.rate * lo
        new_state['hi'] = (1 - norm.rate) * norm.state['hi'] + norm.rate * hi

        if norm.debias:
            new_state['count'] = (1 - norm.rate) * norm.state['count'] + norm.rate

    return norm.replace(state=new_state)


def get_normalizer_stats(norm: Normalizer) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get current normalization statistics (offset and scale).

    Args:
        norm: Normalizer instance

    Returns:
        Tuple of (offset, scale) for normalization: (x - offset) / scale
    """
    if norm.impl == 'none':
        return jnp.array(0.0, dtype=f32), jnp.array(1.0, dtype=f32)

    # Compute debiasing correction
    corr = 1.0
    if norm.debias and 'count' in norm.state:
        corr = 1.0 / jnp.maximum(norm.rate, norm.state['count'])

    if norm.impl == 'meanstd':
        mean = norm.state['mean'] * corr
        var = norm.state['sqrs'] * corr - mean ** 2
        std = jnp.sqrt(jnp.maximum(0.0, var))
        std = jnp.maximum(norm.limit, std)
        return mean, std

    elif norm.impl == 'perc':
        lo = norm.state['lo'] * corr
        hi = norm.state['hi'] * corr
        scale = jnp.maximum(norm.limit, hi - lo)
        return sg(lo), sg(scale)

    else:
        raise NotImplementedError(f"Unknown normalizer implementation: {norm.impl}")


def normalize(norm: Normalizer, x: jnp.ndarray,
              update: bool = True) -> Tuple[jnp.ndarray, Normalizer]:
    """
    Normalize values and optionally update statistics.

    Args:
        norm: Normalizer instance
        x: Values to normalize
        update: Whether to update running statistics

    Returns:
        Tuple of (normalized_values, updated_normalizer)
    """
    # Get current stats
    offset, scale = get_normalizer_stats(norm)

    # Normalize
    normalized = (x - offset) / scale

    # Update statistics if requested
    if update:
        new_norm = update_normalizer(norm, x)
    else:
        new_norm = norm

    return normalized, new_norm


def denormalize(norm: Normalizer, normalized_x: jnp.ndarray) -> jnp.ndarray:
    """
    Denormalize values back to original scale.

    Args:
        norm: Normalizer instance
        normalized_x: Normalized values

    Returns:
        Denormalized values
    """
    offset, scale = get_normalizer_stats(norm)
    return normalized_x * scale + offset


# Slow value network EMA update (pure function)
def update_slow_critic_params(
    slow_params: Any,
    fast_params: Any,
    rate: float = 0.02
) -> Any:
    """
    Update slow critic parameters using exponential moving average.

    Args:
        slow_params: Current slow critic parameters
        fast_params: Current fast critic parameters
        rate: EMA update rate

    Returns:
        Updated slow critic parameters
    """
    return jax.tree.map(
        lambda slow, fast: (1 - rate) * slow + rate * fast,
        slow_params,
        fast_params
    )


