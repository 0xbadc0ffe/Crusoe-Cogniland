"""Lambda returns computation for DreamerV3."""
import jax.numpy as jnp
import chex

# Type aliases
f32 = jnp.float32


def lambda_return(
    last: jnp.ndarray,
    term: jnp.ndarray,
    rew: jnp.ndarray,
    val: jnp.ndarray,
    boot: jnp.ndarray,
    disc: float,
    lam: float
) -> jnp.ndarray:
    """
    Compute lambda returns for value learning.

    This is the exact implementation from the original DreamerV3.

    Args:
        last: Whether each timestep is the last in episode (batch, time)
        term: Whether each timestep is terminal (batch, time)
        rew: Rewards (batch, time)
        val: Value predictions (batch, time)
        boot: Bootstrap values (batch, time)
        disc: Discount factor (e.g., 0.997 or 1 - 1/horizon)
        lam: Lambda parameter for TD(λ) (e.g., 0.95)

    Returns:
        Lambda returns (batch, time)
    """
    chex.assert_equal_shape((last, term, rew, val, boot))

    # Initialize returns list with bootstrap value at the end
    rets = [boot[:, -1]]

    # Compute continuation and lambda factors
    live = (1 - f32(term))[:, 1:] * disc  # Whether episode continues (with discount)
    cont = (1 - f32(last))[:, 1:] * lam   # Whether to continue bootstrapping

    # Intermediate values for TD(λ)
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]

    # Backward pass to compute returns
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])

    # Stack and reverse to get forward time order (exclude bootstrap)
    return jnp.stack(list(reversed(rets))[:-1], 1)
