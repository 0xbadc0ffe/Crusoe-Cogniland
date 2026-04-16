"""Distribution utilities for DreamerV3."""

from typing import Optional

import jax
import jax.numpy as jnp
import distrax

from cogniland.agents.commons.symlog import symexp


class OneHotDist:
    """
    One-hot categorical distribution for discrete latents.

    Used for the stochastic states in RSSM.
    """

    def __init__(self, logits: jnp.ndarray, unimix: float = 0.0):
        """
        Initialize one-hot distribution.

        Args:
            logits: Logits of shape (..., num_classes) or (..., num_dists, num_classes)
            unimix: Uniform mixture coefficient for regularization
        """
        self.logits = logits
        self.unimix = unimix

        # Apply uniform mixture if specified
        if unimix > 0:
            uniform = jnp.ones_like(logits) / logits.shape[-1]
            self.logits = (1 - unimix) * logits + unimix * jnp.log(uniform)

        # Create underlying categorical distribution
        self._dist = distrax.Categorical(logits=self.logits)

    def sample(self, seed: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Sample from the distribution with straight-through gradients.

        Args:
            seed: Random key

        Returns:
            One-hot encoded samples (discrete forward, continuous backward)
        """
        indices = self._dist.sample(seed=seed)

        # Convert to one-hot with same dtype as logits (for mixed precision)
        num_classes = self.logits.shape[-1]
        one_hot = jax.nn.one_hot(indices, num_classes, dtype=self.logits.dtype)

        # Straight-through gradient estimator (as in original DreamerV3)
        # Forward: discrete one-hot samples
        # Backward: gradients flow through continuous probabilities
        probs = jax.nn.softmax(self.logits, axis=-1)
        one_hot = jax.lax.stop_gradient(one_hot) + (probs - jax.lax.stop_gradient(probs))

        return one_hot

    def mode(self) -> jnp.ndarray:
        """
        Get the mode (argmax) of the distribution.

        Returns:
            One-hot encoded mode
        """
        indices = jnp.argmax(self.logits, axis=-1)
        num_classes = self.logits.shape[-1]
        one_hot = jax.nn.one_hot(indices, num_classes, dtype=self.logits.dtype)
        return one_hot

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log probability of values.

        Args:
            value: One-hot encoded values

        Returns:
            Log probabilities
        """
        # Convert one-hot to indices
        indices = jnp.argmax(value, axis=-1)
        return self._dist.log_prob(indices)

    def kl_divergence(self, other: 'OneHotDist') -> jnp.ndarray:
        """
        Compute KL divergence to another OneHotDist.

        Args:
            other: Other OneHotDist

        Returns:
            KL divergence summed across stoch dimension (as in original DreamerV3)
        """
        # Compute KL for each categorical
        kl = self._dist.kl_divergence(other._dist)

        # Sum across the stoch dimension (axis=-1)
        # This aggregates KL from all categoricals into a single value per batch element
        # Matching original DreamerV3: embodied.jax.outs.Agg(out, 1, jnp.sum)
        return kl.sum(axis=-1)

    def entropy(self) -> jnp.ndarray:
        """
        Compute entropy.

        Returns:
            Entropy
        """
        return self._dist.entropy()


class TwoHotDist:
    """
    Two-hot distribution for discrete regression.

    Used for value function and reward prediction with symlog encoding.
    Implements the exact DreamerV3 TwoHot with symmetric sum for zero initialization.
    """

    def __init__(
        self,
        logits: jnp.ndarray,
        num_bins: int = 255,
    ):
        """
        Initialize two-hot distribution.

        Args:
            logits: Logits of shape (..., num_bins)
            num_bins: Number of discrete bins
        """
        self.logits = logits
        self.num_bins = num_bins
        self.probs = jax.nn.softmax(logits, axis=-1)

        # Create bins using symexp (like original DreamerV3)
        # This creates symmetric bins in real space
        if num_bins % 2 == 1:
            # Odd number of bins: -bins, 0, +bins
            half = jnp.linspace(-20.0, 0.0, (num_bins - 1) // 2 + 1, dtype=jnp.float32)
            half = symexp(half)
            self.bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
        else:
            # Even number of bins: symmetric around 0
            half = jnp.linspace(-20.0, 0.0, num_bins // 2, dtype=jnp.float32)
            half = symexp(half)
            self.bins = jnp.concatenate([half, -half[::-1]], 0)

    def mean(self) -> jnp.ndarray:
        """
        Get the mean (expected value) using symmetric sum.

        Uses symmetric summation to ensure zero prediction at initialization
        when probabilities are uniform, avoiding numerical errors.

        Returns:
            Mean values (in real space, NOT symlog)
        """
        n = self.logits.shape[-1]
        if n % 2 == 1:
            # Odd bins: split into left, center, right
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m: m + 1]
            p3 = self.probs[..., m + 1:]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m: m + 1]
            b3 = self.bins[..., m + 1:]
            # Symmetric sum: center + (left[reverse] + right)
            wavg = (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
        else:
            # Even bins: split into left and right
            p1 = self.probs[..., :n // 2]
            p2 = self.probs[..., n // 2:]
            b1 = self.bins[..., :n // 2]
            b2 = self.bins[..., n // 2:]
            # Symmetric sum: (left[reverse] + right)
            wavg = ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(-1)
        return wavg

    def mode(self) -> jnp.ndarray:
        """
        Get the mode of the distribution.

        For TwoHot, mode returns the mean (weighted average) to match the
        original DreamerV3 behavior where pred() returns the weighted average.
        This ensures zero predictions at initialization with uniform logits.

        Returns:
            Mode values (in real space, NOT symlog)
        """
        # Return mean instead of argmax to match original DreamerV3
        # The original only has pred() which computes the weighted average
        return self.mean()

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log probability of values using two-hot encoding.

        Implements exact DreamerV3 two-hot encoding:
        - Find the two nearest bins
        - Interpolate probability between them based on distance

        Args:
            value: Target values (in real space, same as bins)

        Returns:
            Negative log probabilities (for loss)
        """
        # Compute two-hot encoding (find nearest bins and weights)
        # below: index of bin <= target
        # above: index of bin > target
        below = (self.bins <= value[..., None]).astype(jnp.int32).sum(-1) - 1
        above = len(self.bins) - (self.bins > value[..., None]).astype(jnp.int32).sum(-1)

        # Clip to valid range
        below = jnp.clip(below, 0, len(self.bins) - 1)
        above = jnp.clip(above, 0, len(self.bins) - 1)

        # Check if target equals a bin exactly
        equal = (below == above)

        # Compute distances
        dist_to_below = jnp.where(equal, 1.0, jnp.abs(self.bins[below] - value))
        dist_to_above = jnp.where(equal, 1.0, jnp.abs(self.bins[above] - value))
        total = dist_to_below + dist_to_above

        # Compute weights (inversely proportional to distance)
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total

        # Create two-hot target distribution
        target = (
            jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None] +
            jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None]
        )

        # Compute log probabilities
        log_pred = self.logits - jax.scipy.special.logsumexp(
            self.logits, -1, keepdims=True
        )

        # Return actual log probability (negative for valid probs)
        # This follows standard convention where log_prob < 0 for valid probabilities
        return (jax.lax.stop_gradient(target) * log_pred).sum(-1)

    def sample(self, seed: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Sample from the distribution.

        Args:
            seed: Random key

        Returns:
            Sampled values (in real space)
        """
        # Sample from categorical over bins
        cat_dist = distrax.Categorical(logits=self.logits)
        indices = cat_dist.sample(seed=seed)

        # Map to bin values (real space)
        values = self.bins[indices]

        return values

    def loss(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute loss (negative log probability).

        Args:
            value: Target values (in real space)

        Returns:
            Loss values (positive for minimization)
        """
        return -self.log_prob(value)

    def entropy(self) -> jnp.ndarray:
        """
        Compute entropy.

        Returns:
            Entropy
        """
        probs = jax.nn.softmax(self.logits, axis=-1)
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        return entropy


class MSEDist:
    """
    Mean Squared Error distribution for continuous outputs.

    This is used for image and vector observation reconstruction in DreamerV3.
    Unlike Normal distribution, this doesn't include the log normalizing constant.
    """

    def __init__(self, mean: jnp.ndarray, scale: Optional[float] = 1.0, transform=None):
        """
        Initialize MSE distribution.

        Args:
            mean: Mean predictions
            scale: Optional scale factor for the loss
            transform: Optional transform function (e.g., symlog) to apply to targets
        """
        self.mean = mean
        self.scale = scale
        self.transform = transform

    def mode(self) -> jnp.ndarray:
        """Return the mean (mode for MSE)."""
        return self.mean

    def sample(self, seed: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """
        For MSE, sampling just returns the mean (deterministic).

        Args:
            seed: Random key (unused)

        Returns:
            Mean values
        """
        return self.mean

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute negative MSE loss (acts as log probability).

        Args:
            value: Target values

        Returns:
            Negative MSE (per sample)
        """
        # Apply transform to targets if specified (e.g., symlog for rewards/values)
        # This check happens at trace time, so it's JIT-compatible
        if self.transform is not None:
            target = self.transform(value)
        else:
            target = value

        # Compute MSE and return negative (since log_prob should be maximized)
        # IMPORTANT: Sum over spatial dimensions (not mean), matching original DreamerV3
        # Original uses Agg(MSE, 3, jnp.sum) which sums over last 3 axes (H, W, C)
        mse = jnp.sum((self.mean - target) ** 2, axis=tuple(range(1, value.ndim)))
        return -mse / self.scale

    def loss(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute MSE loss.

        Args:
            value: Target values

        Returns:
            MSE loss (per sample)
        """
        return -self.log_prob(value)


class BinaryDist:
    """Binary (Bernoulli) distribution for binary classification."""

    def __init__(self, logits):
        """
        Initialize binary distribution.

        Args:
            logits: Logits for binary prediction
        """
        self.logits = jnp.asarray(logits, dtype=jnp.float32)

    def mode(self):
        """Get mode (most likely value)."""
        return self.logits > 0

    def prob(self, value):
        """
        Get probability of specific value.

        Args:
            value: Target value (0 or 1, or False/True)

        Returns:
            Probability of the value
        """
        probs = jax.nn.sigmoid(self.logits)
        value = jnp.asarray(value, dtype=jnp.float32)
        return value * probs + (1 - value) * (1 - probs)

    def log_prob(self, target):
        """
        Compute log probability of target.

        Args:
            target: Target values (0/1 or False/True)

        Returns:
            Log probabilities
        """
        target = jnp.asarray(target, dtype=jnp.float32)
        # Numerically stable log-sigmoid computation
        log_probs = jax.nn.log_sigmoid(self.logits)
        log_not_probs = jax.nn.log_sigmoid(-self.logits)
        return target * log_probs + (1 - target) * log_not_probs

    def sample(self, rng=None, shape=()):
        """
        Sample from the distribution.

        Args:
            rng: JAX random key
            shape: Additional shape for samples

        Returns:
            Binary samples
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        probs = jax.nn.sigmoid(self.logits)
        return jax.random.bernoulli(rng, probs, shape=shape)

    def entropy(self):
        """Compute entropy of the distribution."""
        probs = jax.nn.sigmoid(self.logits)
        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
        return -(probs * jnp.log(probs + 1e-8) +
                (1 - probs) * jnp.log(1 - probs + 1e-8))
