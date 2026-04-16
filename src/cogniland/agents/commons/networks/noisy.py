"""Noisy Networks for exploration in value-based RL.

Implements factorized Noisy Linear layers as described in:
"Noisy Networks for Exploration" (Fortunato et al., 2017)
https://arxiv.org/abs/1706.10295

Used in BTR (Beyond The Rainbow) and Rainbow DQN variants.
"""

import math
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class NoisyLinear(nn.Module):
    """
    Factorized Noisy Linear layer for exploration via parameter noise.

    Instead of epsilon-greedy exploration, this layer adds learned noise
    to weights and biases. The noise magnitude is learned, allowing the
    agent to control its own exploration.

    Factorized noise uses: eps_w = eps_in @ eps_out.T
    where eps_in and eps_out are transformed Gaussian noise vectors.

    Args:
        features: Number of output features
        sigma_init: Initial value for noise scale (default 0.5)

    Usage:
        # During training (with noise):
        out = layer(x, rng=rng_key)

        # During evaluation (deterministic):
        out = layer(x, rng=None)
    """
    features: int
    sigma_init: float = 0.5

    @nn.compact
    def __call__(self, x: jnp.ndarray, rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        """
        Forward pass with optional noise.

        Args:
            x: Input tensor of shape (..., in_features)
            rng: Random key for noise sampling. If None, uses mean weights only.

        Returns:
            Output tensor of shape (..., features)
        """
        in_features = x.shape[-1]

        # Compute scales using Python math.sqrt for static initialization
        mu_scale = 1.0 / math.sqrt(in_features)
        sigma_w_scale = self.sigma_init / math.sqrt(in_features)
        sigma_b_scale = self.sigma_init / math.sqrt(self.features)  # Use out_features for bias

        # Mean weights and biases (standard learnable parameters)
        mu_w = self.param(
            'mu_w',
            nn.initializers.uniform(scale=mu_scale),
            (in_features, self.features)
        )
        mu_b = self.param(
            'mu_b',
            nn.initializers.zeros,
            (self.features,)
        )

        # Noise scale parameters (learned)
        sigma_w = self.param(
            'sigma_w',
            nn.initializers.constant(sigma_w_scale),
            (in_features, self.features)
        )
        sigma_b = self.param(
            'sigma_b',
            nn.initializers.constant(sigma_b_scale),
            (self.features,)
        )

        if rng is None:
            # Evaluation mode: use mean weights only (no noise)
            return x @ mu_w + mu_b

        # Training mode: add factorized noise
        rng_in, rng_out = jax.random.split(rng)

        # Sample noise and apply factorized transformation
        # f(x) = sign(x) * sqrt(|x|) as per the paper
        eps_in = self._factorized_noise(rng_in, (in_features,))
        eps_out = self._factorized_noise(rng_out, (self.features,))

        # Factorized noise: outer product
        eps_w = jnp.outer(eps_in, eps_out)
        eps_b = eps_out

        # Noisy weights: w = mu + sigma * eps
        w = mu_w + sigma_w * eps_w
        b = mu_b + sigma_b * eps_b

        return x @ w + b

    def _factorized_noise(self, rng: jax.random.PRNGKey, shape: tuple) -> jnp.ndarray:
        """
        Generate factorized noise: f(x) = sign(x) * sqrt(|x|).

        This transformation from the NoisyNet paper produces
        heavier tails than standard Gaussian noise.
        """
        x = jax.random.normal(rng, shape)
        return jnp.sign(x) * jnp.sqrt(jnp.abs(x))


class NoisyDense(nn.Module):
    """
    Drop-in replacement for nn.Dense with optional noise.

    This wraps NoisyLinear to provide the same interface as nn.Dense,
    making it easy to swap between standard and noisy layers.

    When noisy=False, behaves exactly like nn.Dense.
    When noisy=True, adds factorized parameter noise during training.
    """
    features: int
    noisy: bool = True
    sigma_init: float = 0.5
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jnp.ndarray, rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        if self.noisy:
            return NoisyLinear(
                features=self.features,
                sigma_init=self.sigma_init
            )(x, rng=rng)
        else:
            return nn.Dense(
                features=self.features,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )(x)