"""
MLPHead implementation for DreamerV3.

Unified head that handles actor, critic, and predictor functionality with
configurable output distributions.
"""

from typing import Tuple, Union, Any
import jax.numpy as jnp
import flax.linen as nn
import distrax
import numpy as np

from cl.agents.commons.distributions import OneHotDist, TwoHotDist, MSEDist, BinaryDist
from cl.agents.commons.initializer import Initializer
from cl.agents.commons.symlog import symlog
from cl.config.jax_config import COMPUTE_DTYPE


class MLPHead(nn.Module):
    """
    Multi-layer perceptron head with configurable output distributions.

    This unified implementation replaces separate Actor, Critic, and Predictor modules
    following the API style from the original DreamerV3.

    Args:
        output_shape: Output shape (e.g., (action_dim,) for actor, () for scalar rewards)
        layers: Number of MLP layers
        units: Hidden units per layer
        dist: Distribution type ('categorical', 'onehot', 'mse', 'symlog_mse',
              'symexp_twohot', 'binary')
        act: Activation function ('silu', 'elu', 'relu', 'gelu')
        norm: Normalization type ('rms', 'layer', 'none')
        outscale: Output layer initialization scale (0.0 for zeros, >0 for scaled init)
        unimix: Uniform mixture for exploration (categorical/onehot only)
        bins: Number of bins for twohot distribution
        minstd: Minimum stddev for normal distributions
        maxstd: Maximum stddev for normal distributions
        winit: Weight initializer (str or Initializer instance)
        binit: Bias initializer (str or Initializer instance)
    """

    output_shape: Tuple[int, ...]
    layers: int = 3
    units: int = 1024
    dist: str = 'mse'
    act: str = 'silu'
    norm: str = 'rms'
    outscale: float = 1.0
    unimix: float = 0.01
    bins: int = 255
    minstd: float = 0.1
    maxstd: float = 1.0
    winit: Union[str, Initializer] = 'trunc_normal_in'
    binit: Union[str, Initializer] = 'zeros'

    def setup(self):
        """Setup initializers."""
        # Parse initializers
        if isinstance(self.winit, str):
            if self.winit == 'trunc_normal_in':
                self.weight_init = Initializer(dist='trunc_normal', fan='in', scale=1.0)
            elif self.winit == 'trunc_normal_out':
                self.weight_init = Initializer(dist='trunc_normal', fan='out', scale=1.0)
            else:
                raise ValueError(f"Unknown winit string: {self.winit}")
        else:
            self.weight_init = self.winit

        if isinstance(self.binit, str):
            if self.binit == 'zeros':
                self.bias_init = Initializer(dist='zeros')
            else:
                raise ValueError(f"Unknown binit string: {self.binit}")
        else:
            self.bias_init = self.binit

        # Determine raw output dimension based on distribution
        output_size = int(np.prod(self.output_shape)) if self.output_shape else 1

        if self.dist == 'symexp_twohot':
            self.raw_output_size = output_size * self.bins
        elif self.dist in ['normal', 'normal_logstd']:
            self.raw_output_size = output_size * 2  # mean and stddev
        else:
            self.raw_output_size = output_size

        # Create output layer initializer based on outscale
        if self.outscale > 0:
            self.output_init = Initializer(dist='trunc_normal', fan='in', scale=self.outscale)
        else:
            self.output_init = Initializer(dist='zeros')

    @nn.compact
    def __call__(self, feat: jnp.ndarray, training: bool = True) -> Any:
        """
        Forward pass returning appropriate distribution.

        Args:
            feat: Input features of shape (..., feat_dim)
            training: Whether in training mode (affects some distributions)

        Returns:
            Distribution object based on configured type
        """
        x = feat

        # Pass through MLP layers
        for i in range(self.layers):
            x = nn.Dense(
                self.units,
                kernel_init=self.weight_init,
                bias_init=self.bias_init,
                dtype=COMPUTE_DTYPE,
                name=f'layer{i}_dense'
            )(x)

            # Apply normalization
            x = self._apply_norm(x, f'layer{i}_norm')

            # Apply activation
            x = self._apply_activation(x)

        # Output layer (no activation)
        x = nn.Dense(
            self.raw_output_size,
            kernel_init=self.output_init,
            bias_init=self.bias_init,
            dtype=COMPUTE_DTYPE,
            name='output'
        )(x)

        # Reshape if needed
        if self.output_shape:
            batch_shape = feat.shape[:-1]
            if self.dist == 'symexp_twohot':
                # For scalar outputs (1,), don't add extra dimension
                if self.output_shape == (1,):
                    x = x.reshape(*batch_shape, self.bins)
                else:
                    x = x.reshape(*batch_shape, *self.output_shape, self.bins)
            elif self.dist in ['normal', 'normal_logstd']:
                x = x.reshape(*batch_shape, *self.output_shape, 2)
            else:
                x = x.reshape(*batch_shape, *self.output_shape)

        # Return appropriate distribution
        if self.dist == 'categorical':
            # Apply uniform mixture for exploration
            if training and self.unimix > 0:
                uniform_logits = jnp.zeros_like(x)
                x = (1 - self.unimix) * x + self.unimix * uniform_logits

            return distrax.Categorical(logits=x)

        elif self.dist == 'onehot':
            # OneHotDist handles unimix internally based on training
            unimix = self.unimix if training else 0.0
            return OneHotDist(logits=x, unimix=unimix)

        elif self.dist == 'mse':
            # Squeeze last dimension if scalar output
            if not self.output_shape or self.output_shape == (1,):
                x = jnp.squeeze(x, axis=-1)
            return MSEDist(mean=x)

        elif self.dist == 'symlog_mse':
            # Squeeze last dimension if scalar output
            if not self.output_shape or self.output_shape == (1,):
                x = jnp.squeeze(x, axis=-1)
            # MSEDist will apply symlog transformation
            return MSEDist(mean=x, transform=symlog)

        elif self.dist == 'symexp_twohot':
            # TwoHotDist handles bin creation internally
            return TwoHotDist(logits=x, num_bins=self.bins)

        elif self.dist == 'binary':
            # Binary classification (e.g., continuation)
            # Squeeze last dimension if needed
            logits = jnp.squeeze(x, axis=-1) if x.shape[-1] == 1 else x
            return BinaryDist(logits=logits)

        elif self.dist == 'normal':
            # Normal with bounded stddev
            mean, stddev = jnp.split(x, 2, axis=-1)
            stddev = nn.softplus(stddev) + 0.01
            stddev = jnp.clip(stddev, self.minstd, self.maxstd)
            return distrax.MultivariateNormalDiag(loc=mean, scale_diag=stddev)

        elif self.dist == 'normal_logstd':
            # Normal with log stddev parameterization
            mean, log_stddev = jnp.split(x, 2, axis=-1)
            stddev = jnp.exp(log_stddev)
            stddev = jnp.clip(stddev, self.minstd, self.maxstd)
            return distrax.MultivariateNormalDiag(loc=mean, scale_diag=stddev)

        else:
            raise NotImplementedError(f"Unknown distribution: {self.dist}")

    def _apply_norm(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """Apply normalization."""
        if self.norm == 'rms':
            return nn.RMSNorm(name=name, dtype=COMPUTE_DTYPE)(x)
        elif self.norm == 'layer':
            return nn.LayerNorm(name=name, dtype=COMPUTE_DTYPE)(x)
        elif self.norm == 'none':
            return x
        else:
            raise ValueError(f"Unknown normalization: {self.norm}")

    def _apply_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply activation function."""
        if self.act == 'silu':
            return nn.silu(x)
        elif self.act == 'gelu':
            return nn.gelu(x)
        elif self.act == 'elu':
            return nn.elu(x)
        elif self.act == 'relu':
            return nn.relu(x)
        elif self.act == 'tanh':
            return nn.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.act}")
