"""Common network architectures for RL agents"""

from typing import Sequence

import flax.linen as nn
from jax.nn.initializers import zeros

from cl.agents.commons.utils import ortho_init


class ResidualBlock(nn.Module):
    """Residual block for IMPALA-CNN architecture"""
    channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        return x + inputs


class ConvSequence(nn.Module):
    """Convolutional sequence for IMPALA-CNN architecture"""
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x


class ImpalaCNN(nn.Module):
    """
    IMPALA-CNN architecture for visual RL.

    This is the standard IMPALA-CNN architecture from the Cleanba implementation.
    Note: Conv layers use default initialization (NOT orthogonal),
    only Dense layers use orthogonal initialization.

    Args:
        channels: Number of channels for each ConvSequence block (default: [16, 32, 32])
        hiddens: Hidden sizes for MLP head (default: [256])
    """
    channels: Sequence[int] = (16, 32, 32)
    hiddens: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, x):
        # Input preprocessing: normalize to [0, 1]
        # Note: Input is expected in (batch, height, width, channels) format (JAX convention)
        x = x / 255.0

        # Apply convolutional sequences
        for channels in self.channels:
            x = ConvSequence(channels)(x)

        x = nn.relu(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # MLP head with custom orthogonal initialization
        for hidden in self.hiddens:
            x = nn.Dense(hidden, kernel_init=ortho_init, bias_init=zeros)(x)
            x = nn.relu(x)

        return x
