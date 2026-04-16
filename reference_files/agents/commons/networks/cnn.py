"""Common network architectures for RL agents"""

from typing import Sequence, Optional, Tuple

import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import zeros

import numpy as np

from cl.agents.commons.utils import ortho_init


def adaptive_pool2d(
    x: jnp.ndarray,
    output_size: Tuple[int, int] = (6, 6),
    mode: str = "avg",
) -> jnp.ndarray:
    """
    JIT-safe adaptive pooling that outputs a fixed spatial size.

    Uses NumPy for bin edge computation to ensure Python ints at compile time,
    which is required for JAX slicing under JIT. The loops happen at trace time
    and compile into a single fused kernel.

    Args:
        x: Input tensor of shape (batch, height, width, channels)
        output_size: Target output spatial dimensions (out_h, out_w)
        mode: "avg" for average pooling (BTR default), "max" for max pooling

    Returns:
        Pooled tensor of shape (batch, out_h, out_w, channels)
    """

    _, H, W, _ = x.shape
    out_h, out_w = output_size

    # Bin edges as Python ints (static at JIT compile time)
    hs = np.linspace(0, H, out_h + 1, dtype=np.int32)
    ws = np.linspace(0, W, out_w + 1, dtype=np.int32)

    pool_fn = jnp.mean if mode == "avg" else jnp.max

    rows = []
    for i in range(out_h):
        h0, h1 = int(hs[i]), max(int(hs[i + 1]), int(hs[i]) + 1)
        row = []
        for j in range(out_w):
            w0, w1 = int(ws[j]), max(int(ws[j + 1]), int(ws[j]) + 1)
            row.append(pool_fn(x[:, h0:h1, w0:w1, :], axis=(1, 2)))
        rows.append(jnp.stack(row, axis=1))

    return jnp.stack(rows, axis=1)


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

    def _apply_norm(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """Apply normalization if specified."""
        if self.norm == 'rms':
            return nn.RMSNorm(name=name)(x)
        elif self.norm == 'layer':
            return nn.LayerNorm(name=name)(x)
        elif self.norm == 'none':
            return x
        else:
            raise ValueError(f"Unknown normalization: {self.norm}")


class ResidualBlockSN(nn.Module):
    """
    Residual block with Spectral Normalization on Conv layers.

    Per BTR paper (Appendix D): "Within each Impala CNN block, each residual layer
    (containing two Conv 3x3 + ReLU) has spectral normalization applied."
    """
    channels: int

    @nn.compact
    def __call__(self, x, training: bool = True):
        inputs = x
        x = nn.relu(x)
        # Apply SpectralNorm to Conv layers
        conv1 = nn.Conv(self.channels, kernel_size=(3, 3), name='conv1')
        x = nn.SpectralNorm(conv1, n_steps=1)(x, update_stats=training)
        x = nn.relu(x)
        conv2 = nn.Conv(self.channels, kernel_size=(3, 3), name='conv2')
        x = nn.SpectralNorm(conv2, n_steps=1)(x, update_stats=training)
        return x + inputs


class ConvSequence(nn.Module):
    """Convolutional sequence for IMPALA-CNN architecture"""
    channels: int
    norm: Optional[str] = 'none'

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = self._apply_norm(x, name="conv_norm")
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x

    def _apply_norm(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """Apply normalization if specified."""
        if self.norm == 'rms':
            return nn.RMSNorm(name=name)(x)
        elif self.norm == 'layer':
            return nn.LayerNorm(name=name)(x)
        elif self.norm == 'none':
            return x
        else:
            raise ValueError(f"Unknown normalization: {self.norm}")


class ConvSequenceSN(nn.Module):
    """Convolutional sequence with Spectral Normalization on residual blocks."""
    channels: int
    norm: Optional[str] = 'none'

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Initial conv (no spectral norm per BTR - only on residual blocks)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = self._apply_norm(x, name="conv_norm")
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        # Residual blocks with spectral norm
        x = ResidualBlockSN(self.channels)(x, training=training)
        x = ResidualBlockSN(self.channels)(x, training=training)
        return x

    def _apply_norm(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """Apply normalization if specified."""
        if self.norm == 'rms':
            return nn.RMSNorm(name=name)(x)
        elif self.norm == 'layer':
            return nn.LayerNorm(name=name)(x)
        elif self.norm == 'none':
            return x
        else:
            raise ValueError(f"Unknown normalization: {self.norm}")


class ImpalaCNN(nn.Module):
    """
    IMPALA-CNN architecture for visual RL.

    This is the standard IMPALA-CNN architecture from the Cleanba implementation.
    Note: Conv layers use default initialization (NOT orthogonal),
    only Dense layers use orthogonal initialization.

    Args:
        channels: Number of channels for each ConvSequence block (default: [16, 32, 32])
        hiddens: Hidden sizes for MLP head (default: [256])
        pool_out: Optional output size for adaptive maxpool after conv torso (e.g., (6, 6) for BTR).
                  If None, no adaptive pooling is applied.
        norm: Normalization type for MLP head: 'none', 'layer', or 'rms' (default: 'none')
    """
    channels: Sequence[int] = (16, 32, 32)
    hiddens: Sequence[int] = (256,)
    pool_out: Optional[Tuple[int, int]] = None
    norm: str = 'none'

    def _apply_norm(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """Apply normalization between linear layer and activation."""
        if self.norm == 'rms':
            return nn.RMSNorm(name=name)(x)
        elif self.norm == 'layer':
            return nn.LayerNorm(name=name)(x)
        elif self.norm == 'none':
            return x
        else:
            raise ValueError(f"Unknown normalization: {self.norm}")

    @nn.compact
    def __call__(self, x):
        # Input is expected in (batch, height, width, channels) format (JAX convention)
        # NOTE: Input should already be normalized to [0, 1] by preprocessing functions.
        # Do NOT normalize here to avoid double normalization.

        # Apply convolutional sequences
        for channels in self.channels:
            x = ConvSequence(channels, norm=self.norm)(x)

        x = nn.relu(x)

        # Optional adaptive pooling (BTR uses 6x6 avg pool)
        # Note: if enabled, MLP head is skipped per Figure E.7 of BTR
        if self.pool_out is not None:
            x = adaptive_pool2d(x, self.pool_out, mode="avg")
            return x.reshape((x.shape[0], -1))

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # MLP head with custom orthogonal initialization
        for i, hidden in enumerate(self.hiddens):
            x = nn.Dense(hidden, kernel_init=ortho_init, bias_init=zeros)(x)
            x = self._apply_norm(x, name=f'norm_{i}')
            x = nn.relu(x)

        return x


class ImpalaCNNSN(nn.Module):
    """
    IMPALA-CNN architecture with Spectral Normalization on residual block convolutions.

    Per BTR paper (Appendix D): SpectralNorm is applied to the Conv layers
    within each residual block, not to the initial Conv or Dense layers.

    Args:
        channels: Number of channels for each ConvSequence block (default: [16, 32, 32])
        hiddens: Hidden sizes for MLP head (default: [256])
        pool_out: Optional output size for adaptive maxpool after conv torso (e.g., (6, 6) for BTR).
                  If None, no adaptive pooling is applied.
        norm: Normalization type for MLP head: 'none', 'layer', or 'rms' (default: 'none')
    """
    channels: Sequence[int] = (16, 32, 32)
    hiddens: Sequence[int] = (256,)
    pool_out: Optional[Tuple[int, int]] = None
    norm: str = 'none'

    def _apply_norm(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """Apply normalization between linear layer and activation."""
        if self.norm == 'rms':
            return nn.RMSNorm(name=name)(x)
        elif self.norm == 'layer':
            return nn.LayerNorm(name=name)(x)
        elif self.norm == 'none':
            return x
        else:
            raise ValueError(f"Unknown normalization: {self.norm}")

    @nn.compact
    def __call__(self, x, training: bool = True):
        # NOTE: Input should already be normalized to [0, 1] by preprocessing functions.
        # Do NOT normalize here to avoid double normalization.

        # Apply convolutional sequences with spectral norm
        for channels in self.channels:
            x = ConvSequenceSN(channels, norm=self.norm)(x, training=training)

        x = nn.relu(x)

        # Optional adaptive pooling (BTR uses 6x6 avg pool)
        # Note: if enabled, MLP head is skipped per Figure E.7 of BTR
        if self.pool_out is not None:
            x = adaptive_pool2d(x, self.pool_out, mode="avg")
            return x.reshape((x.shape[0], -1))

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # MLP head
        for i, hidden in enumerate(self.hiddens):
            x = nn.Dense(hidden, kernel_init=ortho_init, bias_init=zeros)(x)
            x = self._apply_norm(x, name=f'norm_{i}')
            x = nn.relu(x)

        return x
