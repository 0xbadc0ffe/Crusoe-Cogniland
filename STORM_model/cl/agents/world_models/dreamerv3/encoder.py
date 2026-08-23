"""Encoder networks for DreamerV3."""

from typing import Dict, Tuple, Union

import jax.numpy as jnp
import flax.linen as nn

from cl.agents.commons.initializer import Initializer
from cl.agents.commons.symlog import symlog
from cl.config.jax_config import COMPUTE_DTYPE

class Encoder(nn.Module):
    """
    Encoder for observations to latent embeddings.

    Handles both vector observations (e.g., state vectors) and image observations
    separately, following the DreamerV3 architecture.
    """

    units: int = 1024           # MLP hidden units
    norm: str = 'rms'           # Normalization type
    act: str = 'silu'           # Activation function
    depth: int = 96             # Base CNN depth (matches DreamerV3 original)
    mults: Tuple[int, ...] = (1, 2, 4, 8)  # Depth multipliers for CNN layers (96, 192, 384, 768)
    layers: int = 3             # Number of MLP layers for vector observations
    kernel: int = 5             # CNN kernel size
    symlog: bool = True         # Apply symlog to vector observations
    strided: bool = False       # Use strided convolutions instead of max pooling
    winit: Union[str, Initializer] = 'trunc_normal_in'  # Weight initialization
    binit: Union[str, Initializer] = 'zeros'  # Bias initialization

    @nn.compact
    def __call__(
        self,
        obs: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Encode observations to embeddings.

        Args:
            obs: Dictionary of observations where each key maps to either:
                - Vector observation: (batch, features) or (batch,)
                - Image observation: (batch, height, width, channels)

        Returns:
            Embeddings (batch, embed_dim)
        """
        # Parse initializers
        if isinstance(self.winit, str):
            if self.winit == 'trunc_normal_in':
                weight_init = Initializer(dist='trunc_normal', fan='in', scale=1.0)
            elif self.winit == 'trunc_normal_out':
                weight_init = Initializer(dist='trunc_normal', fan='out', scale=1.0)
            else:
                raise ValueError(f"Unknown winit: {self.winit}")
        else:
            weight_init = self.winit

        if isinstance(self.binit, str):
            if self.binit == 'zeros':
                bias_init = Initializer(dist='zeros')
            else:
                raise ValueError(f"Unknown binit: {self.binit}")
        else:
            bias_init = self.binit

        # Separate vector and image observations
        veckeys = [k for k, v in obs.items() if v.ndim <= 2]
        imgkeys = [k for k, v in obs.items() if v.ndim == 4]  # (batch, H, W, C)

        outs = []

        # Process vector observations
        if veckeys:
            vecs = [obs[k] for k in veckeys]
            # Flatten and concatenate all vector observations
            vecs_flat = []
            for v in vecs:
                if v.ndim == 1:
                    v = v[..., None]  # Add feature dimension if needed
                elif v.ndim == 2:
                    pass  # Already (batch, features)
                else:
                    v = v.reshape(v.shape[0], -1)
                vecs_flat.append(v)

            x = jnp.concatenate(vecs_flat, axis=-1)

            # Cast to compute dtype (bfloat16) - mixed precision
            x = x.astype(COMPUTE_DTYPE)

            # Apply symlog if enabled
            if self.symlog:
                x = symlog(x)

            # MLP layers
            for i in range(self.layers):
                x = nn.Dense(
                    self.units,
                    kernel_init=weight_init,
                    bias_init=bias_init,
                    dtype=COMPUTE_DTYPE
                )(x)
                x = self._apply_norm(x, f'mlp{i}norm')
                x = self._apply_activation(x)

            outs.append(x)

        # Process image observations
        if imgkeys:
            # Concatenate all image observations along channel dimension
            imgs = [obs[k] for k in sorted(imgkeys)]
            x = jnp.concatenate(imgs, axis=-1)

            # Normalize images from [0, 255] to [-0.5, 0.5] if uint8
            if x.dtype == jnp.uint8:
                x = x.astype(jnp.float32) / 255.0 - 0.5
            else:
                # Assume already normalized to [0, 1], shift to [-0.5, 0.5]
                x = x - 0.5

            # Cast to bfloat16 for computation
            x = x.astype(COMPUTE_DTYPE)

            # CNN layers with configured depths
            depths = tuple(self.depth * mult for mult in self.mults)

            for i, depth in enumerate(depths):
                if self.strided:
                    # Use strided convolution for downsampling (2x per layer)
                    # With 4 layers, this gives 16x total downsampling (64x64 → 4x4)
                    x = nn.Conv(
                        features=depth,
                        kernel_size=(self.kernel, self.kernel),
                        strides=(2, 2),  # Downsample on ALL layers (matches original)
                        padding='SAME',
                        use_bias=True,
                        kernel_init=weight_init,
                        bias_init=bias_init,
                        dtype=COMPUTE_DTYPE,
                    )(x)
                else:
                    # Use regular convolution followed by max pooling
                    x = nn.Conv(
                        features=depth,
                        kernel_size=(self.kernel, self.kernel),
                        strides=(1, 1),
                        padding='SAME',
                        use_bias=True,
                        kernel_init=weight_init,
                        bias_init=bias_init,
                        dtype=COMPUTE_DTYPE,
                    )(x)
                    # Max pool on ALL layers (matches original DreamerV3)
                    # With 4 layers, this gives 16x total downsampling (64x64 → 4x4)
                    B, H, W, C = x.shape
                    x = x.reshape(B, H // 2, 2, W // 2, 2, C)
                    x = x.max(axis=(2, 4))  # Max over the 2x2 windows

                x = self._apply_norm(x, f'cnn{i}norm')
                x = self._apply_activation(x)

            # Flatten spatial dimensions
            x = x.reshape(x.shape[0], -1)
            outs.append(x)

        # Concatenate all encoded features
        if len(outs) > 1:
            x = jnp.concatenate(outs, axis=-1)
        else:
            x = outs[0]

        return x

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
        else:
            raise ValueError(f"Unknown activation: {self.act}")
