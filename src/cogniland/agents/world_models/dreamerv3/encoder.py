"""Encoder networks for DreamerV3."""

from typing import Dict, Tuple, Union

import jax.numpy as jnp
import flax.linen as nn

from cogniland.agents.commons.initializer import Initializer
from cogniland.agents.commons.symlog import symlog
from cogniland.config.jax_config import COMPUTE_DTYPE

class Encoder(nn.Module):
    """
    Encoder for observations to latent embeddings.

    Handles both vector observations (e.g., state vectors) and image observations
    separately, following the DreamerV3 architecture.

    Supports:
    - Vector observations (ndim <= 2): MLP processing
    - Image observations (ndim == 4): CNN processing
    - Symbolic observations (if in flatten_keys): Flatten + MLP
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
    strides: Tuple[int, ...] = None  # Per-layer strides, defaults to (2,)*len(mults). Use (1,1,1,1) for no downsampling.
    winit: Union[str, Initializer] = 'trunc_normal_in'  # Weight initialization
    binit: Union[str, Initializer] = 'zeros'  # Bias initialization
    flatten_keys: Tuple[str, ...] = ()  # Observation keys to flatten and process with MLP
    discrete_keys: Tuple[str, ...] = ()  # Image keys that are discrete/one-hot (skip -0.5 normalization for CNN)

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
        # flatten_keys are treated as vectors even if they have image-like shape
        veckeys = []
        imgkeys = []
        for k, v in obs.items():
            if k in self.flatten_keys:
                # Force flatten even if 4D (for symbolic observations)
                veckeys.append(k)
            elif v.ndim <= 2:
                veckeys.append(k)
            elif v.ndim == 4:  # (batch, H, W, C)
                imgkeys.append(k)

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

            # Check if all image keys are discrete (one-hot encoded)
            all_discrete = all(k in self.discrete_keys for k in imgkeys)

            # Normalize images: skip -0.5 shift for discrete/one-hot inputs
            if x.dtype == jnp.uint8:
                x = x.astype(jnp.float32) / 255.0
                if not all_discrete:
                    x = x - 0.5  # Center continuous images
            elif not all_discrete:
                # Only shift continuous images from [0, 1] to [-0.5, 0.5]
                x = x - 0.5

            # Cast to bfloat16 for computation
            x = x.astype(COMPUTE_DTYPE)

            # CNN layers with configured depths
            depths = tuple(self.depth * mult for mult in self.mults)

            # Determine per-layer strides (default to stride-2 for backward compatibility)
            if self.strides is None:
                strides = (2,) * len(self.mults)
            else:
                assert len(self.strides) == len(self.mults), \
                    f"strides length {len(self.strides)} must match mults length {len(self.mults)}"
                strides = self.strides

            for i, (depth, stride) in enumerate(zip(depths, strides)):
                if self.strided:
                    # Use strided convolution for downsampling
                    x = nn.Conv(
                        features=depth,
                        kernel_size=(self.kernel, self.kernel),
                        strides=(stride, stride),  # Per-layer stride
                        padding='SAME',
                        use_bias=True,
                        kernel_init=weight_init,
                        bias_init=bias_init,
                        dtype=COMPUTE_DTYPE,
                    )(x)
                else:
                    # Use regular convolution followed by optional pooling
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
                    # Apply max pooling only if stride > 1
                    if stride == 2:
                        B, H, W, C = x.shape
                        x = x.reshape(B, H // 2, 2, W // 2, 2, C)
                        x = x.max(axis=(2, 4))  # Max over the 2x2 windows
                    # stride == 1 means no pooling (preserve spatial dimensions)

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
