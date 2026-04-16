"""Decoder networks for DreamerV3."""

from typing import Dict, Tuple, Union

import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import distrax
import einops

from cogniland.agents.commons.initializer import Initializer
from cogniland.agents.commons.distributions import MSEDist, OneHotDist
from cogniland.agents.commons.networks.blocks import BlockLinear
from cogniland.agents.commons.symlog import symexp
from cogniland.config.jax_config import COMPUTE_DTYPE

class Decoder(nn.Module):
    """
    Decoder for reconstructing observations from latent states.

    Handles both vector observations and image observations separately,
    following the DreamerV3 architecture.

    Supports:
    - Vector observations (len(shape) <= 2): MLP reconstruction
    - Image observations (len(shape) == 3): CNN reconstruction
    - Discrete observations (in discrete_keys): OneHotDist output
    - Flattened observations (in flatten_keys): MLP reconstruction
    """

    obs_shapes: Dict[str, Tuple[int, ...]]  # Shape of each observation
    units: int = 1024           # MLP hidden units
    norm: str = 'rms'           # Normalization type
    act: str = 'silu'           # Activation function
    outscale: float = 1.0       # Output scaling
    depth: int = 96             # Base CNN depth (matches DreamerV3 original)
    mults: Tuple[int, ...] = (1, 2, 4, 8)  # Depth multipliers for CNN layers (96, 192, 384, 768)
    layers: int = 3             # Number of MLP layers for vector observations
    kernel: int = 5             # CNN kernel size
    symlog: bool = True         # Use symlog for vector observations
    strided: bool = False       # Use strided transposed convolutions
    strides: Tuple[int, ...] = None  # Per-layer strides, must match encoder. Defaults to (2,)*len(mults).
    bspace: int = 8             # Block space for efficient decoder projection
    winit: Union[str, Initializer] = 'trunc_normal_in'  # Weight initialization
    binit: Union[str, Initializer] = 'zeros'  # Bias initialization
    discrete_keys: Tuple[str, ...] = ()  # Keys for one-hot categorical observations (e.g., 'direction')
    flatten_keys: Tuple[str, ...] = ()  # Keys that were flattened in encoder

    @nn.compact
    def __call__(
        self,
        feat: Dict[str, jnp.ndarray]
    ) -> Dict[str, Union[MSEDist, distrax.Distribution]]:
        """
        Decode features to observation distributions.

        Args:
            feat: Dictionary with 'deter' and 'stoch' keys from RSSM
                  - 'deter': (batch, deter_size) deterministic state
                  - 'stoch': (batch, stoch_size, classes) or (batch, stoch_size * classes) stochastic state

        Returns:
            Dictionary of distributions over observations
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

        # Output initializer (scaled or zeros)
        if self.outscale > 0:
            output_init = Initializer(dist='trunc_normal', fan='in', scale=self.outscale)
        else:
            output_init = Initializer(dist='zeros')

        # Extract deter and stoch from feature dictionary
        feat_deter = feat['deter']  # (batch, deter_size)
        feat_stoch = feat['stoch']  # (batch, stoch_size, classes) or (batch, stoch_size * classes)

        # Flatten stochastic state if needed
        if feat_stoch.ndim > 2:
            feat_stoch = feat_stoch.reshape(feat_stoch.shape[0], -1)

        # Concatenate for vector observations
        feat_concat = jnp.concatenate([feat_deter, feat_stoch], axis=-1)

        # Separate vector and image observation shapes
        # flatten_keys are treated as vectors for reconstruction
        veckeys = []
        imgkeys = []
        for k, s in self.obs_shapes.items():
            if k in self.flatten_keys:
                # Flattened observations use MLP reconstruction
                veckeys.append(k)
            elif len(s) <= 2:
                veckeys.append(k)
            elif len(s) == 3:
                imgkeys.append(k)

        recons = {}

        # Process vector observations
        if veckeys:
            # MLP to project features
            x = feat_concat
            for i in range(self.layers):
                x = nn.Dense(
                    self.units,
                    kernel_init=weight_init,
                    bias_init=bias_init,
                    dtype=COMPUTE_DTYPE
                )(x)
                x = self._apply_norm(x, f'mlp{i}norm')
                x = self._apply_activation(x)

            # Create distributions for each vector observation
            for key in veckeys:
                shape = self.obs_shapes[key]
                # Project to observation size
                # Use int(np.prod) to compute static shape (not traced)
                output_size = int(np.prod(shape))

                if key in self.discrete_keys:
                    # Use OneHotDist for discrete categorical observations (e.g., direction)
                    logits = nn.Dense(
                        output_size,
                        kernel_init=output_init,
                        bias_init=bias_init,
                        dtype=COMPUTE_DTYPE,
                    )(x)

                    if len(shape) > 0:
                        logits = logits.reshape((feat_concat.shape[0],) + shape)

                    # OneHotDist for categorical with straight-through gradients
                    recons[key] = OneHotDist(logits, unimix=0.01)
                else:
                    # Use MSE distribution for continuous values
                    mean = nn.Dense(
                        output_size,
                        kernel_init=output_init,
                        bias_init=bias_init,
                        dtype=COMPUTE_DTYPE,
                    )(x)

                    if len(shape) > 0:
                        mean = mean.reshape((feat_concat.shape[0],) + shape)

                    # Apply symexp if symlog was used during encoding
                    if self.symlog:
                        mean = symexp(mean)

                    recons[key] = MSEDist(mean)

        # Process image observations
        if imgkeys:
            # Determine per-layer strides (must match encoder)
            if self.strides is None:
                strides = (2,) * len(self.mults)
            else:
                assert len(self.strides) == len(self.mults), \
                    f"strides length {len(self.strides)} must match mults length {len(self.mults)}"
                strides = self.strides

            # Determine spatial dimensions after CNN encoding
            # factor = product of all strides
            sample_shape = self.obs_shapes[imgkeys[0]]  # Use first image shape
            H, W = sample_shape[:2]
            factor = 1
            for s in strides:
                factor *= s
            min_h, min_w = H // factor, W // factor

            # Project features to initial spatial dimensions
            depths = tuple(self.depth * mult for mult in self.mults)
            output_size = min_h * min_w * depths[-1]

            # Use block-based projection for efficiency (following original DreamerV3)
            batch_size = feat_deter.shape[0]

            # Project deter using BlockLinear (87.5% parameter reduction)
            x0 = BlockLinear(
                output_size,
                self.bspace,
                kernel_init=weight_init,
                bias_init=bias_init,
                name='sp0'
            )(feat_deter)
            x0 = einops.rearrange(
                x0, 'b (h w c) -> b h w c',
                h=min_h, w=min_w
            )

            # Project stoch using regular MLP
            x1 = nn.Dense(
                2 * self.units,
                kernel_init=weight_init,
                bias_init=bias_init,
                dtype=COMPUTE_DTYPE
            )(feat_stoch)
            x1 = self._apply_activation(self._apply_norm(x1, 'sp1norm'))
            x1 = nn.Dense(
                output_size,
                kernel_init=weight_init,
                bias_init=bias_init,
                dtype=COMPUTE_DTYPE
            )(x1)
            x1 = x1.reshape(batch_size, min_h, min_w, depths[-1])

            # Combine both projections
            x = self._apply_activation(self._apply_norm(x0 + x1, 'spnorm'))

            # Upsample through transposed convolutions or resize + conv
            # Note: we reverse through all but the last layer (depths[:-1])
            # and also reverse through corresponding strides (strides[:-1])
            for i, (depth, stride) in reversed(list(enumerate(zip(depths[:-1], strides[:-1])))):
                if self.strided:
                    # Use strided transposed convolution
                    x = nn.ConvTranspose(
                        features=depth,
                        kernel_size=(self.kernel, self.kernel),
                        strides=(stride, stride),  # Per-layer stride
                        padding='SAME',
                        kernel_init=weight_init,
                        bias_init=bias_init,
                        dtype=COMPUTE_DTYPE,
                    )(x)
                else:
                    # Upsample only if stride > 1
                    if stride == 2:
                        x = jnp.repeat(x, 2, axis=1)  # Height
                        x = jnp.repeat(x, 2, axis=2)  # Width
                    # stride == 1 means no upsampling
                    x = nn.Conv(
                        features=depth,
                        kernel_size=(self.kernel, self.kernel),
                        strides=(1, 1),
                        padding='SAME',
                        kernel_init=weight_init,
                        bias_init=bias_init,
                        dtype=COMPUTE_DTYPE,
                    )(x)

                x = self._apply_norm(x, f'conv{i}norm')
                x = self._apply_activation(x)

            # Final layer to predict each image observation
            # Use the first stride (strides[0]) for the final layer
            final_stride = strides[0]
            for key in imgkeys:
                shape = self.obs_shapes[key]
                channels = shape[-1]

                # Output layer
                if self.strided:
                    mean = nn.ConvTranspose(
                        features=channels,
                        kernel_size=(self.kernel, self.kernel),
                        strides=(final_stride, final_stride),  # Per-layer stride
                        padding='SAME',
                        kernel_init=output_init,
                        bias_init=bias_init,
                        dtype=COMPUTE_DTYPE,
                    )(x)
                else:
                    # Final upsample only if stride > 1
                    if final_stride == 2:
                        x_key = jnp.repeat(x, 2, axis=1)
                        x_key = jnp.repeat(x_key, 2, axis=2)
                    else:
                        x_key = x
                    mean = nn.Conv(
                        features=channels,
                        kernel_size=(self.kernel, self.kernel),
                        strides=(1, 1),
                        padding='SAME',
                        kernel_init=output_init,
                        bias_init=bias_init,
                        dtype=COMPUTE_DTYPE,
                    )(x_key)

                # Check if this image key should use discrete distribution
                if key in self.discrete_keys:
                    # OneHotDist for categorical images (e.g., 7x7x11 with 11 entity classes)
                    # No sigmoid - output raw logits
                    recons[key] = OneHotDist(mean, unimix=0.01)
                else:
                    # MSEDist for continuous images (e.g., RGB)
                    # Use sigmoid for [0, 1] range
                    mean = nn.sigmoid(mean)
                    recons[key] = MSEDist(mean)

        return recons

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
