"""STORM decoder with multi-modal support.

Multi-modal decoder supporting both vector observations (MLP) and image
observations (CNN). Follows the original STORM implementation with BatchNorm
for CNN layers.

Key difference from DreamerV3: Takes ONLY the stochastic latent as input,
not the concatenation of deterministic + stochastic states.
"""

import math
from typing import Any, Dict, Tuple, Union

import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from cogniland.agents.commons.distributions import MSEDist, OneHotDist
from cogniland.config.jax_config import COMPUTE_DTYPE


class STORMDecoder(nn.Module):
    """Multi-modal decoder with BatchNorm for STORM.

    Reconstructs observations from stochastic latents. Supports:
    - Vector observations (len(shape) <= 2): MLP reconstruction
    - Image observations (len(shape) == 3): CNN reconstruction
    - Discrete observations (in discrete_keys): OneHotDist output
    - Flattened observations (in flatten_keys): MLP reconstruction

    Args:
        obs_shapes: Dictionary mapping observation keys to their shapes
        stoch_dim: Flattened stochastic dimension (stoch * classes)
        stem_channels: Number of channels at penultimate layer for CNN
        final_feature_width: Spatial resolution at smallest layer (default: 4)
        image_size: Output image size (default: 64)
        mlp_units: Hidden units for MLP decoder (default: 512)
        mlp_layers: Number of MLP layers (default: 2)
        discrete_keys: Keys for categorical observations using OneHotDist (default: ('direction',))
        flatten_keys: Keys that were flattened in encoder (default: ())
        out_channels: DEPRECATED - kept for backward compatibility, use obs_shapes instead
        mults: Channel multipliers per layer, optional. If provided, uses fixed layer count.
        strides: Per-layer strides, optional. Must match encoder strides.
    """
    obs_shapes: Dict[str, Tuple[int, ...]] = None
    stoch_dim: int = 32 * 32  # stoch * classes
    stem_channels: int = 32
    final_feature_width: int = 4
    image_size: int = 64
    mlp_units: int = 512
    mlp_layers: int = 2
    kernel: int = 4              # CNN kernel size (must match encoder)
    discrete_keys: Tuple[str, ...] = ('direction',)
    flatten_keys: Tuple[str, ...] = ()
    out_channels: int = 3  # Deprecated, kept for backward compatibility
    mults: Tuple[int, ...] = None  # Channel multipliers per layer (e.g., (1, 2, 4, 8))
    strides: Tuple[int, ...] = None  # Per-layer strides (must match encoder)

    @nn.compact
    def __call__(
        self,
        stoch: jnp.ndarray,
        *,
        training: bool = False
    ) -> Union[MSEDist, Dict[str, Any]]:
        """Decode stochastic latents to observation distributions.

        Args:
            stoch: Flattened stochastic latents [B, stoch*classes]
            training: Whether in training mode (affects BatchNorm)

        Returns:
            Either:
            - MSEDist for single image (backward compatible if obs_shapes is None)
            - Dict of distributions for multi-modal observations
        """
        batch_size = stoch.shape[0]

        # Backward compatibility: if no obs_shapes, assume single image output
        if self.obs_shapes is None:
            return self._decode_image(stoch, training=training)

        # Separate observations by type
        veckeys = []
        imgkeys = []

        for k, s in self.obs_shapes.items():
            if k in self.flatten_keys:
                # Flattened observations use MLP
                veckeys.append(k)
            elif len(s) <= 2:
                veckeys.append(k)
            elif len(s) == 3:
                imgkeys.append(k)

        recons = {}

        # Decode vector observations with MLP
        if veckeys:
            x = stoch.astype(COMPUTE_DTYPE)

            # Shared MLP layers
            for i in range(self.mlp_layers):
                x = nn.Dense(
                    self.mlp_units,
                    dtype=COMPUTE_DTYPE,
                    name=f'vec_mlp_dense_{i}'
                )(x)
                x = nn.BatchNorm(
                    use_running_average=not training,
                    dtype=COMPUTE_DTYPE,
                    name=f'vec_mlp_bn_{i}'
                )(x)
                x = nn.relu(x)

            # Output heads for each vector observation
            for key in veckeys:
                shape = self.obs_shapes[key]
                output_size = int(np.prod(shape))

                if key in self.discrete_keys:
                    # OneHotDist for categorical observations
                    logits = nn.Dense(
                        output_size,
                        dtype=COMPUTE_DTYPE,
                        name=f'vec_out_{key}'
                    )(x)

                    if len(shape) > 0:
                        logits = logits.reshape((batch_size,) + shape)

                    # Cast to float32 for distribution
                    logits = logits.astype(jnp.float32)
                    recons[key] = OneHotDist(logits, unimix=0.01)
                else:
                    # MSEDist for continuous observations
                    mean = nn.Dense(
                        output_size,
                        dtype=COMPUTE_DTYPE,
                        name=f'vec_out_{key}'
                    )(x)

                    if len(shape) > 0:
                        mean = mean.reshape((batch_size,) + shape)

                    # Cast to float32 for distribution
                    mean = mean.astype(jnp.float32)
                    recons[key] = MSEDist(mean)

        # Decode image observations with CNN
        if imgkeys:
            # For simplicity, decode one shared image and assign to all imgkeys
            # (typically there's only one 'image' key)
            image_tensor = self._decode_image(stoch, training=training, return_tensor=True)
            for key in imgkeys:
                if key in self.discrete_keys:
                    # OneHotDist for categorical images (e.g., 7x7x10 with 10 entity classes)
                    recons[key] = OneHotDist(image_tensor, unimix=0.01)
                else:
                    # MSEDist for continuous images (e.g., RGB)
                    recons[key] = MSEDist(image_tensor)

        return recons

    def _decode_image(
        self,
        stoch: jnp.ndarray,
        *,
        training: bool = False,
        return_tensor: bool = False
    ) -> Union[MSEDist, jnp.ndarray]:
        """Decode stochastic latents to image distribution using CNN.

        Args:
            stoch: Flattened stochastic latents [B, stoch*classes]
            training: Whether in training mode (affects BatchNorm)
            return_tensor: If True, return raw tensor instead of MSEDist

        Returns:
            MSEDist over reconstructed images [B, H, W, C] or raw tensor if return_tensor=True
        """
        batch_size = stoch.shape[0]
        x = stoch.astype(COMPUTE_DTYPE)

        # Determine output channels from obs_shapes if available
        out_channels = 3  # Default RGB
        if self.obs_shapes is not None:
            for key, shape in self.obs_shapes.items():
                if len(shape) == 3 and key not in self.flatten_keys:
                    out_channels = shape[-1]
                    break

        # Use mults-based architecture if mults is provided (matches DreamerV3 style)
        if self.mults is not None:
            # Determine per-layer strides (must match encoder)
            if self.strides is None:
                strides = (2,) * len(self.mults)
            else:
                assert len(self.strides) == len(self.mults), \
                    f"strides length {len(self.strides)} must match mults length {len(self.mults)}"
                strides = self.strides

            # Calculate depths from mults
            depths = tuple(self.stem_channels * mult for mult in self.mults)

            # Calculate bottleneck spatial size from strides
            factor = 1
            for s in strides:
                factor *= s
            min_h = min_w = self.image_size // factor

            # Linear projection to spatial features at bottleneck
            last_channels = depths[-1]
            spatial_dim = min_h * min_w * last_channels
            x = nn.Dense(
                features=spatial_dim,
                use_bias=False,
                dtype=COMPUTE_DTYPE,
                name='cnn_proj'
            )(x)

            # Reshape to spatial format [B, H, W, C]
            x = x.reshape(batch_size, min_h, min_w, last_channels)

            x = nn.BatchNorm(
                use_running_average=not training,
                dtype=COMPUTE_DTYPE,
                name='cnn_proj_bn'
            )(x)
            x = nn.relu(x)

            # Upsample through transposed convolutions (reversed order)
            # Skip the last depth (already at bottleneck) and first stride (for final layer)
            for i, (depth, stride) in reversed(list(enumerate(zip(depths[:-1], strides[:-1])))):
                x = nn.ConvTranspose(
                    features=depth,
                    kernel_size=(self.kernel, self.kernel),
                    strides=(stride, stride),  # Per-layer stride
                    padding='SAME',
                    use_bias=False,
                    dtype=COMPUTE_DTYPE,
                    name=f'cnn_deconv_{i}'
                )(x)
                x = nn.BatchNorm(
                    use_running_average=not training,
                    dtype=COMPUTE_DTYPE,
                    name=f'cnn_deconv_bn_{i}'
                )(x)
                x = nn.relu(x)

            # Final transposed conv to output channels using first stride
            final_stride = strides[0]
            x = nn.ConvTranspose(
                features=out_channels,
                kernel_size=(4, 4),
                strides=(final_stride, final_stride),
                padding='SAME',
                dtype=COMPUTE_DTYPE,
                name='cnn_out'
            )(x)
        else:
            # Original dynamic architecture based on image_size and final_feature_width
            # Calculate number of upsampling layers needed to reach image_size
            num_upsample_layers = int(math.log2(self.image_size / self.final_feature_width))

            # Calculate channels at each layer (doubles each time going backwards from stem)
            last_channels = self.stem_channels * (2 ** (num_upsample_layers - 1))

            # Linear projection to spatial features
            spatial_dim = last_channels * self.final_feature_width * self.final_feature_width
            x = nn.Dense(
                features=spatial_dim,
                use_bias=False,
                dtype=COMPUTE_DTYPE,
                name='cnn_proj'
            )(x)

            # Reshape to spatial format [B, H, W, C]
            x = x.reshape(
                batch_size,
                self.final_feature_width,
                self.final_feature_width,
                last_channels
            )

            x = nn.BatchNorm(
                use_running_average=not training,
                dtype=COMPUTE_DTYPE,
                name='cnn_proj_bn'
            )(x)
            x = nn.relu(x)

            # Transposed convolutions to upsample
            channels = last_channels
            layer_idx = 0

            while channels > self.stem_channels:
                x = nn.ConvTranspose(
                    features=channels // 2,
                    kernel_size=(self.kernel, self.kernel),
                    strides=(2, 2),
                    padding='SAME',
                    use_bias=False,
                    dtype=COMPUTE_DTYPE,
                    name=f'cnn_deconv_{layer_idx}'
                )(x)
                channels //= 2
                x = nn.BatchNorm(
                    use_running_average=not training,
                    dtype=COMPUTE_DTYPE,
                    name=f'cnn_deconv_bn_{layer_idx}'
                )(x)
                x = nn.relu(x)
                layer_idx += 1

            # Final transposed conv to output channels
            x = nn.ConvTranspose(
                features=out_channels,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='SAME',
                dtype=COMPUTE_DTYPE,
                name='cnn_out'
            )(x)

        # Cast to float32 for distribution
        x = x.astype(jnp.float32)

        # Return raw tensor or MSEDist based on flag
        if return_tensor:
            return x
        return MSEDist(x)
