"""STORM encoder with multi-modal support.

Multi-modal encoder supporting both vector observations (MLP) and image
observations (CNN). Follows the original STORM implementation with BatchNorm
for CNN layers.
"""

from typing import Dict, Tuple, Union

import jax.numpy as jnp
import flax.linen as nn

from cl.config.jax_config import COMPUTE_DTYPE


class STORMEncoder(nn.Module):
    """Multi-modal encoder with BatchNorm for STORM.

    Supports:
    - Vector observations (ndim <= 2): MLP processing
    - Image observations (ndim == 4): CNN processing with strided convolutions
    - Symbolic observations (if in flatten_keys): Flatten + MLP

    Args:
        stem_channels: Initial number of channels for CNN (default: 32)
        final_feature_width: Target spatial resolution for CNN (default: 4)
        image_size: Input image size for CNN (default: 64)
        mlp_units: Hidden units for MLP layers (default: 512)
        mlp_layers: Number of MLP layers for vector observations (default: 2)
        flatten_keys: Observation keys to flatten and process with MLP (default: ())
        mults: Channel multipliers per layer, optional. If provided, uses fixed layer count.
        strides: Per-layer strides, optional. If provided with mults, uses custom strides.
    """
    stem_channels: int = 32
    final_feature_width: int = 4
    image_size: int = 64
    mlp_units: int = 512
    mlp_layers: int = 2
    kernel: int = 4              # CNN kernel size
    flatten_keys: Tuple[str, ...] = ()
    mults: Tuple[int, ...] = None  # Channel multipliers per layer (e.g., (1, 2, 4, 8))
    strides: Tuple[int, ...] = None  # Per-layer strides (e.g., (1, 1, 1, 1) for no downsampling)

    @nn.compact
    def __call__(
        self,
        obs: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
        *,
        training: bool = False
    ) -> jnp.ndarray:
        """Encode observations to embeddings.

        Args:
            obs: Either:
                - Single image array [B, H, W, C] (backward compatible)
                - Dictionary of observations where each key maps to either:
                    - Vector observation: (batch, features) or (batch,)
                    - Image observation: (batch, height, width, channels)
            training: Whether in training mode (affects BatchNorm)

        Returns:
            Flattened features [B, feature_dim]
        """
        # Handle backward compatibility: single array input
        if isinstance(obs, jnp.ndarray):
            return self._encode_image(obs, training=training)

        # Multi-modal processing
        # Separate observations by type
        veckeys = []
        imgkeys = []

        for k, v in obs.items():
            if k in self.flatten_keys:
                # Force flatten even if 4D (for symbolic observations)
                veckeys.append(k)
            elif v.ndim <= 2:
                veckeys.append(k)
            elif v.ndim == 4:
                imgkeys.append(k)

        outs = []

        # Process vector observations with MLP
        if veckeys:
            vecs = []
            for k in veckeys:
                v = obs[k]
                # Flatten if needed (for symbolic observations with shape like [B, H, W, C])
                if v.ndim > 2:
                    v = v.reshape(v.shape[0], -1)
                elif v.ndim == 1:
                    v = v[..., None]
                vecs.append(v)

            x = jnp.concatenate(vecs, axis=-1)
            x = x.astype(COMPUTE_DTYPE)

            # MLP layers
            for i in range(self.mlp_layers):
                x = nn.Dense(
                    self.mlp_units,
                    dtype=COMPUTE_DTYPE,
                    name=f'mlp_dense_{i}'
                )(x)
                x = nn.BatchNorm(
                    use_running_average=not training,
                    dtype=COMPUTE_DTYPE,
                    name=f'mlp_bn_{i}'
                )(x)
                x = nn.relu(x)

            outs.append(x)

        # Process image observations with CNN
        if imgkeys:
            # Concatenate all images along channel dimension
            imgs = [obs[k] for k in sorted(imgkeys)]
            img = jnp.concatenate(imgs, axis=-1)
            cnn_out = self._encode_image(img, training=training)
            outs.append(cnn_out)

        # Concatenate all encoded features
        if len(outs) > 1:
            return jnp.concatenate(outs, axis=-1)
        elif len(outs) == 1:
            return outs[0]
        else:
            raise ValueError("No observations to encode")

    def _encode_image(self, obs: jnp.ndarray, *, training: bool = False) -> jnp.ndarray:
        """Encode image observations using CNN.

        Args:
            obs: Image observations [B, H, W, C] (uint8 or float32)
            training: Whether in training mode (affects BatchNorm)

        Returns:
            Flattened features [B, feature_dim]
        """
        # Normalize uint8 images to [0, 1]
        if obs.dtype == jnp.uint8:
            x = obs.astype(jnp.float32) / 255.0
        else:
            x = obs

        x = x.astype(COMPUTE_DTYPE)

        # Use mults-based architecture if mults is provided (matches DreamerV3 style)
        if self.mults is not None:
            # Determine per-layer strides
            if self.strides is None:
                strides = (2,) * len(self.mults)
            else:
                assert len(self.strides) == len(self.mults), \
                    f"strides length {len(self.strides)} must match mults length {len(self.mults)}"
                strides = self.strides

            # Calculate depths from mults
            depths = tuple(self.stem_channels * mult for mult in self.mults)

            for i, (depth, stride) in enumerate(zip(depths, strides)):
                x = nn.Conv(
                    features=depth,
                    kernel_size=(self.kernel, self.kernel),
                    strides=(stride, stride),  # Per-layer stride
                    padding='SAME',
                    use_bias=False,
                    dtype=COMPUTE_DTYPE,
                    name=f'cnn_conv_{i}'
                )(x)
                x = nn.BatchNorm(
                    use_running_average=not training,
                    dtype=COMPUTE_DTYPE,
                    name=f'cnn_bn_{i}'
                )(x)
                x = nn.relu(x)
        else:
            # Original dynamic architecture based on image_size and final_feature_width
            # Stem: first conv layer
            x = nn.Conv(
                features=self.stem_channels,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='SAME',
                use_bias=False,
                dtype=COMPUTE_DTYPE,
                name='cnn_stem'
            )(x)
            x = nn.BatchNorm(
                use_running_average=not training,
                dtype=COMPUTE_DTYPE,
                name='cnn_stem_bn'
            )(x)
            x = nn.relu(x)

            # Subsequent layers: double channels and halve spatial resolution
            channels = self.stem_channels
            current_width = self.image_size // 2  # After first stride-2 conv

            layer_idx = 0
            while current_width > self.final_feature_width:
                x = nn.Conv(
                    features=channels * 2,
                    kernel_size=(self.kernel, self.kernel),
                    strides=(2, 2),
                    padding='SAME',
                    use_bias=False,
                    dtype=COMPUTE_DTYPE,
                    name=f'cnn_conv_{layer_idx}'
                )(x)
                channels *= 2
                current_width //= 2
                x = nn.BatchNorm(
                    use_running_average=not training,
                    dtype=COMPUTE_DTYPE,
                    name=f'cnn_bn_{layer_idx}'
                )(x)
                x = nn.relu(x)
                layer_idx += 1

        # Flatten spatial dimensions
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        return x
