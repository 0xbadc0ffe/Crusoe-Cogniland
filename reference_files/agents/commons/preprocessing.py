"""Common observation preprocessing utilities.

This module standardizes image preprocessing across ALL agents:
- Resize: Efficient batched resize via flatten/unflatten
- Normalize: uint8 -> float32/255 conversion
- CNN detection: Automatic based on observation shape

*Note*: Replay buffers store RAW observations (uint8).
        Preprocessing is applied at training/inference time.
"""

from typing import Dict, Tuple, Optional, Union
import jax
import jax.numpy as jnp
import numpy as np


def resize_flat(
    obs: jnp.ndarray,
    target_size: Tuple[int, int],
    method: str = 'nearest',
) -> jnp.ndarray:
    """Resize by flattening batch dims, resizing, then unflattening.

    ~30% faster than resize inside nested vmap due to better XLA fusion.

    Args:
        obs: Image tensor with shape (..., H, W, C)
        target_size: Target (H, W) dimensions
        method: Interpolation method ('nearest', 'bilinear', etc.)

    Returns:
        Resized tensor with shape (..., target_H, target_W, C)
    """
    original_shape = obs.shape
    h, w, c = original_shape[-3:]
    batch_shape = original_shape[:-3]

    total_batch = int(np.prod(batch_shape)) if batch_shape else 1
    flat = obs.reshape((total_batch, h, w, c))

    resized = jax.vmap(
        lambda img: jax.image.resize(
            img, shape=(target_size[0], target_size[1], c), method=method
        )
    )(flat)

    output_shape = batch_shape + (target_size[0], target_size[1], c)
    return resized.reshape(output_shape).astype(obs.dtype)


def normalize_image(obs: jnp.ndarray) -> jnp.ndarray:
    """Normalize uint8 image to float32 in [0, 1].

    If already float, returns unchanged. This allows idempotent calls.

    Args:
        obs: Image tensor, either uint8 [0, 255] or float32 [0, 1]

    Returns:
        Float32 tensor in [0, 1]
    """
    if obs.dtype == jnp.uint8:
        return obs.astype(jnp.float32) / 255.0
    return obs


def preprocess_image(
    obs: jnp.ndarray,
    resize_target: Optional[Tuple[int, int]] = None,
    resize_method: str = 'nearest',
    normalize: bool = True,
    flatten_for_mlp: bool = False,
) -> jnp.ndarray:
    """Full preprocessing pipeline: resize, normalize, and optionally flatten.

    Args:
        obs: Image tensor with shape (..., H, W, C)
        resize_target: Optional (H, W) to resize to
        resize_method: Interpolation method for resize
        normalize: Whether to normalize uint8 to float32
        flatten_for_mlp: Whether to flatten spatial dims for MLP input.
            If True, output shape is (..., H*W*C)

    Returns:
        Preprocessed image tensor
    """
    if resize_target is not None:
        obs = resize_flat(obs, resize_target, resize_method)
    if normalize:
        obs = normalize_image(obs)
    if flatten_for_mlp:
        # Flatten spatial dimensions, keeping batch dims
        batch_dims = obs.shape[:-3]
        flat_dim = obs.shape[-3] * obs.shape[-2] * obs.shape[-1]
        obs = obs.reshape(batch_dims + (flat_dim,))
    return obs


def should_use_cnn(obs_shape: Tuple[int, ...]) -> bool:
    """Detect if CNN should be used based on observation shape.

    Returns True for HWC format images with 1, 3, or 4 channels.
    """
    return len(obs_shape) == 3 and obs_shape[-1] in (1, 3, 4)


def extract_image_obs(obs: Union[Dict, jnp.ndarray]) -> jnp.ndarray:
    """Extract image observation from dict or pass through array."""
    if isinstance(obs, dict):
        return obs['image']
    return obs
