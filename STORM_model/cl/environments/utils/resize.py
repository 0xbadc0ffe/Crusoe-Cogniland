"""Lightweight image resizing utilities for RGB observations."""

import jax
import jax.numpy as jnp
from typing import Tuple, Literal


def resize_observation(
    obs: jnp.ndarray,
    target_size: Tuple[int, int],
    method: Literal['bilinear', 'bicubic', 'nearest', 'lanczos3', 'lanczos5'] = 'bilinear',
) -> jnp.ndarray:
    """
    Resize RGB observation to target resolution.

    This is a lightweight, stateless function suitable for preprocessing
    observations before feeding to neural networks. It handles both single
    images and batched images automatically.

    Args:
        obs: Observation array of shape:
            - (H, W, C) for single image
            - (B, H, W, C) for batched images
            - (B, T, H, W, C) for batched sequences
        target_size: Target (height, width) for resizing
        method: Interpolation method:
            - 'bilinear': Fast, smooth (recommended for upsampling)
            - 'bicubic': Smoother but slower
            - 'nearest': Fastest but blocky (not recommended for upsampling)
            - 'lanczos3': High quality but slower
            - 'lanczos5': Highest quality but slowest

    Returns:
        Resized observation with same dtype and same number of dimensions.
        Shape will be:
            - (target_H, target_W, C) for single image
            - (B, target_H, target_W, C) for batched images
            - (B, T, target_H, target_W, C) for batched sequences

    Examples:
        >>> # Single image
        >>> obs = jnp.zeros((56, 56, 3))
        >>> resized = resize_observation(obs, (64, 64))
        >>> resized.shape
        (64, 64, 3)

        >>> # Batched images
        >>> obs = jnp.zeros((16, 56, 56, 3))
        >>> resized = resize_observation(obs, (64, 64))
        >>> resized.shape
        (16, 64, 64, 3)

        >>> # Batched sequences
        >>> obs = jnp.zeros((16, 10, 56, 56, 3))
        >>> resized = resize_observation(obs, (64, 64))
        >>> resized.shape
        (16, 10, 64, 64, 3)
    """
    original_shape = obs.shape
    original_dtype = obs.dtype

    # Handle different input shapes
    if len(original_shape) == 3:
        # Single image (H, W, C)
        h, w, c = original_shape
        resized = jax.image.resize(
            obs,
            shape=(target_size[0], target_size[1], c),
            method=method,
        )
    elif len(original_shape) == 4:
        # Batched images (B, H, W, C)
        b, h, w, c = original_shape
        # Vectorize over batch dimension
        resized = jax.vmap(
            lambda img: jax.image.resize(
                img,
                shape=(target_size[0], target_size[1], c),
                method=method,
            )
        )(obs)
    elif len(original_shape) == 5:
        # Batched sequences (B, T, H, W, C)
        b, t, h, w, c = original_shape
        # Vectorize over batch and time dimensions
        resized = jax.vmap(
            jax.vmap(
                lambda img: jax.image.resize(
                    img,
                    shape=(target_size[0], target_size[1], c),
                    method=method,
                )
            )
        )(obs)
    else:
        raise ValueError(
            f"Unsupported observation shape: {original_shape}. "
            f"Expected (H, W, C), (B, H, W, C), or (B, T, H, W, C)."
        )

    # Preserve original dtype (JAX resize returns float32 by default)
    if original_dtype != resized.dtype:
        resized = resized.astype(original_dtype)

    return resized


def resize_observation_dict(
    obs_dict: dict,
    target_size: Tuple[int, int],
    image_keys: Tuple[str, ...] = ('image', 'rgb', 'pixels'),
    method: str = 'bilinear',
) -> dict:
    """
    Resize RGB observations in a dictionary (for dict observation spaces).

    This is useful when your environment returns observations as a dictionary
    with multiple keys, and you only want to resize specific image keys.

    Args:
        obs_dict: Dictionary of observations
        target_size: Target (height, width) for resizing
        image_keys: Keys that contain RGB images to resize
        method: Interpolation method (see resize_observation)

    Returns:
        New dictionary with resized images for specified keys, other keys unchanged.

    Example:
        >>> obs = {
        ...     'image': jnp.zeros((56, 56, 3)),
        ...     'state': jnp.zeros((10,)),
        ... }
        >>> resized = resize_observation_dict(obs, (64, 64))
        >>> resized['image'].shape
        (64, 64, 3)
        >>> resized['state'].shape
        (10,)
    """
    result = {}
    for key, value in obs_dict.items():
        if key in image_keys and len(value.shape) >= 3:
            # Resize this image observation
            result[key] = resize_observation(value, target_size, method)
        else:
            # Keep other observations unchanged
            result[key] = value
    return result


# JIT-compiled versions for better performance
resize_observation_jit = jax.jit(
    resize_observation,
    static_argnames=('target_size', 'method')
)

resize_observation_dict_jit = jax.jit(
    resize_observation_dict,
    static_argnames=('target_size', 'image_keys', 'method')
)


def get_resized_obs_shape(
    original_shape: Tuple[int, ...],
    target_size: Tuple[int, int],
) -> Tuple[int, ...]:
    """
    Get the output shape after resizing without actually resizing.

    Useful for determining observation space shapes.

    Args:
        original_shape: Original observation shape
        target_size: Target (height, width)

    Returns:
        Output shape after resizing

    Example:
        >>> get_resized_obs_shape((56, 56, 3), (64, 64))
        (64, 64, 3)
        >>> get_resized_obs_shape((16, 56, 56, 3), (64, 64))
        (16, 64, 64, 3)
    """
    if len(original_shape) == 3:
        h, w, c = original_shape
        return (target_size[0], target_size[1], c)
    elif len(original_shape) == 4:
        b, h, w, c = original_shape
        return (b, target_size[0], target_size[1], c)
    elif len(original_shape) == 5:
        b, t, h, w, c = original_shape
        return (b, t, target_size[0], target_size[1], c)
    else:
        raise ValueError(f"Unsupported shape: {original_shape}")
