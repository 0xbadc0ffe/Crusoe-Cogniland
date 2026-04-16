"""Utility functions for symbolic observation visualization.

This module provides utilities for converting symbolic Navix observations
to RGB images for visualization in WandB videos.
"""

import numpy as np


# Entity IDs after remapping (see cl/environments/navix.py ENTITY_REMAP)
# Original Navix IDs had a gap at class 3, which has been closed.
ENTITY_IDS = {
    'UNKNOWN': 0,
    'FLOOR': 1,
    'WALL': 2,
    'DOOR': 3,   # Remapped from 4
    'KEY': 4,    # Remapped from 5
    'BALL': 5,   # Remapped from 6
    'BOX': 6,    # Remapped from 7
    'GOAL': 7,   # Remapped from 8
    'LAVA': 8,   # Remapped from 9
    'PLAYER': 9, # Remapped from 10
}

# RGB colors for visualization (0-255 range for uint8 compatibility)
# Colors chosen to be visually distinct and match typical MiniGrid conventions
# NOTE: These use the REMAPPED entity IDs (0-9), not the original Navix IDs.
ENTITY_COLORS = {
    0: (51, 51, 51),        # UNKNOWN - dark gray
    1: (230, 230, 230),     # FLOOR - light gray
    2: (102, 102, 102),     # WALL - gray
    3: (153, 76, 25),       # DOOR - brown (was ID 4)
    4: (255, 214, 0),       # KEY - gold (was ID 5)
    5: (0, 0, 255),         # BALL - blue (was ID 6)
    6: (128, 0, 128),       # BOX - purple (was ID 7)
    7: (0, 204, 0),         # GOAL - green (was ID 8)
    8: (255, 102, 0),       # LAVA - orange (was ID 9)
    9: (255, 0, 0),         # PLAYER - red (was ID 10)
}


def symbolic_to_rgb(symbolic_obs: np.ndarray) -> np.ndarray:
    """Convert symbolic observation to RGB image for visualization.

    Args:
        symbolic_obs: Array of shape [..., H, W, 3] with entity tags in channel 0.
                     Can be float32 (from model predictions) or integer types.
                     The 3 channels are: [entity_id, color_idx, state]
                     Only entity_id (channel 0) is used for visualization.

    Returns:
        RGB array of shape [..., H, W, 3] with uint8 values in [0, 255]
    """
    # Get entity tags from channel 0
    if symbolic_obs.shape[-1] == 3:
        entity_tags = symbolic_obs[..., 0]
    else:
        # Assume the entire array is entity tags
        entity_tags = symbolic_obs

    # Convert to integer for indexing (round if float)
    if not np.issubdtype(entity_tags.dtype, np.integer):
        entity_tags = np.round(entity_tags).astype(np.int32)

    # Clamp to valid range [0, 9] (after entity ID remapping)
    entity_tags = np.clip(entity_tags, 0, 9)

    # Create output RGB array
    output_shape = entity_tags.shape + (3,)
    rgb = np.zeros(output_shape, dtype=np.uint8)

    # Map entity IDs to colors
    for entity_id, color in ENTITY_COLORS.items():
        mask = entity_tags == entity_id
        rgb[mask] = color

    return rgb


def categorical_to_rgb(categorical_obs: np.ndarray) -> np.ndarray:
    """Convert categorical observation to RGB image for visualization.

    This is an alias for symbolic_to_rgb that handles categorical observations
    which have shape [..., H, W] with just entity IDs (no color/state channels).

    Args:
        categorical_obs: Array of shape [..., H, W] with entity IDs.
                        Can be float32 (from model predictions) or integer types.

    Returns:
        RGB array of shape [..., H, W, 3] with uint8 values in [0, 255]
    """
    return symbolic_to_rgb(categorical_obs)


def is_symbolic_observation(navix_obs_type: str) -> bool:
    """Check if the observation type is symbolic or categorical (not RGB).

    Args:
        navix_obs_type: Observation type string from config
                       (e.g., 'rgb_first_person', 'symbolic_first_person', 'categorical_first_person')

    Returns:
        True if the observation type is symbolic or categorical, False otherwise
    """
    if navix_obs_type is None:
        return False
    return navix_obs_type in ['symbolic', 'symbolic_first_person', 'categorical', 'categorical_first_person']


def is_categorical_observation(navix_obs_type: str) -> bool:
    """Check if the observation type is categorical (entity IDs only).

    Args:
        navix_obs_type: Observation type string from config

    Returns:
        True if the observation type is categorical, False otherwise
    """
    if navix_obs_type is None:
        return False
    return navix_obs_type in ['categorical', 'categorical_first_person']
