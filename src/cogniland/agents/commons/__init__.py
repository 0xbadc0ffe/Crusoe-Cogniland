"""Common utilities for RL agents."""

# Import all replay buffer classes from the replay_buffer package
from .replay_buffer import (
    # Flat buffers
    FlatBuffer,
    FlatBufferState,
    ReservoirFlatBuffer,
    SumTree,
    PERBufferState,
    PERFlatBuffer,
    # Sequence buffers
    ReplayBuffer,
    ReplayBufferState,
    ReservoirReplayBuffer,
    # R2D2 buffers
    R2D2ReplayBuffer,
    R2D2ReservoirReplayBuffer,
)

from .utils import (
    orthogonal_init,
    ortho_init,
    ortho_init_small,
    ortho_init_one,
)

from .preprocessing import (
    resize_flat,
    normalize_image,
    preprocess_image,
    should_use_cnn,
    extract_image_obs,
)

__all__ = [
    # Flat replay buffers (for DQN-family agents)
    'FlatBuffer',
    'FlatBufferState',
    'ReservoirFlatBuffer',
    'SumTree',
    'PERBufferState',
    'PERFlatBuffer',
    # Sequence replay buffers (for world model agents)
    'ReplayBuffer',
    'ReplayBufferState',
    'ReservoirReplayBuffer',
    # R2D2 replay buffers (sequence + hidden state)
    'R2D2ReplayBuffer',
    'R2D2ReservoirReplayBuffer',
    # Utilities
    'orthogonal_init',
    'ortho_init',
    'ortho_init_small',
    'ortho_init_one',
    # Preprocessing utilities
    'resize_flat',
    'normalize_image',
    'preprocess_image',
    'should_use_cnn',
    'extract_image_obs',
]
