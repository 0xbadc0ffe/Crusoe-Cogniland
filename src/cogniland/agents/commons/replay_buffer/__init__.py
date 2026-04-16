"""
Replay buffer implementations for RL agents.

This package contains:
- Flat buffers: Simple transition-level storage (FlatBuffer, ReservoirFlatBuffer, PERFlatBuffer)
- Sequence buffers: DreamerV3-style sequence storage (ReplayBuffer, ReservoirReplayBuffer)
- R2D2 buffers: Sequence buffers with LSTM hidden state storage (R2D2ReplayBuffer, R2D2ReservoirReplayBuffer)

JAX-native (Flashbax-based) buffers are experimental and live under:
    cl.agents.experimental.replay_buffer.jax_native
"""

# Flat buffers (transition-level)
from .flat import (
    FlatBuffer,
    FlatBufferState,
    ReservoirFlatBuffer,
    SumTree,
    PERBufferState,
    PERFlatBuffer,
)

# Sequence buffers (DreamerV3-style)
from .sequence import (
    ReplayBuffer,
    ReplayBufferState,
    ReservoirReplayBuffer,
)

# R2D2 buffers (with LSTM hidden state)
from .r2d2 import (
    R2D2ReplayBuffer,
    R2D2ReservoirReplayBuffer,
)

__all__ = [
    # Flat buffers (NumPy-based)
    "FlatBuffer",
    "FlatBufferState",
    "ReservoirFlatBuffer",
    "SumTree",
    "PERBufferState",
    "PERFlatBuffer",
    # Sequence buffers (NumPy-based)
    "ReplayBuffer",
    "ReplayBufferState",
    "ReservoirReplayBuffer",
    # R2D2 buffers (NumPy-based)
    "R2D2ReplayBuffer",
    "R2D2ReservoirReplayBuffer",
]
