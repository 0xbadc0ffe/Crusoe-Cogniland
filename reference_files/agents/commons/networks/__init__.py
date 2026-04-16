"""Shared neural network building blocks for agents.

This package contains reusable network components that can be used across
different agent implementations (DreamerV3, STORM, IRIS, PPO, etc.).

Key components:
    - blocks.py: Building blocks (BlockLinear, LayerNorm, etc.)
    - mlp.py: MLP heads with distribution outputs
    - cnn.py: CNN components (ResidualBlock, ImpalaCNN, etc.)
    - rnn.py: Recurrent components (ScannedRNN, LSTMState)
    - noisy.py: Noisy layers for exploration (NoisyNet)
"""

from .blocks import BlockLinear
from .mlp import MLPHead
from .cnn import (
    ResidualBlock,
    ResidualBlockSN,
    ConvSequence,
    ConvSequenceSN,
    ImpalaCNN,
    ImpalaCNNSN,
    adaptive_pool2d,
)
from .rnn import LSTMState, ScannedRNN
from .noisy import NoisyLinear, NoisyDense

__all__ = [
    'BlockLinear',
    'MLPHead',
    'ResidualBlock',
    'ResidualBlockSN',
    'ConvSequence',
    'ConvSequenceSN',
    'ImpalaCNN',
    'ImpalaCNNSN',
    'adaptive_pool2d',
    'LSTMState',
    'ScannedRNN',
    'NoisyLinear',
    'NoisyDense',
]
