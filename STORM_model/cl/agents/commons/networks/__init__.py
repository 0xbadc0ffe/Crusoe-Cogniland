"""Shared neural network building blocks for agents.

This package contains reusable network components that can be used across
different agent implementations (DreamerV3, STORM, IRIS, PPO, etc.).

Key components:
    - blocks.py: Building blocks (BlockLinear, LayerNorm, etc.)
    - mlp.py: MLP heads with distribution outputs
"""

from .blocks import BlockLinear
from .mlp import MLPHead
from .cnn import (
    ResidualBlock,
    ConvSequence,
    ImpalaCNN,
)

__all__ = [
    'BlockLinear',
    'MLPHead',
    'ResidualBlock',
    'ConvSequence',
    'ImpalaCNN',
]
