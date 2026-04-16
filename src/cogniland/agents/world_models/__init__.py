"""World model zoo for JAX-based continual reinforcement learning.

This package provides a modular framework for implementing different world model
architectures (DreamerV3-RSSM, STORM, IRIS, etc.) with shared infrastructure.

Key components:
    - base.py: WorldModel protocol and base dataclasses
    - dreamerv3/: DreamerV3 RSSM-based world model implementation
    - encoders/: Reusable observation encoders
    - decoders/: Reusable observation decoders

Future extensions:
    - storm/: STORM, a transformer-based world model 
    - iris/: VQ-tokenized world models (IRIS, delta-IRIS)

Design philosophy:
    1. Parameterless protocols: Methods take params explicitly (Flax style)
    2. Composability: Mix encoders, dynamics, decoders from different sources
    3. Reusability: Share infrastructure (replay, optimizer, training loops)
    4. Extensibility: Add new world models by implementing WorldModel protocol
"""

from .base import WorldModel, WorldModelParams, WorldModelState

__all__ = [
    # Base protocol
    'WorldModel',
    'WorldModelParams',
    'WorldModelState',
]
