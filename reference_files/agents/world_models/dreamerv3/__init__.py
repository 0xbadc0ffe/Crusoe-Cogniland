"""DreamerV3 world model implementation.

This package contains the DreamerV3 world model components:
    - encoder.py: Multi-modal encoder (handles both vector and image observations)
    - decoder.py: Multi-modal decoder (reconstructs both vector and image observations)
    - rssm.py: RSSM dynamics model (includes RSSMState)
    - world_model.py: DreamerV3WorldModel wrapper (implements WorldModel protocol)
    - state.py: DreamerV3-specific state definitions (DreamerV3Params, DreamerV3TrainState)

The DreamerV3 world model combines:
    - Encoder: Multi-modal encoder (MLP for vectors, CNN for images)
    - Decoder: Multi-modal decoder (MLP for vectors, CNN for images)
    - RSSM: Recurrent State-Space Model (GRU + categorical latents)
    - MLP heads: Reward and continuation predictors
"""

from .rssm import RSSM, RSSMState
from .encoder import Encoder
from .decoder import Decoder
from .world_model import DreamerV3WorldModel
from .state import DreamerV3Params, DreamerV3TrainState

__all__ = [
    'RSSM',
    'RSSMState',
    'Encoder',
    'Decoder',
    'DreamerV3WorldModel',
    'DreamerV3Params',
    'DreamerV3TrainState',
]
