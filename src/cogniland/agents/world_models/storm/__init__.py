"""STORM world model implementation.

STORM (Stochastic Transformer-based wORld Model) replaces DreamerV3's RSSM with a
Transformer State-Space Model (TSSM) for improved sequence modeling.

Key differences from DreamerV3:
- Uses Transformer instead of GRU for dynamics
- Maintains KV-cache for efficient autoregressive generation
- Uses BatchNorm/LayerNorm instead of RMSNorm for normalization
- Two-phase imagination: context phase (with observations) + rollout phase (pure imagination)
"""

from cogniland.agents.world_models.storm.state import TSSMState
from cogniland.agents.world_models.storm.tssm import TSSM
from cogniland.agents.world_models.storm.world_model import STORMWorldModel
from cogniland.agents.world_models.storm.attention import TransformerBlock, AttentionBlockKVCache

__all__ = [
    'TSSMState',
    'TSSM',
    'STORMWorldModel',
    'TransformerBlock',
    'AttentionBlockKVCache',
]
