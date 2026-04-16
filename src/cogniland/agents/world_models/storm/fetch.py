"""Fetch module for JTP multi-step prediction from a single bottleneck."""

import jax.numpy as jnp
import flax.linen as nn

from cogniland.config.jax_config import COMPUTE_DTYPE


class Fetch(nn.Module):
    bottleneck_dim: int          # Dimension of bottleneck feature (hidden + stoch_flat)
    embed_dim: int               # Dimension of encoded observations
    action_dim: int              # Action dimension (one-hot)
    stoch: int                   # Number of categorical variables
    classes: int                 # Classes per categorical
    hidden_dim: int = 512        # Internal hidden size
    num_heads: int = 4           # Attention heads
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        bottleneck: jnp.ndarray,          # [B, F]
        future_embeds: jnp.ndarray,       # [B, H, E]
        future_actions: jnp.ndarray,      # [B, H, A]
        *,
        training: bool = False,
    ):
        B, H, _ = future_embeds.shape

        bottleneck_tiled = jnp.tile(bottleneck[:, None, :], (1, H, 1))
        token_inputs = jnp.concatenate([bottleneck_tiled, future_embeds, future_actions], axis=-1)
        token_inputs = token_inputs.astype(COMPUTE_DTYPE)

        x = nn.Dense(self.hidden_dim, dtype=COMPUTE_DTYPE)(token_inputs)
        x = nn.LayerNorm(dtype=COMPUTE_DTYPE)(x)
        x = nn.relu(x)

        # Single-layer causal attention
        attn_mask = jnp.tril(jnp.ones((H, H), dtype=bool))
        attn_mask = attn_mask[None, None, ...]  # [1,1,H,H]
        attn_out = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            dtype=COMPUTE_DTYPE,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
            use_bias=True,
        )(x, x, x, mask=attn_mask)
        x = x + attn_out
        x = nn.LayerNorm(dtype=COMPUTE_DTYPE)(x)

        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        prior_hidden = x  # [B, H, hidden_dim]

        logits = nn.Dense(self.stoch * self.classes, dtype=COMPUTE_DTYPE)(x)
        logits = logits.reshape(B, H, self.stoch, self.classes)

        return logits, prior_hidden
