"""STORM Transformer - Stochastic Transformer for World Modeling.

This module implements the transformer-based dynamics model for STORM,
replacing the recurrent RSSM with a causal transformer with KV caching.

Based on: "STORM: Efficient Stochastic Transformer based World Models 
for Reinforcement Learning" (Zhang et al.)
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex

from cl.agents.commons.initializer import Initializer
from cl.agents.commons.distributions import OneHotDist
from cl.config.jax_config import COMPUTE_DTYPE


@chex.dataclass
class StormState:
    """State of the STORM transformer.
    
    Unlike RSSM which has deterministic + stochastic components,
    STORM maintains:
    - stoch: Discrete stochastic latents [B, stoch_dim, classes]
    - logits: Logits for the stochastic distribution
    - kv_cache: Cached key-value pairs for efficient autoregressive generation
    """
    stoch: jnp.ndarray  # Stochastic latents [B, stoch, classes]
    logits: jnp.ndarray  # Logits for stochastic state [B, stoch * classes]
    kv_cache: Optional[jnp.ndarray] = None  # KV cache for imagination [B, seq_len, feat_dim]


class PositionalEncoding(nn.Module):
    """Learned positional encoding for transformer.
    
    Args:
        max_length: Maximum sequence length
        embed_dim: Embedding dimension
    """
    max_length: int
    embed_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, position: Optional[int] = None) -> jnp.ndarray:
        """Add positional encoding to input.
        
        Args:
            x: Input features [B, L, D]
            position: Optional specific position (for KV cache forward)
            
        Returns:
            Features with positional encoding added [B, L, D]
        """
        # Learned positional embeddings
        pos_emb = self.param(
            'pos_emb',
            nn.initializers.normal(stddev=0.02),
            (self.max_length, self.embed_dim),
        )
        
        if position is not None:
            # Single position (for KV cache)
            assert x.shape[1] == 1, "position argument requires sequence length 1"
            return x + pos_emb[position:position+1, :]
        else:
            # Full sequence
            seq_len = x.shape[1]
            return x + pos_emb[:seq_len, :]


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention.
    
    Args:
        feat_dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    feat_dim: int
    num_heads: int
    dropout: float = 0.1
    
    @nn.compact
    def __call__(
        self, 
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Multi-head attention.
        
        Args:
            q: Query [B, L_q, D]
            k: Key [B, L_k, D]
            v: Value [B, L_v, D]
            mask: Attention mask [B, L_q, L_k] or [1, L_q, L_k]
            training: Whether in training mode
            
        Returns:
            output: Attention output [B, L_q, D]
            attn: Attention weights [B, num_heads, L_q, L_k]
        """
        batch_size = q.shape[0]
        len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
        head_dim = self.feat_dim // self.num_heads
        
        assert self.feat_dim % self.num_heads == 0, "feat_dim must be divisible by num_heads"
        
        # Linear projections
        init = Initializer(dist='trunc_normal', fan='in', scale=1.0)
        
        q_proj = nn.Dense(
            self.feat_dim, 
            use_bias=False, 
            kernel_init=init,
            dtype=COMPUTE_DTYPE,
            name='q_proj'
        )(q)
        k_proj = nn.Dense(
            self.feat_dim, 
            use_bias=False, 
            kernel_init=init,
            dtype=COMPUTE_DTYPE,
            name='k_proj'
        )(k)
        v_proj = nn.Dense(
            self.feat_dim, 
            use_bias=False, 
            kernel_init=init,
            dtype=COMPUTE_DTYPE,
            name='v_proj'
        )(v)
        
        # Reshape to [B, L, num_heads, head_dim]
        q_proj = q_proj.reshape(batch_size, len_q, self.num_heads, head_dim)
        k_proj = k_proj.reshape(batch_size, len_k, self.num_heads, head_dim)
        v_proj = v_proj.reshape(batch_size, len_v, self.num_heads, head_dim)
        
        # Transpose to [B, num_heads, L, head_dim]
        q_proj = jnp.transpose(q_proj, (0, 2, 1, 3))
        k_proj = jnp.transpose(k_proj, (0, 2, 1, 3))
        v_proj = jnp.transpose(v_proj, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        scale = jnp.sqrt(head_dim).astype(COMPUTE_DTYPE)
        attn_logits = jnp.matmul(q_proj, jnp.transpose(k_proj, (0, 1, 3, 2))) / scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for heads: [B, 1, L_q, L_k]
            if mask.ndim == 3:
                mask = jnp.expand_dims(mask, 1)
            # Use large negative value for masking
            attn_logits = jnp.where(mask, attn_logits, -1e9)
        
        # Softmax
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        
        # Dropout
        if training and self.dropout > 0:
            attn_weights = nn.Dropout(rate=self.dropout)(attn_weights, deterministic=not training)
        
        # Apply attention to values
        output = jnp.matmul(attn_weights, v_proj)  # [B, num_heads, L_q, head_dim]
        
        # Transpose and reshape back to [B, L_q, feat_dim]
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch_size, len_q, self.feat_dim)
        
        # Output projection
        output = nn.Dense(
            self.feat_dim,
            use_bias=False,
            kernel_init=init,
            dtype=COMPUTE_DTYPE,
            name='out_proj'
        )(output)
        
        # Dropout on output
        if training and self.dropout > 0:
            output = nn.Dropout(rate=self.dropout)(output, deterministic=not training)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network.
    
    Args:
        feat_dim: Feature dimension
        hidden_dim: Hidden dimension for FFN (typically 2x feat_dim)
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    feat_dim: int
    hidden_dim: int
    num_heads: int
    dropout: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Transformer block forward pass.
        
        Args:
            x: Input features [B, L, D]
            mask: Attention mask [B, L, L]
            training: Whether in training mode
            
        Returns:
            output: Block output [B, L, D]
            attn: Attention weights [B, num_heads, L, L]
        """
        init = Initializer(dist='trunc_normal', fan='in', scale=1.0)
        
        # Self-attention with residual
        residual = x
        x_norm = nn.LayerNorm(dtype=COMPUTE_DTYPE, name='attn_ln')(x)
        attn_out, attn = MultiHeadAttention(
            feat_dim=self.feat_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            name='attn'
        )(x_norm, x_norm, x_norm, mask=mask, training=training)
        x = residual + attn_out
        
        # Feed-forward network with residual
        residual = x
        x_norm = nn.LayerNorm(dtype=COMPUTE_DTYPE, name='ffn_ln')(x)
        
        # FFN: Linear -> ReLU -> Linear
        ffn = nn.Dense(
            self.hidden_dim,
            kernel_init=init,
            dtype=COMPUTE_DTYPE,
            name='ffn_1'
        )(x_norm)
        ffn = nn.relu(ffn)
        
        if training and self.dropout > 0:
            ffn = nn.Dropout(rate=self.dropout)(ffn, deterministic=not training)
        
        ffn = nn.Dense(
            self.feat_dim,
            kernel_init=init,
            dtype=COMPUTE_DTYPE,
            name='ffn_2'
        )(ffn)
        
        if training and self.dropout > 0:
            ffn = nn.Dropout(rate=self.dropout)(ffn, deterministic=not training)
        
        x = residual + ffn
        
        return x, attn


class StochasticTransformer(nn.Module):
    """Stochastic Transformer for STORM world model.
    
    This transformer processes sequences of (stochastic_latent, action) pairs
    and outputs features that can be used to predict the next stochastic latent.
    
    Args:
        stoch_dim: Dimension of flattened stochastic latents (stoch * classes)
        action_dim: Dimension of actions (discrete action space size)
        feat_dim: Transformer feature dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_length: Maximum sequence length for positional encoding
        dropout: Dropout rate
    """
    stoch_dim: int
    action_dim: int
    feat_dim: int
    num_layers: int
    num_heads: int
    max_length: int
    dropout: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        samples: jnp.ndarray,  # [B, L, stoch_dim]
        actions: jnp.ndarray,  # [B, L] (discrete actions)
        mask: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through transformer.
        
        Args:
            samples: Flattened stochastic samples [B, L, stoch_dim]
            actions: Discrete actions [B, L]
            mask: Causal attention mask [B, L, L] or [1, L, L]
            training: Whether in training mode
            
        Returns:
            features: Transformer output features [B, L, feat_dim]
        """
        init = Initializer(dist='trunc_normal', fan='in', scale=1.0)
        
        # One-hot encode actions
        actions_onehot = jax.nn.one_hot(actions, self.action_dim, dtype=COMPUTE_DTYPE)
        
        # Concatenate samples and actions
        inputs = jnp.concatenate([samples, actions_onehot], axis=-1)  # [B, L, stoch_dim + action_dim]
        
        # Stem: Project to feature dimension
        feats = nn.Dense(
            self.feat_dim,
            use_bias=False,
            kernel_init=init,
            dtype=COMPUTE_DTYPE,
            name='stem_1'
        )(inputs)
        feats = nn.LayerNorm(dtype=COMPUTE_DTYPE, name='stem_ln1')(feats)
        feats = nn.relu(feats)
        
        feats = nn.Dense(
            self.feat_dim,
            use_bias=False,
            kernel_init=init,
            dtype=COMPUTE_DTYPE,
            name='stem_2'
        )(feats)
        feats = nn.LayerNorm(dtype=COMPUTE_DTYPE, name='stem_ln2')(feats)
        
        # Add positional encoding
        feats = PositionalEncoding(
            max_length=self.max_length,
            embed_dim=self.feat_dim,
            name='pos_enc'
        )(feats)
        
        feats = nn.LayerNorm(dtype=COMPUTE_DTYPE, name='post_pos_ln')(feats)
        
        # Transformer layers
        for i in range(self.num_layers):
            feats, _ = TransformerBlock(
                feat_dim=self.feat_dim,
                hidden_dim=self.feat_dim * 2,
                num_heads=self.num_heads,
                dropout=self.dropout,
                name=f'layer_{i}'
            )(feats, mask=mask, training=training)
        
        return feats
    
    def forward_with_kv_cache(
        self,
        sample: jnp.ndarray,  # [B, 1, stoch_dim]
        action: jnp.ndarray,  # [B, 1]
        kv_cache: jnp.ndarray,  # [B, cache_len, feat_dim]
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass with KV cache for efficient autoregressive generation.
        
        This is used during imagination rollouts where we generate one step at a time.
        
        Args:
            sample: Single stochastic sample [B, 1, stoch_dim]
            action: Single action [B, 1]
            kv_cache: Cached transformer features [B, cache_len, feat_dim]
            training: Whether in training mode (typically False during imagination)
            
        Returns:
            feats: Transformer output for current step [B, 1, feat_dim]
            new_kv_cache: Updated KV cache [B, cache_len+1, feat_dim]
        """
        init = Initializer(dist='trunc_normal', fan='in', scale=1.0)
        
        # One-hot encode action
        action_onehot = jax.nn.one_hot(action, self.action_dim, dtype=COMPUTE_DTYPE)
        
        # Concatenate sample and action
        inputs = jnp.concatenate([sample, action_onehot], axis=-1)
        
        # Stem
        feats = nn.Dense(
            self.feat_dim,
            use_bias=False,
            kernel_init=init,
            dtype=COMPUTE_DTYPE,
            name='stem_1'
        )(inputs)
        feats = nn.LayerNorm(dtype=COMPUTE_DTYPE, name='stem_ln1')(feats)
        feats = nn.relu(feats)
        
        feats = nn.Dense(
            self.feat_dim,
            use_bias=False,
            kernel_init=init,
            dtype=COMPUTE_DTYPE,
            name='stem_2'
        )(feats)
        feats = nn.LayerNorm(dtype=COMPUTE_DTYPE, name='stem_ln2')(feats)
        
        # Add positional encoding at current position
        position = kv_cache.shape[1]  # Current cache length
        feats = PositionalEncoding(
            max_length=self.max_length,
            embed_dim=self.feat_dim,
            name='pos_enc'
        )(feats, position=position)
        
        feats = nn.LayerNorm(dtype=COMPUTE_DTYPE, name='post_pos_ln')(feats)
        
        # Update KV cache
        new_kv_cache = jnp.concatenate([kv_cache, feats], axis=1)
        
        # Transformer layers with KV cache
        # Note: In a full implementation, each layer would have its own KV cache
        # For simplicity, we're using a single cache and processing through all layers
        for i in range(self.num_layers):
            # Create mask for attending to all cached positions
            cache_len = new_kv_cache.shape[1]
            mask = jnp.ones((1, 1, cache_len), dtype=bool)
            
            # Self-attention with full cache as K and V
            residual = feats
            feats_norm = nn.LayerNorm(dtype=COMPUTE_DTYPE, name=f'layer_{i}/attn_ln')(feats)
            
            attn_out, _ = MultiHeadAttention(
                feat_dim=self.feat_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                name=f'layer_{i}/attn'
            )(feats_norm, new_kv_cache, new_kv_cache, mask=mask, training=training)
            
            feats = residual + attn_out
            
            # FFN
            residual = feats
            feats_norm = nn.LayerNorm(dtype=COMPUTE_DTYPE, name=f'layer_{i}/ffn_ln')(feats)
            
            ffn = nn.Dense(
                self.feat_dim * 2,
                kernel_init=init,
                dtype=COMPUTE_DTYPE,
                name=f'layer_{i}/ffn_1'
            )(feats_norm)
            ffn = nn.relu(ffn)
            
            ffn = nn.Dense(
                self.feat_dim,
                kernel_init=init,
                dtype=COMPUTE_DTYPE,
                name=f'layer_{i}/ffn_2'
            )(ffn)
            
            feats = residual + ffn
        
        return feats, new_kv_cache


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """Create causal attention mask for transformer.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        mask: Causal mask [1, seq_len, seq_len] with True for valid positions
    """
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    return mask[None, :, :]  # Add batch dimension
