"""Attention blocks with KV-cache support for STORM's TSSM.

This module implements transformer attention blocks that maintain key-value caches
for efficient autoregressive generation during imagination rollouts.
"""

from typing import Tuple, Optional, Dict, Callable
import jax
import jax.numpy as jnp
import flax.linen as nn

from cogniland.config.jax_config import COMPUTE_DTYPE


# Type alias for attention kernels
AttentionKernel = Callable[
    [
        jnp.ndarray,              # query: [B, T_q, H, Dh]
        jnp.ndarray,              # key:   [B, T_kv, H, Dh]
        jnp.ndarray,              # value: [B, T_kv, H, Dh]
        Optional[jnp.ndarray],    # mask:  [B, 1 or H, T_q, T_kv] or None
        bool,                     # training
        Optional[jax.Array],      # dropout_rng
        float,                    # dropout_rate
    ],
    jnp.ndarray,                  # output: [B, T_q, H, Dh]
]


def standard_jax_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: Optional[jnp.ndarray],
    training: bool,
    dropout_rng: Optional[jax.Array],
    dropout_rate: float,
) -> jnp.ndarray:
    """Standard JAX attention using jax.nn.dot_product_attention.

    Args:
        query: Query tensor [B, T_q, H, Dh]
        key: Key tensor [B, T_kv, H, Dh]
        value: Value tensor [B, T_kv, H, Dh]
        mask: Optional attention mask [B, 1 or H, T_q, T_kv]
        training: Whether in training mode
        dropout_rng: Optional RNG key for dropout
        dropout_rate: Dropout rate

    Returns:
        output: Attention output [B, T_q, H, Dh]
    """
    # JAX's dot_product_attention expects [B, T, H, D] format already!
    # No need to transpose - query, key, value are already [B, T, H, D]

    # Apply attention using JAX's optimized implementation
    output = jax.nn.dot_product_attention(
        query,  # [B, T_q, H, Dh]
        key,    # [B, T_kv, H, Dh]
        value,  # [B, T_kv, H, Dh]
        mask=mask,  # [B, 1 or H, T_q, T_kv] or None
    )

    # Apply dropout to attention output if training
    if training and dropout_rate > 0 and dropout_rng is not None:
        keep_prob = 1.0 - dropout_rate
        keep = jax.random.bernoulli(dropout_rng, keep_prob, output.shape)
        output = jnp.where(keep, output / keep_prob, 0)

    # Output is already in [B, T_q, H, Dh] format
    return output.astype(query.dtype)  # Cast back to input dtype


class AttentionBlockKVCache(nn.Module):
    """Multi-head self-attention with KV-cache support.

    - x: [B, T_q, hidden_dim]
    - kv_cache (optional): {"keys": [B, T_prev, hidden_dim], "values": same}
    - returns (output, new_cache)
    """
    hidden_dim: int = 512
    num_heads: int = 8
    dropout_rate: float = 0.0
    use_bias: bool = True
    attention_fn: AttentionKernel = standard_jax_attention
    max_cache_length: int = 64  # Ring buffer size for KV-cache

    def setup(self):
        """Initialize attention components."""
        # Validate dimensions
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"

        self.head_dim = self.hidden_dim // self.num_heads

        # Fused QKV projection
        self.qkv = nn.Dense(
            3 * self.hidden_dim,
            use_bias=self.use_bias,
            dtype=COMPUTE_DTYPE,
            name='qkv'
        )

        # Output projection
        self.out = nn.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            dtype=COMPUTE_DTYPE,
            name='out'
        )

        # Layer norm (pre-attention)
        self.norm = nn.LayerNorm(dtype=COMPUTE_DTYPE, name='norm')

    def _causal_mask(self, q_len: int, kv_len: int) -> jnp.ndarray:
        """Return standard causal mask of shape [1, 1, q_len, kv_len].

        Suitable for jax.nn.dot_product_attention.
        """
        # Create causal mask where positions can only attend to earlier positions
        # mask[i, j] = True if query position i can attend to key position j
        # For causal: i >= j - (kv_len - q_len)
        row_indices = jnp.arange(q_len)[:, None]
        col_indices = jnp.arange(kv_len)[None, :]

        # Offset for when kv_len > q_len (due to cache)
        offset = kv_len - q_len
        mask = row_indices >= (col_indices - offset)

        # Reshape to [1, 1, q_len, kv_len] for broadcasting
        mask = mask[None, None, :, :]

        return mask

    def __call__(
        self,
        x: jnp.ndarray,                    # [B, T_q, D]
        kv_cache: Optional[Dict[str, jnp.ndarray]] = None,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Forward pass with optional KV-cache.

        Args:
            x: Input tensor [B, T_q, hidden_dim]
            kv_cache: Optional dict with:
                     - 'keys': Fixed-size cache [B, max_cache_length, hidden_dim]
                     - 'values': Fixed-size cache [B, max_cache_length, hidden_dim]
                     - 'step': Number of valid tokens [B] (for rolling buffer)
            training: Whether in training mode

        Returns:
            output: Attention output [B, T_q, hidden_dim]
            new_cache: Updated KV-cache with fixed size
        """
        batch_size, q_len, _ = x.shape

        # 1. Fused QKV projection (Post-Norm: projection directly on x)
        qkv = self.qkv(x)  # [B, T_q, 3 * hidden_dim]
        qkv = qkv.reshape(batch_size, q_len, 3, self.hidden_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Each: [B, T_q, hidden_dim]

        # 3. Handle KV-cache with rolling buffer
        if kv_cache is not None and 'keys' in kv_cache and 'step' in kv_cache:
            # Fixed-size rolling buffer mode (autoregressive inference)
            prev_keys = kv_cache['keys']  # [B, max_cache_length, hidden_dim]
            prev_values = kv_cache['values']  # [B, max_cache_length, hidden_dim]
            cache_step = kv_cache['step']  # [B]

            # Update cache at rolling position
            # For each batch element, update at position cache_step[b] % max_cache_length
            def update_cache_single(prev_cache, new_token, step):
                """Update single batch element's cache."""
                pos = step % self.max_cache_length
                # Use dynamic_update_slice to update at position pos
                updated = jax.lax.dynamic_update_slice(
                    prev_cache,
                    new_token[None, :],  # [1, hidden_dim]
                    (pos, 0)
                )
                return updated

            # Vectorize over batch dimension
            k_cache = jax.vmap(update_cache_single)(prev_keys, k[:, 0, :], cache_step)
            v_cache = jax.vmap(update_cache_single)(prev_values, v[:, 0, :], cache_step)

            # Use full cache for attention
            kv_len = self.max_cache_length

            # Build mask that only attends to valid tokens
            # Each batch element has (cache_step[b] + 1) valid tokens
            # Mask shape: [B, 1, 1, max_cache_length]
            valid_len = jnp.minimum(cache_step + 1, self.max_cache_length)  # [B]
            position_indices = jnp.arange(self.max_cache_length)[None, :]  # [1, max_len]
            valid_mask = position_indices < valid_len[:, None]  # [B, max_len]
            # Reshape for broadcasting: [B, 1, 1, max_len]
            valid_mask = valid_mask[:, None, None, :]

            # Combine with causal mask
            causal_mask = self._causal_mask(q_len, kv_len)  # [1, 1, q_len, kv_len]
            mask = causal_mask & valid_mask

            # Update step counter
            new_step = cache_step + 1

        elif kv_cache is not None and 'keys' in kv_cache:
            # Legacy mode: cache without step counter (training with observe_sequence)
            # Just use the cache as-is
            k_cache = kv_cache['keys']
            v_cache = kv_cache['values']
            kv_len = k_cache.shape[1]
            mask = self._causal_mask(q_len, kv_len)
            # Use zeros as placeholder step counter for consistent PyTree structure
            new_step = jnp.zeros((batch_size,), dtype=jnp.int32)

        else:
            # No cache - first step or training mode
            k_cache = k  # [B, T_q, hidden_dim]
            v_cache = v  # [B, T_q, hidden_dim]
            kv_len = q_len
            mask = self._causal_mask(q_len, kv_len)
            # Use appropriate step counter: 1 for single-token, 0 for multi-token
            new_step = jnp.ones((batch_size,), dtype=jnp.int32) if q_len == 1 else jnp.zeros((batch_size,), dtype=jnp.int32)

        # Reshape Q, K, V from hidden space to head space for attention
        q = q.reshape(batch_size, q_len, self.num_heads, self.head_dim)  # [B, T_q, H, Dh]
        k_attn = k_cache.reshape(batch_size, kv_len, self.num_heads, self.head_dim)  # [B, T_kv, H, Dh]
        v_attn = v_cache.reshape(batch_size, kv_len, self.num_heads, self.head_dim)  # [B, T_kv, H, Dh]

        # 5. Build dropout_rng if needed
        dropout_rng = None
        if training and self.dropout_rate > 0:
            dropout_rng = self.make_rng('dropout')

        # 6. Call attention kernel
        # pylint: disable=redundant-keyword-arg
        attn_output = self.attention_fn(
            q, k_attn, v_attn,
            mask=mask,
            training=training,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate
        )  # [B, T_q, H, Dh]

        # 7. Merge heads back to [B, T_q, hidden_dim]
        attn_output = attn_output.reshape(batch_size, q_len, self.hidden_dim)

        # Apply output projection
        output = self.out(attn_output)

        # Add residual connection
        output = output + x

        # Apply LayerNorm (Post-Norm: norm after residual)
        output = self.norm(output)

        # 8. Return updated cache with fixed size
        # Always include 'step' for consistent PyTree structure
        new_cache = {
            'keys': k_cache,  # [B, max_cache_length, hidden_dim]
            'values': v_cache,  # [B, max_cache_length, hidden_dim]
            'step': new_step  # [B] - always present for PyTree consistency
        }

        return output, new_cache


class TransformerBlock(nn.Module):
    """Complete transformer block with attention and MLP."""

    hidden_dim: int = 512
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    use_bias: bool = True
    max_cache_length: int = 64
    attention_fn: AttentionKernel = standard_jax_attention

    def setup(self):
        """Initialize attention and MLP layers."""
        self.attention = AttentionBlockKVCache(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            max_cache_length=self.max_cache_length,
            attention_fn=self.attention_fn
        )

        mlp_dim = int(self.hidden_dim * self.mlp_ratio)

        # MLP layers
        self.mlp_norm = nn.LayerNorm(dtype=COMPUTE_DTYPE)
        self.mlp1 = nn.Dense(mlp_dim, use_bias=self.use_bias, dtype=COMPUTE_DTYPE)
        self.mlp2 = nn.Dense(self.hidden_dim, use_bias=self.use_bias, dtype=COMPUTE_DTYPE)

        # Define dropout modules in setup
        if self.dropout_rate > 0:
            self.dropout1 = nn.Dropout(rate=self.dropout_rate)
            self.dropout2 = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x: jnp.ndarray,
        kv_cache: Optional[Dict[str, jnp.ndarray]] = None,
        training: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Forward pass through transformer block.

        Args:
            x: Input tensor [B, seq_len, hidden_dim]
            kv_cache: Optional KV-cache from previous steps
            training: Whether in training mode

        Returns:
            output: Block output [B, seq_len, hidden_dim]
            new_cache: Updated KV-cache
        """
        # Self-attention with residual (Post-Norm handled inside)
        attn_out, new_cache = self.attention(x, kv_cache, training)

        # MLP with residual (Post-Norm)
        # 1. MLP layers directly on attention output
        mlp_hidden = self.mlp1(attn_out)
        mlp_hidden = nn.gelu(mlp_hidden)

        if self.dropout_rate > 0:
            mlp_hidden = self.dropout1(mlp_hidden, deterministic=not training)

        mlp_out = self.mlp2(mlp_hidden)

        if self.dropout_rate > 0:
            mlp_out = self.dropout2(mlp_out, deterministic=not training)

        # 2. Residual connection
        output = attn_out + mlp_out

        # 3. Norm after residual
        output = self.mlp_norm(output)

        return output, new_cache
