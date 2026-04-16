"""TSSM (Transformer State-Space Model) for STORM.

This module implements the Transformer State-Space Model which replaces
DreamerV3's RSSM (GRU-based) with transformer-based dynamics.
"""

from typing import Tuple, Dict
import jax
import jax.numpy as jnp
import flax.linen as nn

from cl.agents.commons.initializer import Initializer
from cl.agents.commons.distributions import OneHotDist
from cl.agents.world_models.storm.state import TSSMState
from cl.agents.world_models.storm.attention import TransformerBlock
from cl.config.jax_config import COMPUTE_DTYPE


class TSSM(nn.Module):
    """Transformer State-Space Model for STORM.

    Replaces RSSM's GRU with transformer layers for sequence modeling.
    Maintains compatibility with the same observe/imagine interface.
    """

    # Model dimensions
    stoch: int = 32             # Number of categorical distributions
    classes: int = 32           # Classes per categorical
    hidden_dim: int = 512       # Transformer hidden dimension

    # Transformer architecture
    num_layers: int = 2         # Number of transformer layers
    num_heads: int = 8          # Number of attention heads
    mlp_ratio: float = 4.0      # MLP dimension ratio
    dropout_rate: float = 0.0   # Dropout rate
    max_length: int = 64        # Maximum sequence length for positional encoding
    max_cache_length: int = 64  # Ring buffer size for KV-cache (prevents unbounded growth)

    # Other hyperparameters
    unimix: float = 0.01        # Uniform mixture for regularization
    act: str = 'gelu'           # Activation function
    norm: str = 'layer'         # Normalization type (layer for transformers)
    free_nats: float = 1.0      # Free nats for KL divergence
    winit: str = 'trunc_normal_in'  # Weight initialization
    binit: str = 'zeros'        # Bias initialization
    action_dim: int = 0         # Action dimension (set from outside)

    def setup(self):
        """Initialize TSSM layers."""
        # Parse initializers
        if isinstance(self.winit, str):
            if self.winit == 'trunc_normal_in':
                weight_init = Initializer(dist='trunc_normal', fan='in', scale=1.0)
            elif self.winit == 'trunc_normal_out':
                weight_init = Initializer(dist='trunc_normal', fan='out', scale=1.0)
            else:
                raise ValueError(f"Unknown winit: {self.winit}")
        else:
            weight_init = self.winit

        if isinstance(self.binit, str):
            if self.binit == 'zeros':
                bias_init = Initializer(dist='zeros')
            else:
                raise ValueError(f"Unknown binit: {self.binit}")
        else:
            bias_init = self.binit

        # Input processing: combine stoch + action → transformer input
        self.input_stem = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=weight_init, bias_init=bias_init, dtype=COMPUTE_DTYPE),
            nn.LayerNorm(dtype=COMPUTE_DTYPE),
            self._get_activation(),
            nn.Dense(self.hidden_dim, kernel_init=weight_init, bias_init=bias_init, dtype=COMPUTE_DTYPE),
            nn.LayerNorm(dtype=COMPUTE_DTYPE),
        ])

        # Positional encoding (learnable)
        self.pos_encoding = self.param(
            'pos_encoding',
            nn.initializers.normal(stddev=0.02),
            (self.max_length, self.hidden_dim)
        )

        # Transformer layers with KV-cache support
        self.transformer_blocks = [
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                use_bias=True,
                max_cache_length=self.max_cache_length,
                name=f'transformer_{i}'
            )
            for i in range(self.num_layers)
        ]

        # Prior head: predicts next stoch from transformer hidden state
        self.prior_head = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=weight_init, bias_init=bias_init, dtype=COMPUTE_DTYPE),
            nn.LayerNorm(dtype=COMPUTE_DTYPE),
            self._get_activation(),
            nn.Dense(self.stoch * self.classes, kernel_init=weight_init, bias_init=bias_init, dtype=COMPUTE_DTYPE),
        ])

        # Posterior head: combines transformer hidden + observation embedding
        self.posterior_stem = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=weight_init, bias_init=bias_init, dtype=COMPUTE_DTYPE),
            nn.LayerNorm(dtype=COMPUTE_DTYPE),
            self._get_activation(),
        ])

        self.posterior_head = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=weight_init, bias_init=bias_init, dtype=COMPUTE_DTYPE),
            nn.LayerNorm(dtype=COMPUTE_DTYPE),
            self._get_activation(),
            nn.Dense(self.stoch * self.classes, kernel_init=weight_init, bias_init=bias_init, dtype=COMPUTE_DTYPE),
        ])

    def _get_activation(self):
        """Get activation function."""
        if self.act == 'gelu':
            return nn.gelu
        elif self.act == 'relu':
            return nn.relu
        elif self.act == 'elu':
            return nn.elu
        elif self.act == 'silu':
            return nn.silu
        else:
            raise ValueError(f"Unknown activation: {self.act}")

    def initial_state(self, batch_size: int) -> TSSMState:
        """Create initial (zero) latent state.

        Args:
            batch_size: Batch dimension size

        Returns:
            TSSMState with zeros (including zero logits for PyTree structure consistency)
        """
        return TSSMState(
            stoch=jnp.zeros((batch_size, self.stoch, self.classes), dtype=COMPUTE_DTYPE),
            hidden=jnp.zeros((batch_size, self.hidden_dim), dtype=COMPUTE_DTYPE),
            # Fixed-size cache for JAX scan compatibility
            kv_cache=[
                {'keys': jnp.zeros((batch_size, self.max_cache_length, self.hidden_dim), dtype=COMPUTE_DTYPE),
                 'values': jnp.zeros((batch_size, self.max_cache_length, self.hidden_dim), dtype=COMPUTE_DTYPE),
                 'step': jnp.zeros((batch_size,), dtype=jnp.int32)}
                for _ in range(self.num_layers)
            ],
            cache_step=jnp.zeros((batch_size,), dtype=jnp.int32),
            # Initialize logits with zeros (not None) to maintain consistent PyTree structure
            # This is important for JAX scan operations where carry must have same structure
            logits=jnp.zeros((batch_size, self.stoch, self.classes), dtype=COMPUTE_DTYPE)
        )

    def extract_context_windows(
        self,
        full_kv: jnp.ndarray,
        context_len: int
    ) -> jnp.ndarray:
        """Extract context windows from full sequence K/V tensors.

        Efficiently slices [B, T, ...] into [B*T, C, ...] where each of the B*T
        positions gets a context window of length C ending at that position.

        Args:
            full_kv: Full sequence keys or values [B, T, H, D] or [B, T, D]
            context_len: Context window length C

        Returns:
            windowed_kv: Context windows [B*T, C, ...] where each position has its context
        """
        shape = full_kv.shape
        B, T = shape[0], shape[1]
        remaining_dims = shape[2:]

        # Pad beginning with zeros so early timesteps have zero-context
        # [B, T + C - 1, ...]
        pad_width = [(0, 0), (context_len - 1, 0)] + [(0, 0)] * len(remaining_dims)
        padded_kv = jnp.pad(full_kv, pad_width, mode='constant', constant_values=0)

        # Extract sliding windows for each timestep
        # For t=0: padded[0:C], t=1: padded[1:C+1], ..., t=T-1: padded[T-1:T+C-1]
        def extract_window(t):
            # Extract window [t : t+C] from padded sequence
            # Returns [B, C, ...]
            return jax.lax.dynamic_slice_in_dim(padded_kv, t, context_len, axis=1)

        # Vectorize over all timesteps: [T, B, C, ...]
        windows = jax.vmap(extract_window)(jnp.arange(T))

        # Reshape: [T, B, C, ...] -> [B, T, C, ...] -> [B*T, C, ...]
        windows = windows.transpose(1, 0, *range(2, len(windows.shape)))
        flat_shape = (B * T, context_len) + remaining_dims
        windows = windows.reshape(flat_shape)

        return windows

    def observe(
        self,
        prev_state: TSSMState,
        action: jnp.ndarray,
        embed: jnp.ndarray,
        is_first: jnp.ndarray,
        training: bool,
        rng: jax.random.PRNGKey
    ) -> Tuple[TSSMState, Dict]:
        """Update state with new observation (posterior inference).

        This is the "encoder" path: given observation, infer latent state.
        Computes both prior (from prev state + action) and posterior (from embed).

        Args:
            prev_state: Previous latent state
            action: Action taken [B, action_dim] (one-hot or continuous)
            embed: Observation embedding from encoder [B, embed_dim]
            is_first: Episode reset flags [B,]
            training: Whether in training mode
            rng: Random key for stochastic sampling

        Returns:
            new_state: Updated latent state (posterior)
            info: Dict with 'prior', 'posterior', 'dyn_loss', 'rep_loss'
        """
        batch_size = action.shape[0]

        # Reset state if is_first
        # Expand is_first to match tensor shapes for broadcasting
        is_first_stoch = is_first[:, None, None]  # [B, 1, 1] for [B, stoch, classes]
        is_first_hidden = is_first[:, None]        # [B, 1] for [B, hidden_dim]

        # Reset scalar values
        stoch = jnp.where(is_first_stoch, 0, prev_state.stoch)
        hidden = jnp.where(is_first_hidden, 0, prev_state.hidden)

        # Reset KV-cache for batch elements where is_first=True
        # Note: This zeros out cache values but keeps tensor shapes. Combined with
        # ring buffer (added below), this prevents unbounded memory growth.
        is_first_expanded = is_first[:, None, None]  # [B, 1, 1] for broadcasting

        kv_cache = []
        for layer_cache in prev_state.kv_cache:
            # Zero out cache entries where is_first=True
            keys = jnp.where(is_first_expanded, 0.0, layer_cache['keys'])
            values = jnp.where(is_first_expanded, 0.0, layer_cache['values'])
            # Reset step counter where is_first=True
            step = jnp.where(is_first, 0, layer_cache.get('step', jnp.zeros((batch_size,), dtype=jnp.int32)))
            kv_cache.append({'keys': keys, 'values': values, 'step': step})

        # Prepare input: flatten stoch and concatenate with action
        stoch_flat = stoch.reshape(batch_size, self.stoch * self.classes)

        # Ensure action is the right shape
        if action.ndim == 1:
            action = jnp.expand_dims(action, -1)

        prior_input = jnp.concatenate([stoch_flat, action], axis=-1)

        # Process through input stem
        transformer_input = self.input_stem(prior_input)

        # Add positional encoding based on logical step count
        # Use cache_step from state, clamped to max_length
        # cache_step is [B], pos_encoding is [max_length, hidden_dim]
        # We need to select specific embedding for each batch element
        seq_position = jnp.minimum(prev_state.cache_step, self.max_length - 1)
        pos = self.pos_encoding[seq_position] # [B, hidden_dim] via advanced indexing
        transformer_input = transformer_input + pos

        # Expand to sequence dimension for transformer blocks
        transformer_input = jnp.expand_dims(transformer_input, 1)  # [B, 1, hidden_dim]

        # Process through transformer layers with KV-cache
        hidden = transformer_input
        new_kv_cache = []
        for i, block in enumerate(self.transformer_blocks):
            hidden, new_cache = block(hidden, kv_cache[i], training=training)
            new_kv_cache.append(new_cache)

        # Remove sequence dimension and ensure dtype consistency
        hidden = jnp.squeeze(hidden, 1)  # [B, hidden_dim]
        hidden = hidden.astype(COMPUTE_DTYPE)  # Ensure hidden stays in COMPUTE_DTYPE

        # Compute prior logits
        prior_logits = self.prior_head(hidden).reshape(batch_size, self.stoch, self.classes)

        # Compute posterior: observation-only (no hidden state dependency!)
        # This matches STORM's architecture where posterior q(z_t | x_t) only depends on observation
        embed = embed.astype(COMPUTE_DTYPE)
        post_hidden = self.posterior_stem(embed)
        post_logits = self.posterior_head(post_hidden).reshape(batch_size, self.stoch, self.classes)

        # Create distributions
        prior_dist = OneHotDist(prior_logits, unimix=self.unimix)
        post_dist = OneHotDist(post_logits, unimix=self.unimix)

        # Sample from posterior (ensure output is in COMPUTE_DTYPE)
        if training:
            stoch_sample = post_dist.sample(seed=rng).astype(COMPUTE_DTYPE)
        else:
            stoch_sample = post_dist.mode().astype(COMPUTE_DTYPE)

        # Extract cache_step from first layer's cache (all layers have same step)
        # Reset step counter where is_first=True
        new_cache_step = new_kv_cache[0].get('step', prev_state.cache_step)
        new_cache_step = jnp.where(is_first, 0, new_cache_step)

        # Create new state (ensure all components are in COMPUTE_DTYPE)
        new_state = TSSMState(
            stoch=stoch_sample,
            hidden=hidden.astype(COMPUTE_DTYPE),
            kv_cache=new_kv_cache,
            cache_step=new_cache_step,
            logits=post_logits.astype(COMPUTE_DTYPE)
        )

        # Compute KL divergence for losses
        kl = self.kl_divergence(post_dist, prior_dist)

        # Return state and info
        info = {
            'prior': prior_dist,
            'posterior': post_dist,
            'dyn_loss': kl['dyn'],
            'rep_loss': kl['rep']
        }

        return new_state, info

    def observe_sequence(
        self,
        initial_state: TSSMState,
        actions: jnp.ndarray,
        embeds: jnp.ndarray,
        is_first: jnp.ndarray,
        training: bool,
        rng: jax.random.PRNGKey
    ) -> Tuple[TSSMState, Dict]:
        """Process full sequence with parallel Transformer computation.

        This method leverages the Transformer's ability to process sequences in parallel
        with causal masking, unlike RNN-based models that require sequential processing.

        Key insight: We prepare inputs by shifting the stochastic sequence and prepending
        the initial state, allowing the Transformer to process all timesteps at once.

        Args:
            initial_state: Initial latent state [B, ...]
            actions: Action sequence [B, T, action_dim]
            embeds: Observation embedding sequence [B, T, embed_dim]
            is_first: Episode reset flags [B, T]
            training: Whether in training mode
            rng: Random key for stochastic sampling

        Returns:
            states: TSSMState with sequence dimension [B, T, ...]
            info: Dict with 'prior', 'posterior', 'dyn_loss', 'rep_loss' for all timesteps
        """
        batch_size, seq_len = actions.shape[:2]

        # ============================================================
        # Phase 1: Compute ALL posteriors in parallel
        # KEY: Posterior q(z_t | x_t) depends ONLY on observation
        # ============================================================
        embeds = embeds.astype(COMPUTE_DTYPE)  # [B, T, embed_dim]

        # Posterior is observation-only (no hidden state dependency!)
        post_hidden = self.posterior_stem(embeds)  # [B, T, hidden_dim]
        post_logits_seq = self.posterior_head(post_hidden).reshape(
            batch_size, seq_len, self.stoch, self.classes
        )  # [B, T, stoch, classes]

        # Sample stochastic states (vectorized over time)
        rngs = jax.random.split(rng, batch_size * seq_len).reshape(batch_size, seq_len, -1)

        def sample_step(logits, rng_t):
            dist = OneHotDist(logits, unimix=self.unimix)
            if training:
                return dist.sample(seed=rng_t).astype(COMPUTE_DTYPE)
            else:
                return dist.mode().astype(COMPUTE_DTYPE)

        # Vectorize over both batch and time dimensions
        stochs = jax.vmap(jax.vmap(sample_step))(post_logits_seq, rngs)  # [B, T, stoch, classes]

        # ============================================================
        # Phase 2: Teacher Forcing
        # ============================================================
        # Shift right: Input at time t uses stoch_{t-1}
        initial_stoch = initial_state.stoch[:, None, :, :]  # [B, 1, stoch, classes]
        shifted_stochs = jnp.concatenate([initial_stoch, stochs[:, :-1]], axis=1)  # [B, T, stoch, classes]
        shifted_stochs = jax.lax.stop_gradient(shifted_stochs)

        # Flatten and concatenate with actions
        shifted_stochs_flat = shifted_stochs.reshape(batch_size, seq_len, self.stoch * self.classes)
        transformer_inputs = jnp.concatenate([shifted_stochs_flat, actions], axis=-1)

        # ============================================================
        # Phase 3: Transformer (PARALLEL!)
        # ============================================================
        transformer_inputs = self.input_stem(transformer_inputs)  # [B, T, hidden_dim]

        # Add positional encoding (0 to T-1)
        # pos_encoding: [max_length, hidden_dim]
        # We take first T embeddings and broadcast to batch
        pos_emb = self.pos_encoding[:seq_len]  # [T, hidden_dim]
        pos_emb = jnp.expand_dims(pos_emb, 0)  # [1, T, hidden_dim]
        transformer_inputs = transformer_inputs + pos_emb

        # Process entire sequence with causal masking and collect K/V caches
        hiddens = transformer_inputs
        collected_caches = []
        for block in self.transformer_blocks:
            hiddens, cache = block(hiddens, kv_cache=None, training=training)
            # Store the full sequence K/V for context window extraction
            collected_caches.append(cache)
        hiddens = hiddens.astype(COMPUTE_DTYPE)  # [B, T, hidden_dim]

        # ============================================================
        # Phase 4: Compute priors (PARALLEL!)
        # ============================================================
        prior_logits = self.prior_head(hiddens).reshape(
            batch_size, seq_len, self.stoch, self.classes
        )  # [B, T, stoch, classes]

        # ============================================================
        # Phase 5: Compute KL divergences (vectorized)
        # ============================================================
        def compute_kl(post_logits, prior_logits):
            post_dist = OneHotDist(post_logits, unimix=self.unimix)
            prior_dist = OneHotDist(prior_logits, unimix=self.unimix)
            kl = self.kl_divergence(post_dist, prior_dist)
            return kl['dyn'], kl['rep']

        # Vectorize over both batch and time
        dyn_losses, rep_losses = jax.vmap(jax.vmap(compute_kl))(post_logits_seq, prior_logits)
        # Results are [B, T]

        # Store full sequence K/V caches for context window extraction
        # Each cache in collected_caches has keys/values of shape [B, T, hidden_dim]
        # These will be used to extract context windows for imagination
        kv_cache = collected_caches if training else [
            {'keys': jnp.zeros((batch_size, 0, self.hidden_dim), dtype=COMPUTE_DTYPE),
             'values': jnp.zeros((batch_size, 0, self.hidden_dim), dtype=COMPUTE_DTYPE),
             'step': jnp.zeros((batch_size,), dtype=jnp.int32)}  # Include step for PyTree consistency
            for _ in range(self.num_layers)
        ]

        # Create states with sequence dimension
        # cache_step represents the length of the sequence at each timestep
        # For parallel training, we don't track step-by-step but T as a whole
        cache_step_seq = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)  # [B, T]

        states = TSSMState(
            stoch=stochs,
            hidden=hiddens,
            kv_cache=kv_cache,
            cache_step=cache_step_seq,
            logits=post_logits_seq
        )

        # Info dict
        info = {
            'dyn_loss': dyn_losses,  # [B, T]
            'rep_loss': rep_losses,  # [B, T]
        }

        return states, info

    def imagine(
        self,
        state: TSSMState,
        action: jnp.ndarray,
        training: bool,
        rng: jax.random.PRNGKey
    ) -> Tuple[TSSMState, Dict]:
        """Predict next state without observation (prior prediction).

        This is the "decoder" path: given current state + action, predict next state.
        Used during imagination rollouts for policy learning.

        Args:
            state: Current latent state
            action: Action to take [B, action_dim] (one-hot or continuous)
            training: Whether in training mode
            rng: Random key for stochastic sampling

        Returns:
            next_state: Predicted next state (prior)
            info: Dict with 'prior'
        """
        batch_size = action.shape[0]

        # Prepare input: flatten stoch and concatenate with action
        stoch_flat = state.stoch.reshape(batch_size, self.stoch * self.classes)

        # Ensure action is the right shape
        if action.ndim == 1:
            action = jnp.expand_dims(action, -1)

        prior_input = jnp.concatenate([stoch_flat, action], axis=-1)

        # Process through input stem
        transformer_input = self.input_stem(prior_input)

        # Add positional encoding based on logical step count
        seq_position = jnp.minimum(state.cache_step, self.max_length - 1)
        pos = self.pos_encoding[seq_position] # [B, hidden_dim]
        transformer_input = transformer_input + pos

        # Expand to sequence dimension
        transformer_input = jnp.expand_dims(transformer_input, 1)  # [B, 1, hidden_dim]

        # Process through transformer layers with KV-cache
        hidden = transformer_input
        new_kv_cache = []
        for i, block in enumerate(self.transformer_blocks):
            hidden, new_cache = block(hidden, state.kv_cache[i], training=training)
            new_kv_cache.append(new_cache)

        # Remove sequence dimension and ensure dtype consistency
        hidden = jnp.squeeze(hidden, 1)  # [B, hidden_dim]
        hidden = hidden.astype(COMPUTE_DTYPE)  # Ensure hidden stays in COMPUTE_DTYPE

        # Compute prior logits
        prior_logits = self.prior_head(hidden).reshape(batch_size, self.stoch, self.classes)

        # Create distribution
        prior_dist = OneHotDist(prior_logits, unimix=self.unimix)

        # Sample from prior (ensure output is in COMPUTE_DTYPE)
        if training:
            stoch_sample = prior_dist.sample(seed=rng).astype(COMPUTE_DTYPE)
        else:
            stoch_sample = prior_dist.mode().astype(COMPUTE_DTYPE)

        # Extract cache_step from first layer's cache (all layers have same step)
        new_cache_step = new_kv_cache[0].get('step', state.cache_step)

        # Create new state (ensure all components are in COMPUTE_DTYPE)
        new_state = TSSMState(
            stoch=stoch_sample,
            hidden=hidden.astype(COMPUTE_DTYPE),
            kv_cache=new_kv_cache,
            cache_step=new_cache_step,
            logits=prior_logits.astype(COMPUTE_DTYPE)
        )

        # Return state and info
        info = {'prior': prior_dist}

        return new_state, info

    def kl_divergence(self, post_dist: OneHotDist, prior_dist: OneHotDist) -> Dict:
        """Compute KL divergence between posterior and prior.

        Uses free nats and stop gradients to balance dynamics and representation learning.
        This matches RSSM's implementation exactly - using OneHotDist's kl_divergence method
        which correctly sums over the stoch dimension (32 categoricals).

        Args:
            post_dist: Posterior distribution
            prior_dist: Prior distribution

        Returns:
            Dict with 'dyn' and 'rep' losses for different gradient flows
        """
        # Create distributions with stop_gradient on logits for selective gradient flow
        post_sg = OneHotDist(jax.lax.stop_gradient(post_dist.logits), unimix=self.unimix)
        prior_sg = OneHotDist(jax.lax.stop_gradient(prior_dist.logits), unimix=self.unimix)

        # Dynamics loss: D(sg(post) || prior) - gradients flow to prior (dynamics model)
        dyn = post_sg.kl_divergence(prior_dist)

        # Representation loss: D(post || sg(prior)) - gradients flow to post (encoder)
        rep = post_dist.kl_divergence(prior_sg)

        # Apply free nats (minimum KL to prevent posterior collapse)
        dyn = jnp.maximum(dyn, self.free_nats)
        rep = jnp.maximum(rep, self.free_nats)

        return {'dyn': dyn, 'rep': rep}

    def get_feat(self, state: TSSMState) -> jnp.ndarray:
        """Extract features for policy/value networks.

        Concatenates transformer hidden state with flattened stochastic state.

        Args:
            state: TSSM state

        Returns:
            features: [B, hidden_dim + stoch * classes] or [B, T, hidden_dim + stoch * classes]
        """
        # Handle both (batch, ...) and (batch, seq_len, ...) shapes
        stoch_shape = state.stoch.shape
        if len(stoch_shape) == 3:  # (batch, stoch, classes)
            stoch_flat = state.stoch.reshape(stoch_shape[0], self.stoch * self.classes)
        elif len(stoch_shape) == 4:  # (batch, seq_len, stoch, classes)
            stoch_flat = state.stoch.reshape(stoch_shape[0], stoch_shape[1], self.stoch * self.classes)
        else:
            raise ValueError(f"Unexpected stoch shape: {stoch_shape}")

        return jnp.concatenate([state.hidden, stoch_flat], axis=-1)
