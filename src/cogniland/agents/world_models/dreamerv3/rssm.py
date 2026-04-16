"""RSSM (Recurrent State-Space Model) for DreamerV3."""

from typing import Tuple, Optional, Dict

import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
import einops

from cogniland.agents.commons.initializer import Initializer
from cogniland.agents.commons.distributions import OneHotDist
from cogniland.agents.commons.networks.blocks import BlockLinear
from cogniland.config.jax_config import COMPUTE_DTYPE


@chex.dataclass
class RSSMState:
    """State of the RSSM."""

    deter: jnp.ndarray  # Deterministic state (GRU hidden state)
    stoch: jnp.ndarray  # Stochastic state (discrete latent)
    logits: jnp.ndarray  # Logits for stochastic state


class RSSM(nn.Module):
    """
    Recurrent State-Space Model for DreamerV3.

    The RSSM combines a deterministic BlockGRU with stochastic categorical latents
    to model environment dynamics.
    """

    hidden: int = 2048          # Hidden size for MLPs (larger than in base model)
    deter: int = 4096           # GRU hidden size
    stoch: int = 32             # Number of categorical distributions
    classes: int = 32           # Classes per categorical
    blocks: int = 8             # Number of blocks for BlockGRU
    norm: str = 'rms'           # Normalization type
    act: str = 'gelu'           # Activation function
    unimix: float = 0.01        # Uniform mixture for regularization
    outscale: float = 1.0       # Output scale
    imglayers: int = 2          # Layers for prior from deterministic state
    obslayers: int = 1          # Layers for posterior from observations
    dynlayers: int = 1          # Layers for dynamics
    free_nats: float = 1.0      # Free nats for KL divergence
    winit: str = 'trunc_normal_in'  # Weight initialization
    binit: str = 'zeros'        # Bias initialization

    def setup(self):
        """Initialize layers."""
        assert self.deter % self.blocks == 0, (
            f"deter ({self.deter}) must be divisible by blocks ({self.blocks})"
        )

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

        # Output initializer (scaled or zeros)
        if self.outscale > 0:
            output_init = Initializer(dist='trunc_normal', fan='in', scale=self.outscale)
        else:
            output_init = Initializer(dist='zeros')

        # Input processing layers
        self.dynin0 = nn.Dense(
            self.hidden,
            kernel_init=weight_init,
            bias_init=bias_init,
            dtype=COMPUTE_DTYPE
        )
        self.dynin0_norm = self._get_norm('dynin0_norm')

        self.dynin1 = nn.Dense(
            self.hidden,
            kernel_init=weight_init,
            bias_init=bias_init,
            dtype=COMPUTE_DTYPE
        )
        self.dynin1_norm = self._get_norm('dynin1_norm')

        self.dynin2 = nn.Dense(
            self.hidden,
            kernel_init=weight_init,
            bias_init=bias_init,
            dtype=COMPUTE_DTYPE
        )
        self.dynin2_norm = self._get_norm('dynin2_norm')

        # Dynamics layers (BlockLinear for efficiency)
        self.dyn_blocks = [
            BlockLinear(
                self.deter,
                self.blocks,
                kernel_init=weight_init,
                bias_init=bias_init,
                name=f'dynhid{i}'
            )
            for i in range(self.dynlayers)
        ]
        self.dyn_norms = [
            self._get_norm(f'dynhid{i}_norm')
            for i in range(self.dynlayers)
        ]

        # BlockGRU for core dynamics
        self.gru = BlockLinear(
            3 * self.deter,
            self.blocks,
            kernel_init=weight_init,
            bias_init=bias_init,
            name='dyngru'
        )

        # Prior network: p(z_t | h_t)
        self.prior_layers = [
            nn.Dense(
                self.hidden,
                kernel_init=weight_init,
                bias_init=bias_init,
                dtype=COMPUTE_DTYPE,
                name=f'prior{i}'
            )
            for i in range(self.imglayers)
        ]
        self.prior_norms = [
            self._get_norm(f'prior{i}_norm')
            for i in range(self.imglayers)
        ]
        self.prior_logits = nn.Dense(
            self.stoch * self.classes,
            kernel_init=output_init,
            bias_init=bias_init,
            dtype=COMPUTE_DTYPE,
            name='priorlogit'
        )

        # Posterior network: q(z_t | h_t, o_t)
        self.obs_layers = [
            nn.Dense(
                self.hidden,
                kernel_init=weight_init,
                bias_init=bias_init,
                dtype=COMPUTE_DTYPE,
                name=f'obs{i}'
            )
            for i in range(self.obslayers)
        ]
        self.obs_norms = [
            self._get_norm(f'obs{i}_norm')
            for i in range(self.obslayers)
        ]
        self.obs_logits = nn.Dense(
            self.stoch * self.classes,
            kernel_init=output_init,
            bias_init=bias_init,
            dtype=COMPUTE_DTYPE,
            name='obslogit'
        )

    def initial_state(self, batch_size: int) -> RSSMState:
        """
        Create initial RSSM state.

        Args:
            batch_size: Batch size

        Returns:
            Initial RSSMState with zeros
        """
        return RSSMState(
            deter=jnp.zeros((batch_size, self.deter), dtype=COMPUTE_DTYPE),
            stoch=jnp.zeros((batch_size, self.stoch, self.classes), dtype=COMPUTE_DTYPE),
            logits=jnp.zeros((batch_size, self.stoch * self.classes), dtype=COMPUTE_DTYPE),
        )

    def __call__(
        self,
        prev_state: RSSMState,
        action: jnp.ndarray,
        embed: Optional[jnp.ndarray] = None,
        is_first: Optional[jnp.ndarray] = None,
        training: bool = True,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[RSSMState, RSSMState]:
        """
        Forward pass through RSSM.

        Args:
            prev_state: Previous RSSMState
            action: Action (one-hot or continuous)
            embed: Embedding from encoder (for posterior)
            is_first: Mask for first timestep (batch,)
            training: Whether in training mode
            rng: Random key for sampling (required if training=True)

        Returns:
            Tuple of (posterior_state, prior_state)
        """
        # Reset state if first timestep
        if is_first is not None:
            # Expand is_first to match state dimensions
            is_first_exp = is_first[..., None]
            prev_state = RSSMState(
                deter=prev_state.deter * (1 - is_first_exp),
                stoch=prev_state.stoch * (1 - is_first_exp[..., None]),
                logits=prev_state.logits * (1 - is_first_exp),
            )

        # Update deterministic state with BlockGRU
        deter = self._core(prev_state.deter, prev_state.stoch, action)

        # Split RNG if needed
        if rng is not None and training:
            rng, prior_rng, post_rng = jax.random.split(rng, 3)
        else:
            prior_rng = post_rng = None

        # Compute prior: p(z_t | h_t)
        prior_logits = self._prior(deter)
        prior_stoch = self._sample_stochastic(prior_logits, training, prior_rng)
        prior_state = RSSMState(
            deter=deter,
            stoch=prior_stoch,
            logits=prior_logits,
        )

        # If no embedding provided, return prior as posterior
        if embed is None:
            return prior_state, prior_state

        # Compute posterior: q(z_t | h_t, o_t)
        posterior_logits = self._posterior(deter, embed)
        posterior_stoch = self._sample_stochastic(posterior_logits, training, post_rng)
        posterior_state = RSSMState(
            deter=deter,
            stoch=posterior_stoch,
            logits=posterior_logits,
        )

        return posterior_state, prior_state

    def _core(
        self,
        deter: jnp.ndarray,
        stoch: jnp.ndarray,
        action: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Update deterministic state using BlockGRU.

        Args:
            deter: Previous deterministic state (batch, deter)
            stoch: Previous stochastic state (batch, stoch, classes)
            action: Action (batch, action_dim)

        Returns:
            Updated deterministic state
        """
        # Flatten stochastic state
        stoch_flat = stoch.reshape(stoch.shape[0], -1)

        # Normalize action by its magnitude (helps stability)
        # Element-wise normalization (matches original DreamerV3)
        action = action / jax.lax.stop_gradient(jnp.maximum(1, jnp.abs(action)))

        # Process inputs through separate pathways
        x0 = self.dynin0(deter)
        x0 = self._get_activation()(self.dynin0_norm(x0))

        x1 = self.dynin1(stoch_flat)
        x1 = self._get_activation()(self.dynin1_norm(x1))

        x2 = self.dynin2(action)
        x2 = self._get_activation()(self.dynin2_norm(x2))

        # Combine inputs
        # Broadcast to all blocks
        g = self.blocks
        combined = jnp.concatenate([x0, x1, x2], axis=-1)
        combined = combined[..., None, :].repeat(g, axis=-2)  # (batch, groups, hidden)

        # Reshape for block processing
        flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
        group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)

        # Combine with deterministic state
        x = group2flat(jnp.concatenate([flat2group(deter), combined], axis=-1))

        # Apply dynamics layers
        for i, (layer, norm) in enumerate(zip(self.dyn_blocks, self.dyn_norms)):
            x = layer(x)
            x = self._get_activation()(norm(x))

        # GRU gates (following original DreamerV3 structure exactly)
        gates = self.gru(x)
        gates_grouped = flat2group(gates)
        reset_g, cand_g, update_g = jnp.split(gates_grouped, 3, axis=-1)

        # Flatten all gates FIRST (like original)
        reset = group2flat(reset_g)
        cand = group2flat(cand_g)
        update = group2flat(update_g)

        # Then apply activations (exactly like original)
        reset = nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = nn.sigmoid(update - 1)  # Bias towards not updating

        # Update deterministic state
        deter = update * cand + (1 - update) * deter

        return deter

    def _prior(self, deter: jnp.ndarray) -> jnp.ndarray:
        """
        Compute prior distribution logits from deterministic state.

        Args:
            deter: Deterministic state (batch, deter)

        Returns:
            Prior logits (batch, stoch * classes)
        """
        x = deter
        for layer, norm in zip(self.prior_layers, self.prior_norms):
            x = layer(x)
            x = self._get_activation()(norm(x))

        logits = self.prior_logits(x)
        return logits

    def _posterior(
        self,
        deter: jnp.ndarray,
        embed: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute posterior distribution logits from deterministic state and observations.

        Args:
            deter: Deterministic state (batch, deter)
            embed: Observation embedding (batch, embed_size)

        Returns:
            Posterior logits (batch, stoch * classes)
        """
        # Concatenate deterministic state and embedding
        x = jnp.concatenate([deter, embed], axis=-1)

        for layer, norm in zip(self.obs_layers, self.obs_norms):
            x = layer(x)
            x = self._get_activation()(norm(x))

        logits = self.obs_logits(x)
        return logits

    def _sample_stochastic(
        self,
        logits: jnp.ndarray,
        training: bool = True,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Sample stochastic state from logits.

        Args:
            logits: Logits of shape (batch, stoch * classes)
            training: Whether to sample or use mode
            rng: Random key for sampling (required if training=True)

        Returns:
            Stochastic state of shape (batch, stoch, classes)
        """
        # Reshape to separate distributions
        logits = logits.reshape(-1, self.stoch, self.classes)

        # Create distribution with unimix (handles uniform mixing internally)
        dist = OneHotDist(logits, unimix=self.unimix)

        if training and rng is not None:
            # Sample from distribution
            stoch = dist.sample(seed=rng)
        else:
            # Use mode (argmax)
            stoch = dist.mode()

        return stoch

    def get_feat(self, state: RSSMState) -> jnp.ndarray:
        """
        Get feature vector from RSSM state.

        Args:
            state: RSSMState

        Returns:
            Concatenated deterministic and stochastic features
        """
        # Handle both (batch, ...) and (batch, seq_len, ...) shapes
        stoch_shape = state.stoch.shape
        if len(stoch_shape) == 3:  # (batch, stoch, classes)
            stoch_flat = state.stoch.reshape(stoch_shape[0], -1)
        elif len(stoch_shape) == 4:  # (batch, seq_len, stoch, classes)
            stoch_flat = state.stoch.reshape(stoch_shape[0], stoch_shape[1], -1)
        else:
            raise ValueError(f"Unexpected stoch shape: {stoch_shape}")
        return jnp.concatenate([state.deter, stoch_flat], axis=-1)

    def get_dist(self, state: RSSMState) -> OneHotDist:
        """
        Get distribution from state logits.

        Args:
            state: RSSMState

        Returns:
            OneHotDist distribution
        """
        logits = state.logits.reshape(-1, self.stoch, self.classes)
        return OneHotDist(logits, unimix=self.unimix)

    def kl_divergence(
        self,
        post: RSSMState,
        prior: RSSMState,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute KL divergence losses with balancing.

        Args:
            post: Posterior state
            prior: Prior state

        Returns:
            Dictionary with 'dyn' and 'rep' losses
        """
        # Apply stop_gradient as in original DreamerV3 equation 3
        # Dynamics loss: D(sg(post) || prior) - gradients flow to prior (dynamics model)
        post_sg = jax.tree.map(jax.lax.stop_gradient, post)
        dyn = self.get_dist(post_sg).kl_divergence(self.get_dist(prior))

        # Representation loss: D(post || sg(prior)) - gradients flow to post (encoder)
        prior_sg = jax.tree.map(jax.lax.stop_gradient, prior)
        rep = self.get_dist(post).kl_divergence(self.get_dist(prior_sg))

        # Apply free nats (as in original DreamerV3)
        if self.free_nats > 0:
            dyn = jnp.maximum(dyn, self.free_nats)
            rep = jnp.maximum(rep, self.free_nats)

        return {'dyn': dyn, 'rep': rep}

    def _get_norm(self, name: str) -> nn.Module:
        """Get normalization layer."""
        if self.norm == 'rms':
            return nn.RMSNorm(name=name, dtype=COMPUTE_DTYPE)
        elif self.norm == 'layer':
            return nn.LayerNorm(name=name, dtype=COMPUTE_DTYPE)
        elif self.norm == 'none':
            return lambda x: x
        else:
            raise ValueError(f"Unknown normalization: {self.norm}")

    def _get_activation(self):
        """Get activation function."""
        if self.act == 'gelu':
            return nn.gelu
        elif self.act == 'silu':
            return nn.silu
        elif self.act == 'elu':
            return nn.elu
        elif self.act == 'relu':
            return nn.relu
        else:
            raise ValueError(f"Unknown activation: {self.act}")