"""STORM world model implementation using the WorldModel protocol.

This module wraps STORM's components (Encoder, Decoder, TSSM, MLP heads) to
implement the WorldModel protocol defined in base.py.

Key differences from DreamerV3WorldModel:
- Uses TSSM (Transformer State-Space Model) instead of RSSM
- Uses BatchNorm in CNN layers instead of RMSNorm
- get_feat() concatenates hidden (deter) + stoch
- Supports multi-modal observations (images + vectors)
"""

from typing import Any, Dict, Optional, Tuple
import jax
import jax.numpy as jnp

from cl.agents.world_models.base import WorldModel, WorldModelParams, WorldModelState
from cl.agents.world_models.storm.tssm import TSSM
from cl.agents.world_models.storm.encoder import STORMEncoder
from cl.agents.world_models.storm.decoder import STORMDecoder
from cl.agents.commons.networks.mlp import MLPHead
from cl.config.jax_config import COMPUTE_DTYPE


class STORMWorldModel(WorldModel):
    """STORM world model implementation.

    Combines:
        - Multi-modal Encoder (MLP + CNN with BatchNorm)
        - Multi-modal Decoder (MLP + transposed CNN)
        - TSSM dynamics (Transformer + categorical latents + KV-cache)
        - Reward and continuation predictors (MLP heads)

    This class is a STATELESS WRAPPER that implements the WorldModel protocol.
    All methods take parameters explicitly (Flax functional style).

    Args:
        obs_space: Observation space specification
        action_space: Number of discrete actions
        encoder_config: Configuration for encoder (units, depth, etc.)
        decoder_config: Configuration for decoder
        tssm_config: Configuration for TSSM (hidden_dim, stoch, classes, etc.)
        reward_config: Configuration for reward head
        cont_config: Configuration for continuation head
    """

    def __init__(
        self,
        obs_space: Dict,
        action_space: int,
        encoder_config: Dict,
        decoder_config: Dict,
        tssm_config: Dict,
        reward_config: Dict,
        cont_config: Dict,
    ):
        # Store config
        self.obs_space = obs_space
        self.action_space = action_space

        # Normalize obs_space to consistent format {key: tuple_shape}
        self.obs_shapes = {}
        for key, spec in obs_space.items():
            if isinstance(spec, dict):
                self.obs_shapes[key] = tuple(spec['shape'])
            else:
                self.obs_shapes[key] = tuple(spec)

        # Extract image shape if present (for backward compatibility)
        self.image_shape = None
        for key, shape in self.obs_shapes.items():
            if len(shape) == 3:  # H, W, C
                self.image_shape = shape
                break

        # Get flatten_keys and discrete_keys from encoder/decoder config
        flatten_keys = tuple(encoder_config.get('flatten_keys', ()))
        discrete_keys = tuple(decoder_config.get('discrete_keys', ('direction',)))

        # Create Flax modules (specs only, no params)
        encoder_config = dict(encoder_config)  # Make mutable copy
        if self.image_shape is not None:
            encoder_config['image_size'] = self.image_shape[0]
        encoder_config['flatten_keys'] = flatten_keys
        self.encoder = STORMEncoder(**encoder_config)

        # Decoder needs stoch_dim from tssm_config
        stoch_dim = tssm_config.get('stoch', 32) * tssm_config.get('classes', 32)
        decoder_config = dict(decoder_config)  # Make mutable copy
        if self.image_shape is not None:
            decoder_config['image_size'] = self.image_shape[0]
        decoder_config['stoch_dim'] = stoch_dim
        decoder_config['obs_shapes'] = self.obs_shapes
        decoder_config['discrete_keys'] = discrete_keys
        decoder_config['flatten_keys'] = flatten_keys
        self.decoder = STORMDecoder(**decoder_config)

        # TSSM dynamics (transformer-based)
        self.tssm = TSSM(action_dim=action_space, **tssm_config)

        # Reward and continuation heads
        self.reward_head = MLPHead(output_shape=(1,), **reward_config)
        self.cont_head = MLPHead(output_shape=(1,), **cont_config)

        # Store for reference
        self.flatten_keys = flatten_keys
        self.discrete_keys = discrete_keys

    def init_params(
        self,
        rng: jax.random.PRNGKey,
        obs_space: Dict,
        action_space: int,
    ) -> WorldModelParams:
        """Initialize all world model parameters.

        Args:
            rng: Random key for initialization
            obs_space: Observation space (e.g., {"image": {"shape": (64, 64, 3)}})
            action_space: Number of actions

        Returns:
            WorldModelParams with initialized encoder, decoder, dynamics, heads
        """
        rng_enc, rng_dec, rng_tssm, rng_rew, rng_cont = jax.random.split(rng, 5)

        # Create dummy data for initialization
        batch_size = 1

        # Create dummy observation dict for multi-modal encoder
        dummy_obs = {}
        for key, shape in self.obs_shapes.items():
            if len(shape) == 3 and key not in self.flatten_keys:
                # Image observation
                dummy_obs[key] = jnp.zeros((batch_size,) + shape, dtype=jnp.uint8)
            else:
                # Vector observation (or flattened symbolic)
                dummy_obs[key] = jnp.zeros((batch_size,) + shape, dtype=jnp.float32)

        # Dummy action (one-hot) - use COMPUTE_DTYPE for consistency
        dummy_action = jnp.zeros((batch_size, action_space), dtype=COMPUTE_DTYPE)

        # Initialize encoder (BatchNorm needs mutable batch_stats)
        encoder_vars = self.encoder.init(
            {'params': rng_enc},
            dummy_obs,
            training=True,  # Init uses training mode
        )
        encoder_params = encoder_vars  # Contains both 'params' and 'batch_stats'

        # Get embedding dimension (use training=False since we already init'd)
        dummy_embed = self.encoder.apply(
            encoder_vars,
            dummy_obs,
            training=False,  # Use running average
        )
        embed_dim = dummy_embed.shape[-1]

        # Initialize TSSM (needs embed_dim and action_dim) - use COMPUTE_DTYPE
        dummy_state = self.tssm.initial_state(batch_size)
        dummy_embed_input = jnp.zeros((batch_size, embed_dim), dtype=COMPUTE_DTYPE)
        dummy_is_first = jnp.zeros((batch_size,), dtype=jnp.bool_)

        tssm_params = self.tssm.init(
            {'params': rng_tssm},
            dummy_state,
            dummy_action,
            dummy_embed_input,
            dummy_is_first,
            training=False,
            rng=rng_tssm,
            method=self.tssm.observe,
        )

        # Initialize decoder (takes only flattened stoch, also has BatchNorm)
        dummy_stoch_flat = dummy_state.stoch.reshape(batch_size, -1)
        decoder_params = self.decoder.init(
            {'params': rng_dec},
            dummy_stoch_flat,
            training=True,
        )

        # Initialize reward and continuation heads
        # They take features from get_feat() which concatenates hidden + stoch_flat
        feat_dim = self.tssm.hidden_dim + (self.tssm.stoch * self.tssm.classes)
        dummy_feat = jnp.zeros((batch_size, feat_dim), dtype=COMPUTE_DTYPE)

        reward_params = self.reward_head.init(rng_rew, dummy_feat)
        cont_params = self.cont_head.init(rng_cont, dummy_feat)

        return WorldModelParams(
            encoder=encoder_params,
            decoder=decoder_params,
            dynamics=tssm_params,
            reward=reward_params,
            continuation=cont_params,
        )

    def encode(
        self,
        wm_params: WorldModelParams,
        obs: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Encode observations to embeddings.

        Args:
            wm_params: World model parameters (contains 'params' and 'batch_stats')
            obs: Dictionary of observations

        Returns:
            Observation embeddings [B, D]
        """
        # Pass full obs dict to multi-modal encoder
        # Use training=False for inference (use running average for BatchNorm)
        return self.encoder.apply(wm_params.encoder, obs, training=False)

    def initial_state(
        self,
        wm_params: WorldModelParams,
        batch_size: int,
    ) -> WorldModelState:
        """Create initial TSSM state (zeros).

        Args:
            wm_params: World model parameters (unused - TSSM just creates zeros)
            batch_size: Number of parallel states

        Returns:
            Initial TSSMState with zeros
        """
        return self.tssm.initial_state(batch_size)

    def observe(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
        action: jnp.ndarray,
        embed: jnp.ndarray,
        is_first: jnp.ndarray,
        training: bool,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[WorldModelState, Dict[str, Any]]:
        """Update state with observation (posterior inference).

        Args:
            wm_params: World model parameters
            state: Current TSSM state
            action: Previous action [B, A] (one-hot)
            embed: Observation embedding [B, D]
            is_first: Episode start flags [B]
            training: Whether in training mode
            rng: Random key for stochastic sampling

        Returns:
            new_state: Posterior state after observing
            info: Dict with 'prior', 'posterior', 'dyn_loss', 'rep_loss'
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # Split RNG for sampling and dropout
        rng_sample, rng_dropout = jax.random.split(rng)

        # Call TSSM observe
        # Pass dropout RNG through rngs parameter for Flax modules
        new_state, info = self.tssm.apply(
            wm_params.dynamics,
            state,
            action,
            embed,
            is_first,
            training,
            rng_sample,
            method=self.tssm.observe,
            rngs={'dropout': rng_dropout} if training else None,
        )

        return new_state, info

    def observe_sequence(
        self,
        wm_params: WorldModelParams,
        initial_state: WorldModelState,
        actions: jnp.ndarray,
        embeds: jnp.ndarray,
        is_first: jnp.ndarray,
        training: bool,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[WorldModelState, Dict[str, Any]]:
        """Process full observation sequence (for training).

        This method processes sequences in a way that leverages Transformer parallelism
        while maintaining autoregressive correctness for the stochastic states.

        Args:
            wm_params: World model parameters
            initial_state: Initial TSSM state [B, ...]
            actions: Action sequence [B, T, A] (one-hot)
            embeds: Observation embedding sequence [B, T, D]
            is_first: Episode start flags [B, T]
            training: Whether in training mode
            rng: Random key for stochastic sampling

        Returns:
            states: TSSMState with sequence dimension [B, T, ...]
            info: Dictionary with losses for all timesteps
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # Split RNG for sampling and dropout
        rng_sample, rng_dropout = jax.random.split(rng)

        # Call TSSM observe_sequence
        # Pass dropout RNG through rngs parameter for Flax modules
        states, info = self.tssm.apply(
            wm_params.dynamics,
            initial_state,
            actions,
            embeds,
            is_first,
            training,
            rng_sample,
            method=self.tssm.observe_sequence,
            rngs={'dropout': rng_dropout} if training else None,
        )

        return states, info

    def imagine(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
        action: jnp.ndarray,
        training: bool,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[WorldModelState, Dict[str, Any]]:
        """Predict next state without observation (prior only).

        Args:
            wm_params: World model parameters
            state: Current TSSM state
            action: Action to take [B, A]
            training: Whether in training mode
            rng: Random key for stochastic sampling

        Returns:
            new_state: Predicted next state (prior)
            info: Dict with 'prior'
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # Split RNG for sampling and dropout
        rng_sample, rng_dropout = jax.random.split(rng)

        # Call TSSM imagine
        # Pass dropout RNG through rngs parameter for Flax modules
        new_state, info = self.tssm.apply(
            wm_params.dynamics,
            state,
            action,
            training,
            rng_sample,
            method=self.tssm.imagine,
            rngs={'dropout': rng_dropout} if training else None,
        )

        # NOTE: KV-cache is now maintained across imagination steps for context warmup
        # The attention blocks use ring buffer logic to keep cache at fixed max_cache_length
        # This preserves PyTree structure for JAX scan while providing context to the Transformer
        return new_state, info

    def get_feat(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
    ) -> jnp.ndarray:
        """Extract features for policy/value heads.

        For TSSM: concatenate hidden state + flattened stoch.

        Args:
            wm_params: World model parameters (unused)
            state: TSSM state

        Returns:
            Features [B, hidden_dim + stoch*classes]
        """
        return self.tssm.get_feat(state)

    def decode(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
    ) -> Dict[str, Any]:
        """Decode state to observation distributions.

        Args:
            wm_params: World model parameters
            state: TSSM state

        Returns:
            obs_dists: Dictionary of observation distributions
        """
        # STORM decoder takes ONLY the flattened stochastic latent
        batch_size = state.stoch.shape[0]
        stoch_flat = state.stoch.reshape(batch_size, -1)

        # Decoder returns dict of distributions for multi-modal observations
        return self.decoder.apply(wm_params.decoder, stoch_flat, training=False)

    def kl_divergence(
        self,
        post: Dict,
        prior: Dict,
    ) -> jnp.ndarray:
        """Compute KL divergence between posterior and prior.

        Args:
            post: Posterior distribution (OneHotDist)
            prior: Prior distribution (OneHotDist)

        Returns:
            KL divergence [B]
        """
        return post.kl_divergence(prior)

    def predict_reward(
        self,
        wm_params: WorldModelParams,
        feat: jnp.ndarray,
    ) -> Any:
        """Predict reward from features.

        Args:
            wm_params: World model parameters
            feat: Features [B, F]

        Returns:
            Reward distribution
        """
        return self.reward_head.apply(wm_params.reward, feat)

    def predict_continuation(
        self,
        wm_params: WorldModelParams,
        feat: jnp.ndarray,
    ) -> Any:
        """Predict continuation (1 - done) from features.

        Args:
            wm_params: World model parameters
            feat: Features [B, F]

        Returns:
            Continuation distribution
        """
        return self.cont_head.apply(wm_params.continuation, feat)


# Export public API
__all__ = [
    'STORMWorldModel',
]
