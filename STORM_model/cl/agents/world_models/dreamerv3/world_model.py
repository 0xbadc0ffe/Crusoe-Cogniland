"""DreamerV3 world model implementation using the WorldModel protocol.

This module wraps DreamerV3's components (Encoder, Decoder, RSSM, MLP heads) to
implement the WorldModel protocol defined in base.py.
"""

from typing import Any, Dict, Optional, Tuple
import jax
import jax.numpy as jnp

from cl.agents.world_models.base import WorldModel, WorldModelParams, WorldModelState
from cl.agents.world_models.dreamerv3.encoder import Encoder
from cl.agents.world_models.dreamerv3.decoder import Decoder
from cl.agents.world_models.dreamerv3.rssm import RSSM
from cl.agents.commons.networks.mlp import MLPHead


class DreamerV3WorldModel(WorldModel):
    """DreamerV3 world model implementation.

    Combines:
        - Multi-modal Encoder (MLP + CNN)
        - Multi-modal Decoder (MLP + transposed CNN)
        - RSSM dynamics (BlockGRU + categorical latents)
        - Reward and continuation predictors (MLP heads)

    This class is a STATELESS WRAPPER that implements the WorldModel protocol.
    All methods take parameters explicitly (Flax functional style).

    Args:
        obs_space: Observation space specification
        action_space: Number of discrete actions
        encoder_config: Configuration for encoder (units, depth, etc.)
        decoder_config: Configuration for decoder
        rssm_config: Configuration for RSSM (deter_dim, stoch_dim, etc.)
        reward_config: Configuration for reward head
        cont_config: Configuration for continuation head
    """

    def __init__(
        self,
        obs_space: Dict,
        action_space: int,
        encoder_config: Dict,
        decoder_config: Dict,
        rssm_config: Dict,
        reward_config: Dict,
        cont_config: Dict,
    ):
        # Store config
        self.obs_space = obs_space
        self.action_space = action_space

        # Create Flax modules (specs only, no params)
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(obs_shapes=obs_space, **decoder_config)
        self.rssm = RSSM(**rssm_config)
        self.reward_head = MLPHead(output_shape=(1,), **reward_config)
        self.cont_head = MLPHead(output_shape=(1,), **cont_config)

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
        rng_enc, rng_dec, rng_rssm, rng_rew, rng_cont = jax.random.split(rng, 5)

        # Create dummy data for initialization
        batch_size = 1

        # Dummy observation
        dummy_obs = {}
        for key, spec in obs_space.items():
            # Handle both formats: spec can be a tuple (64, 64, 3) or dict {"shape": (64, 64, 3)}
            if isinstance(spec, dict):
                shape = (batch_size,) + tuple(spec['shape'])
            else:
                shape = (batch_size,) + tuple(spec)

            if 'image' in key.lower():
                dummy_obs[key] = jnp.zeros(shape, dtype=jnp.uint8)
            else:
                dummy_obs[key] = jnp.zeros(shape, dtype=jnp.float32)

        # Dummy action (one-hot)
        dummy_action = jnp.zeros((batch_size, action_space), dtype=jnp.float32)

        # Initialize encoder
        encoder_params = self.encoder.init(rng_enc, dummy_obs)

        # Get embedding dimension
        dummy_embed = self.encoder.apply(encoder_params, dummy_obs)
        embed_dim = dummy_embed.shape[-1]

        # Initialize RSSM (needs embed_dim and action_dim)
        dummy_state = self.rssm.initial_state(batch_size)
        dummy_embed_input = jnp.zeros((batch_size, embed_dim), dtype=jnp.float32)
        dummy_is_first = jnp.zeros((batch_size,), dtype=jnp.bool_)

        rssm_params = self.rssm.init(
            rng_rssm,
            dummy_state,
            dummy_action,
            dummy_embed_input,
            dummy_is_first,
            training=False,
        )

        # Initialize decoder (takes state dict)
        dummy_state_dict = {'deter': dummy_state.deter, 'stoch': dummy_state.stoch}
        decoder_params = self.decoder.init(rng_dec, dummy_state_dict)

        # Initialize reward and continuation heads
        # They take features (concatenation of stoch and deter)
        feat_dim = dummy_state.stoch.size + dummy_state.deter.size
        dummy_feat = jnp.zeros((batch_size, feat_dim), dtype=jnp.float32)

        reward_params = self.reward_head.init(rng_rew, dummy_feat)
        cont_params = self.cont_head.init(rng_cont, dummy_feat)

        return WorldModelParams(
            encoder=encoder_params,
            decoder=decoder_params,
            dynamics=rssm_params,
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
            wm_params: World model parameters
            obs: Dictionary of observations

        Returns:
            Observation embeddings [B, D]
        """
        return self.encoder.apply(wm_params.encoder, obs)

    def initial_state(
        self,
        wm_params: WorldModelParams,
        batch_size: int,
    ) -> WorldModelState:
        """Create initial RSSM state (zeros).

        Args:
            wm_params: World model parameters (unused - RSSM just creates zeros)
            batch_size: Number of parallel states

        Returns:
            Initial RSSMState with zeros

        Note: wm_params is unused but required by WorldModel protocol for consistency.
              Other world models might have learned initial states.
        """
        return self.rssm.initial_state(batch_size)

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

        Computes both prior (prediction) and posterior (incorporating observation),
        then returns posterior state and KL divergence for training.

        Args:
            wm_params: World model parameters
            state: Current RSSM state
            action: Previous action [B, A] (one-hot)
            embed: Observation embedding [B, D]
            is_first: Episode start flags [B]
            training: Whether in training mode
            rng: Random key (unused for DreamerV3)

        Returns:
            new_state: Posterior state after observing
            info: Dictionary with 'prior' and 'post' for KL computation
        """
        # RSSM's __call__ does both prior and posterior
        new_state, info = self.rssm.apply(
            wm_params.dynamics,
            state,
            action,
            embed,
            is_first,
            training,
            rng,  # Pass RNG for stochastic sampling
        )
        return new_state, info

    def imagine(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
        action: jnp.ndarray,
        training: bool,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[WorldModelState, Dict[str, Any]]:
        """Predict next state without observation (prior only).

        DreamerV3 imagination: calls RSSM with embed=None to skip posterior.
        The stochastic component is sampled from the prior, requiring RNG.

        Args:
            wm_params: World model parameters
            state: Current RSSM state
            action: Action to take [B, A] (one-hot)
            training: Whether in training mode (typically False during imagination)
            rng: Random key for sampling stochastic component

        Returns:
            new_state: Predicted next state (prior)
            info: Dictionary with predictions
        """
        batch_size = state.deter.shape[0]

        # RSSM uses embed=None to distinguish imagination from observation
        # During imagination, only the prior is computed (no posterior)
        new_state, info = self.rssm.apply(
            wm_params.dynamics,
            state,
            action,
            embed=None,  # No observation embedding during imagination
            is_first=jnp.zeros(batch_size, dtype=bool),
            training=training,
            rng=rng,  # RNG needed for sampling stochastic component
        )
        return new_state, info

    def get_feat(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
    ) -> jnp.ndarray:
        """Extract features for policy/value heads.

        Concatenates stochastic and deterministic components.

        Args:
            wm_params: World model parameters (unused)
            state: RSSM state

        Returns:
            Features [B, F] where F = stoch_size + deter_dim
        """
        return self.rssm.apply(
            wm_params.dynamics,
            state,
            method=self.rssm.get_feat,
        )

    def decode(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
    ) -> Dict[str, Any]:
        """Decode state to observation distributions.

        Args:
            wm_params: World model parameters
            state: RSSM state

        Returns:
            Dictionary of observation distributions
        """
        # DreamerV3's decoder takes raw state components, not features
        state_dict = {'deter': state.deter, 'stoch': state.stoch}
        return self.decoder.apply(wm_params.decoder, state_dict)

    def kl_divergence(
        self,
        post: Dict,
        prior: Dict,
    ) -> jnp.ndarray:
        """Compute KL divergence between posterior and prior.

        Args:
            post: Posterior distribution info from observe()
            prior: Prior distribution info from observe()

        Returns:
            KL divergence [B]
        """
        return self.rssm.kl_divergence(post, prior)

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
