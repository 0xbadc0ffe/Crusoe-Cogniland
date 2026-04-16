"""Base protocols and dataclasses for world models in the zoo.

This module defines the core WorldModel protocol that all world model implementations
must follow. The protocol is parameterless (Flax functional style) - all methods take
parameters explicitly rather than storing them internally.

Architecture:
    WorldModel: Abstract protocol defining the interface
    WorldModelParams: Container for all trainable world model parameters
    WorldModelState: Base class for recurrent/latent states (subclassed per model)

Design principles:
    1. Parameterless: Methods take wm_params explicitly (no self.params)
    2. Functional: Pure functions that don't modify state
    3. Composable: Can mix encoders, dynamics, decoders from different sources
    4. JIT-friendly: Clear boundaries for compilation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import jax
import jax.numpy as jnp
import chex


@chex.dataclass
class WorldModelParams:
    """Parameters for world model components.

    This is the SINGLE SOURCE OF TRUTH for world model parameters.
    All world model implementations should use this structure.

    Components:
        encoder: Observation encoder parameters (CNN, VQ, MLP, etc.)
        decoder: Observation decoder parameters
        dynamics: Dynamics model parameters (RSSM, Transformer, etc.)
        reward: Reward predictor head parameters
        continuation: Continuation (1 - done) predictor head parameters

    Note: This dataclass is defined here and should be imported elsewhere
          (e.g., in state.py) to maintain a single source of truth.
    """
    encoder: Any
    decoder: Any
    dynamics: Any  # RSSM, Transformer, etc.
    reward: Any
    continuation: Any


@chex.dataclass
class WorldModelState:
    """Base class for recurrent/latent state of the world model.

    This is an abstract base class. Each world model implementation
    should subclass this to define its specific state structure.

    Examples:
        - RSSMState: stochastic + deterministic components
        - TransformerState: token embeddings + attention cache
        - IRISState: VQ tokens + recurrent state

    The state represents the world model's internal "belief" about
    the current state of the world. It can be updated via:
        - observe(): Incorporate new observations (posterior)
        - imagine(): Predict next state without observations (prior)
    """
    pass  # Subclasses define structure


class WorldModel(ABC):
    """Base protocol for world models in the zoo.

    A world model learns to predict future observations and rewards given
    past observations and actions. It maintains a latent state that summarizes
    history and can be used for planning/imagination.

    Design: WorldModel is a STATELESS SPEC holding Flax module definitions.
    All methods take params explicitly (Flax functional style). This avoids
    JIT complications and makes it clear what data flows where.

    Key methods:
        - encode/decode: Observation ↔ latent embedding
        - observe: Update state with new observation (posterior inference)
        - imagine: Predict next state without observation (prior only)
        - get_feat: Extract features for policy/value
        - predict_reward/continuation: Predict task outcomes

    Usage pattern:
        # 1. Create world model spec (parameterless)
        wm = DreamerV3WorldModel(config)

        # 2. Initialize parameters
        wm_params = wm.init_params(rng, obs_space, action_space)

        # 3. Use functionally
        embed = wm.encode(wm_params, obs)
        state, info = wm.observe(wm_params, state, action, embed, is_first, training, rng)
        next_state, pred = wm.imagine(wm_params, state, action, training, rng)
    """

    @abstractmethod
    def init_params(
        self,
        rng: jax.random.PRNGKey,
        obs_space: Dict,
        action_space: int,
    ) -> WorldModelParams:
        """Initialize world model parameters.

        This method creates all trainable parameters for the world model:
        encoder, decoder, dynamics, reward head, continuation head.

        Args:
            rng: Random key for initialization
            obs_space: Dictionary describing observation space
                      (e.g., {"image": {"shape": (64, 64, 3)}, "vector": {"shape": (10,)}})
            action_space: Number of discrete actions (or action dimension)

        Returns:
            WorldModelParams with initialized encoder, decoder, dynamics, heads

        Note: Use dummy data to initialize Flax modules if needed.
        """
        pass

    @abstractmethod
    def encode(
        self,
        wm_params: WorldModelParams,
        obs: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Encode observations to embeddings.

        This is typically a CNN that processes images and/or an MLP for vectors.
        The embedding is then fed to the dynamics model.

        Args:
            wm_params: World model parameters
            obs: Dictionary of observations
                (e.g., {"image": [B, H, W, C], "vector": [B, D]})

        Returns:
            embeddings: Latent observation embeddings [B, D]

        Note: Encoder should handle multi-modal observations and flatten to a
              single embedding vector for the dynamics model.
        """
        pass

    @abstractmethod
    def initial_state(
        self,
        wm_params: WorldModelParams,
        batch_size: int,
    ) -> WorldModelState:
        """Initialize recurrent/latent state.

        Creates the initial "belief" state for the world model.
        Typically zeros for recurrent states, but could be learned.

        Args:
            wm_params: World model parameters (may contain learned init)
            batch_size: Number of parallel states

        Returns:
            Initial state for the world model (e.g., RSSMState with zeros)

        Note: This is called at the start of each episode.
        """
        pass

    @abstractmethod
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
        """Update state given observation (posterior inference).

        This is the "filtering" step where we incorporate new observations
        to update our belief about the latent state. It combines:
            1. Prior prediction: What we expected based on previous state + action
            2. Observation: What we actually observed
            3. Posterior: Updated belief combining both (via inference network)

        Args:
            wm_params: World model parameters
            state: Current world model state
            action: Previous action [B, A] (one-hot or continuous)
            embed: Observation embedding [B, D] (from encode())
            is_first: Episode start flags [B] (reset state if True)
            training: Whether in training mode (affects dropout, sampling, etc.)
            rng: Optional random key (for stochastic inference)

        Returns:
            new_state: Updated state after observing (posterior)
            info: Dictionary with auxiliary information:
                  - KL terms (for RSSM: kl between posterior and prior)
                  - logits, distributions, etc. (for training loss)

        Note: For RSSM, this computes both prior and posterior and returns KL.
              For deterministic models, posterior = prior (no KL).
        """
        pass

    def observe_sequence(
        self,
        wm_params: WorldModelParams,
        initial_state: WorldModelState,
        actions: jnp.ndarray,  # [T, B, A]
        embeds: jnp.ndarray,   # [T, B, D]
        is_first: jnp.ndarray, # [T, B]
        training: bool,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[WorldModelState, Dict[str, Any]]:
        """Sequence-wise update (optional, for efficient training).

        Default implementation scans `observe` over time.
        Transformer-based models (STORM) can override with parallelized version.

        IMPORTANT: This default implementation splits RNG at each step to prevent
        "RNG not changing" bugs in stochastic models (e.g., RSSM sampling).

        Args:
            wm_params: World model parameters
            initial_state: Initial state
            actions: Action sequence [T, B, A]
            embeds: Embedding sequence [T, B, D]
            is_first: Episode start flags [T, B]
            training: Whether in training mode
            rng: Optional random key

        Returns:
            final_state: Final state after sequence
            info: Dictionary with sequence-level info (stacked over time)
                  All arrays in info will have shape [T, B, ...]

        Note: This is provided for convenience. Recurrent models (RSSM) will use
              the default scan implementation. Transformer models can override
              with a parallelized version for efficiency.
        """
        # Initialize RNG if not provided (for deterministic models)
        if rng is None:
            rng = jax.random.PRNGKey(0)

        def step(carry, inputs):
            state, rng = carry
            rng, step_rng = jax.random.split(rng)  # Split RNG to prevent reuse!
            action, embed, is_first_t = inputs
            state, info = self.observe(wm_params, state, action, embed, is_first_t, training, step_rng)
            return (state, rng), info

        (final_state, _), info_seq = jax.lax.scan(
            step, (initial_state, rng), (actions, embeds, is_first)
        )
        return final_state, info_seq

    @abstractmethod
    def imagine(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
        action: jnp.ndarray,
        training: bool,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[WorldModelState, Dict[str, Any]]:
        """Predict next state without observation (prior only).

        This is the "prediction" step where we imagine future states
        based only on actions, without observing outcomes. Used for:
            1. Planning/imagination during policy training
            2. Generating video predictions
            3. Model-based rollouts

        Args:
            wm_params: World model parameters
            state: Current world model state
            action: Action to take [B, A]
            training: Whether in training mode
            rng: Optional random key (for stochastic transitions)

        Returns:
            new_state: Predicted next state (prior)
            info: Dictionary with predictions:
                  - reward: Predicted reward [B]
                  - continuation: Predicted continuation (1 - done) [B]
                  - distributions, etc. (for visualization)

        Note: This is pure prediction - no observation is incorporated.
              For RSSM, this is the prior (no posterior correction).
        """
        pass

    @abstractmethod
    def get_feat(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
    ) -> jnp.ndarray:
        """Extract features for policy/value heads.

        Converts the world model state into a flat feature vector that
        can be consumed by actor/critic networks. This typically involves
        concatenating different components of the state.

        Args:
            wm_params: World model parameters (unused in most cases)
            state: World model state

        Returns:
            features: Concatenated features [B, F] for downstream heads

        Example (RSSM):
            feat = jnp.concatenate([state.stoch.flatten(), state.deter], -1)

        Note: The feature dimensionality F must be consistent across all states.
        """
        pass

    @abstractmethod
    def decode(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
    ) -> Dict[str, Any]:
        """Decode state to observation distributions.

        Reconstructs observations from the latent state. Returns distributions
        (not point estimates) so we can compute log-likelihood losses.

        Args:
            wm_params: World model parameters
            state: World model state

        Returns:
            obs_dists: Dictionary of observation distributions, one per modality
                      (e.g., {"image": MSEDist(...), "direction": OneHotDist(...)})

        Note: Each distribution should have a log_prob() method for computing
              reconstruction loss. The decoder typically takes state components
              (e.g., stoch + deter for RSSM) and produces distribution parameters.
        """
        pass

    @abstractmethod
    def kl_divergence(
        self,
        post: Dict,
        prior: Dict,
    ) -> jnp.ndarray:
        """Compute KL divergence between posterior and prior.

        Used to regularize the world model during training. For RSSM-style models,
        this ensures the posterior (incorporating observations) doesn't deviate
        too far from the prior (pure prediction).

        Args:
            post: Posterior distribution info (from observe())
            prior: Prior distribution info (from imagine() or observe()'s prior step)

        Returns:
            kl: KL divergence [B] (scalar per batch element)

        Note: For deterministic models, return zeros.
              For RSSM, compute D_KL(post || prior) for stochastic component.
        """
        pass

    @abstractmethod
    def predict_reward(
        self,
        wm_params: WorldModelParams,
        feat: jnp.ndarray,
    ) -> Any:
        """Predict reward from features.

        A learned reward predictor head (typically MLP) that estimates rewards
        from world model features. Used for:
            1. Training the world model (reward prediction loss)
            2. Planning/imagination (predicting returns)

        Args:
            wm_params: World model parameters
            feat: Features [B, F] (from get_feat())

        Returns:
            Reward distribution (e.g., TwoHotDist, MSEDist)

        Note: Returns a distribution, not a point estimate, so we can compute
              log-likelihood loss during training.
        """
        pass

    @abstractmethod
    def predict_continuation(
        self,
        wm_params: WorldModelParams,
        feat: jnp.ndarray,
    ) -> Any:
        """Predict continuation (1 - done) from features.

        A learned continuation predictor head that estimates whether the episode
        continues or terminates. Used for:
            1. Training the world model (continuation prediction loss)
            2. Planning/imagination (bootstrap value correctly)

        Args:
            wm_params: World model parameters
            feat: Features [B, F] (from get_feat())

        Returns:
            Continuation distribution (e.g., BinaryDist)

        Note: Continuation = 1 - done (i.e., 1 if episode continues, 0 if done).
              This is more numerically stable than predicting "done" directly.
        """
        pass


# Export public API
__all__ = [
    'WorldModel',
    'WorldModelParams',
    'WorldModelState',
]
