"""STORM world model implementation using the WorldModel protocol.

This module implements STORM (Stochastic Transformer-based World Model) by:
- Reusing DreamerV3's encoder and decoder
- Replacing RSSM with a causal transformer (StochasticTransformer)
- Using discrete categorical latents (like DreamerV3)
"""

from typing import Any, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from cl.agents.world_models.base import WorldModel, WorldModelParams, WorldModelState
from cl.agents.world_models.dreamerv3.encoder import Encoder
from cl.agents.world_models.dreamerv3.decoder import Decoder
from cl.agents.world_models.storm.transformer import StochasticTransformer, StormState, create_causal_mask
from cl.agents.commons.networks.mlp import MLPHead
from cl.agents.commons.initializer import Initializer
from cl.agents.commons.distributions import OneHotDist
from cl.config.jax_config import COMPUTE_DTYPE


class DistHead(nn.Module):
    """Distribution head for prior and posterior networks.
    
    This is similar to the DistHead in the PyTorch STORM implementation.
    It produces logits for categorical distributions over stochastic latents.
    
    Args:
        stoch_dim: Number of categorical distributions
        classes: Number of classes per categorical
        unimix: Uniform mixture ratio for regularization
    """
    stoch_dim: int
    classes: int
    unimix: float = 0.01
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass to produce distribution logits.
        
        Args:
            x: Input features [B, D]
            
        Returns:
            logits: Distribution logits [B, stoch_dim * classes]
        """
        init = Initializer(dist='trunc_normal', fan='in', scale=1.0)
        
        # Project to logits
        logits = nn.Dense(
            self.stoch_dim * self.classes,
            kernel_init=init,
            dtype=COMPUTE_DTYPE,
            name='logits'
        )(x)
        
        return logits


class StormWorldModel(WorldModel):
    """STORM world model implementation.
    
    Combines:
        - Multi-modal Encoder (MLP + CNN, reused from DreamerV3)
        - Multi-modal Decoder (MLP + transposed CNN, reused from DreamerV3)
        - Stochastic Transformer dynamics (causal transformer with discrete latents)
        - Reward and continuation predictors (MLP heads)
    
    This class is a STATELESS WRAPPER that implements the WorldModel protocol.
    All methods take parameters explicitly (Flax functional style).
    
    Args:
        obs_space: Observation space specification
        action_space: Number of discrete actions
        encoder_config: Configuration for encoder
        decoder_config: Configuration for decoder
        transformer_config: Configuration for transformer (feat_dim, num_layers, num_heads, etc.)
        reward_config: Configuration for reward head
        cont_config: Configuration for continuation head
        stoch_dim: Number of categorical distributions (default: 32)
        classes: Number of classes per categorical (default: 32)
        unimix: Uniform mixture ratio for regularization (default: 0.01)
        free_nats: Free nats for KL regularization (default: 1.0)
    """
    
    def __init__(
        self,
        obs_space: Dict,
        action_space: int,
        encoder_config: Dict,
        decoder_config: Dict,
        transformer_config: Dict,
        reward_config: Dict,
        cont_config: Dict,
        stoch_dim: int = 32,
        classes: int = 32,
        unimix: float = 0.01,
        free_nats: float = 1.0,
    ):
        # Store config
        self.obs_space = obs_space
        self.action_space = action_space
        self.stoch_dim = stoch_dim
        self.classes = classes
        self.unimix = unimix
        self.free_nats = free_nats
        self.stoch_flattened_dim = stoch_dim * classes
        
        # Create Flax modules (specs only, no params)
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(obs_shapes=obs_space, **decoder_config)
        
        # Transformer dynamics
        self.transformer = StochasticTransformer(
            stoch_dim=self.stoch_flattened_dim,
            action_dim=action_space,
            **transformer_config
        )
        
        # Distribution heads
        self.post_head = DistHead(stoch_dim=stoch_dim, classes=classes, unimix=unimix)
        self.prior_head = DistHead(stoch_dim=stoch_dim, classes=classes, unimix=unimix)
        
        # Prediction heads
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
            obs_space: Observation space
            action_space: Number of actions
            
        Returns:
            WorldModelParams with initialized encoder, decoder, dynamics, heads
        """
        rng_enc, rng_dec, rng_trans, rng_post, rng_prior, rng_rew, rng_cont = jax.random.split(rng, 7)
        
        # Create dummy data for initialization
        batch_size = 1
        
        # Dummy observation
        dummy_obs = {}
        for key, spec in obs_space.items():
            if isinstance(spec, dict):
                shape = (batch_size,) + tuple(spec['shape'])
            else:
                shape = (batch_size,) + tuple(spec)
            
            if 'image' in key.lower():
                dummy_obs[key] = jnp.zeros(shape, dtype=jnp.uint8)
            else:
                dummy_obs[key] = jnp.zeros(shape, dtype=jnp.float32)
        
        # Initialize encoder
        encoder_params = self.encoder.init(rng_enc, dummy_obs)
        
        # Get embedding dimension
        dummy_embed = self.encoder.apply(encoder_params, dummy_obs)
        embed_dim = dummy_embed.shape[-1]
        
        # Initialize post and prior heads
        dummy_embed_input = jnp.zeros((batch_size, embed_dim), dtype=COMPUTE_DTYPE)
        dummy_feat_input = jnp.zeros((batch_size, self.transformer.feat_dim), dtype=COMPUTE_DTYPE)
        
        post_head_params = self.post_head.init(rng_post, dummy_embed_input)
        prior_head_params = self.prior_head.init(rng_prior, dummy_feat_input)
        
        # Initialize transformer
        dummy_samples = jnp.zeros((batch_size, 1, self.stoch_flattened_dim), dtype=COMPUTE_DTYPE)
        dummy_actions = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        dummy_mask = create_causal_mask(1)
        
        trans_params = self.transformer.init(
            rng_trans,
            dummy_samples,
            dummy_actions,
            mask=dummy_mask,
            training=False,
        )
        
        # Combine post_head and prior_head params with transformer params
        # The dynamics params includes transformer + distribution heads
        dynamics_params = {
            'transformer': trans_params,
            'post_head': post_head_params,
            'prior_head': prior_head_params,
        }
        
        # Initialize decoder
        dummy_state_dict = {
            'deter': jnp.zeros((batch_size, self.stoch_flattened_dim), dtype=COMPUTE_DTYPE),
            'stoch': jnp.zeros((batch_size, self.stoch_dim, self.classes), dtype=COMPUTE_DTYPE),
        }
        decoder_params = self.decoder.init(rng_dec, dummy_state_dict)
        
        # Initialize reward and continuation heads
        # They take features (just flattened stoch for STORM)
        # Note: Unlike DreamerV3 which has deter+stoch, STORM only uses stoch
        feat_dim = self.stoch_flattened_dim
        dummy_feat = jnp.zeros((batch_size, feat_dim), dtype=COMPUTE_DTYPE)
        
        reward_params = self.reward_head.init(rng_rew, dummy_feat)
        cont_params = self.cont_head.init(rng_cont, dummy_feat)
        
        return WorldModelParams(
            encoder=encoder_params,
            decoder=decoder_params,
            dynamics=dynamics_params,
            reward=reward_params,
            continuation=cont_params,
        )
    
    def encode(
        self,
        wm_params: WorldModelParams,
        obs: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Encode observations to embeddings."""
        return self.encoder.apply(wm_params.encoder, obs)
    
    def initial_state(
        self,
        wm_params: WorldModelParams,
        batch_size: int,
    ) -> WorldModelState:
        """Create initial STORM state (zeros)."""
        return StormState(
            stoch=jnp.zeros((batch_size, self.stoch_dim, self.classes), dtype=COMPUTE_DTYPE),
            logits=jnp.zeros((batch_size, self.stoch_flattened_dim), dtype=COMPUTE_DTYPE),
            kv_cache=None,  # No cache initially
        )
    
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
        
        For STORM, this:
        1. Computes posterior from observation embedding
        2. Samples stochastic latent from posterior
        3. Computes prior from previous state + action via transformer
        4. Returns posterior state and info dict with prior/post for KL
        """
        # Reset state if first timestep
        if is_first is not None:
            is_first_exp = is_first[..., None]
            state = StormState(
                stoch=state.stoch * (1 - is_first_exp[..., None]),
                logits=state.logits * (1 - is_first_exp),
                kv_cache=None,  # Reset cache on episode boundary
            )
        
        # Flatten previous stochastic state
        prev_stoch_flat = state.stoch.reshape(state.stoch.shape[0], -1)
        
        # Compute prior using transformer
        # Input: previous stochastic latent + action
        # Convert one-hot action to discrete indices if needed
        if action.shape[-1] > 1:  # One-hot encoded [B, A]
            action_indices = jnp.argmax(action, axis=-1)  # [B]
        else:  # Already discrete [B] or [B, 1]
            action_indices = action.squeeze(-1) if action.ndim > 1 else action
        
        # For single step, we need to expand dims for transformer
        samples_input = prev_stoch_flat[:, None, :]  # [B, 1, stoch_flat_dim]
        actions_input = action_indices[:, None] if action_indices.ndim == 1 else action_indices  # [B, 1]
        
        # Create causal mask (single token attends to itself)
        mask = create_causal_mask(1)
        
        # Get transformer features
        if training and rng is not None:
            rng_trans = jax.random.split(rng)[0]
            trans_feats = self.transformer.apply(
                wm_params.dynamics['transformer'],
                samples_input,
                actions_input,
                mask=mask,
                training=training,
                rngs={'dropout': rng_trans},
            )  # [B, 1, feat_dim]
        else:
            trans_feats = self.transformer.apply(
                wm_params.dynamics['transformer'],
                samples_input,
                actions_input,
                mask=mask,
                training=training,
            )  # [B, 1, feat_dim]
        
        trans_feats = trans_feats[:, 0, :]  # Remove sequence dimension [B, feat_dim]
        
        # Compute prior logits from transformer features
        prior_logits = self.prior_head.apply(
            wm_params.dynamics['prior_head'],
            trans_feats,
        )  # [B, stoch_flat_dim]
        
        # Compute posterior logits from observation embedding
        post_logits = self.post_head.apply(
            wm_params.dynamics['post_head'],
            embed,
        )  # [B, stoch_flat_dim]
        
        # Sample from posterior
        post_logits_reshaped = post_logits.reshape(-1, self.stoch_dim, self.classes)
        post_dist = OneHotDist(post_logits_reshaped, unimix=self.unimix)
        
        if training and rng is not None:
            post_stoch = post_dist.sample(seed=rng)
        else:
            post_stoch = post_dist.mode()
        
        # Create posterior state
        post_state = StormState(
            stoch=post_stoch,
            logits=post_logits,
            kv_cache=None,  # Cache not used during observe
        )
        
        # Create prior state for KL computation
        prior_logits_reshaped = prior_logits.reshape(-1, self.stoch_dim, self.classes)
        prior_dist = OneHotDist(prior_logits_reshaped, unimix=self.unimix)
        
        if training and rng is not None:
            rng_prior = jax.random.split(rng)[0]
            prior_stoch = prior_dist.sample(seed=rng_prior)
        else:
            prior_stoch = prior_dist.mode()
        
        prior_state = StormState(
            stoch=prior_stoch,
            logits=prior_logits,
            kv_cache=None,
        )
        
        return post_state, {'prior': prior_state, 'post': post_state}
    
    def imagine(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
        action: jnp.ndarray,
        training: bool,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[WorldModelState, Dict[str, Any]]:
        """Predict next state without observation (prior only).
        
        For STORM imagination, we use the transformer with KV cache for efficiency.
        """
        # Flatten stochastic state
        prev_stoch_flat = state.stoch.reshape(state.stoch.shape[0], -1)
        
        # Convert one-hot action to discrete indices if needed
        if action.shape[-1] > 1:  # One-hot encoded [B, A]
            action_indices = jnp.argmax(action, axis=-1)  # [B]
        else:  # Already discrete [B] or [B, 1]
            action_indices = action.squeeze(-1) if action.ndim > 1 else action
        
        # Prepare input for transformer
        samples_input = prev_stoch_flat[:, None, :]  # [B, 1, stoch_flat_dim]
        actions_input = action_indices[:, None] if action_indices.ndim == 1 else action_indices  # [B, 1]
        
        # Create mask
        mask = create_causal_mask(1)
        
        # Get transformer features
        if training and rng is not None:
            rng_trans = jax.random.split(rng)[0]
            trans_feats = self.transformer.apply(
                wm_params.dynamics['transformer'],
                samples_input,
                actions_input,
                mask=mask,
                training=training,
                rngs={'dropout': rng_trans},
            )  # [B, 1, feat_dim]
        else:
            trans_feats = self.transformer.apply(
                wm_params.dynamics['transformer'],
                samples_input,
                actions_input,
                mask=mask,
                training=training,
            )  # [B, 1, feat_dim]
        
        trans_feats = trans_feats[:, 0, :]  # [B, feat_dim]
        
        # Compute prior logits
        prior_logits = self.prior_head.apply(
            wm_params.dynamics['prior_head'],
            trans_feats,
        )
        
        # Sample from prior
        prior_logits_reshaped = prior_logits.reshape(-1, self.stoch_dim, self.classes)
        prior_dist = OneHotDist(prior_logits_reshaped, unimix=self.unimix)
        
        if training and rng is not None:
            prior_stoch = prior_dist.sample(seed=rng)
        else:
            prior_stoch = prior_dist.mode()
        
        # Create new state
        new_state = StormState(
            stoch=prior_stoch,
            logits=prior_logits,
            kv_cache=state.kv_cache,  # Preserve KV cache (if any)
        )
        
        return new_state, {}
    
    def get_feat(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
    ) -> jnp.ndarray:
        """Extract features for policy/value heads.
        
        For STORM, features are the flattened stochastic latents.
        Unlike DreamerV3's RSSM which has deter + stoch, STORM only has stoch.
        
        Handles both single states and batches of timesteps:
        - state.stoch shape [B, stoch_dim, classes] -> returns [B, stoch_dim * classes]
        - state.stoch shape [B, T, stoch_dim, classes] -> returns [B, T, stoch_dim * classes]
        """
        # Handle both (batch, ...) and (batch, seq_len, ...) shapes
        stoch_shape = state.stoch.shape
        if len(stoch_shape) == 3:  # (batch, stoch_dim, classes)
            stoch_flat = state.stoch.reshape(stoch_shape[0], -1)
        elif len(stoch_shape) == 4:  # (batch, seq_len, stoch_dim, classes)
            stoch_flat = state.stoch.reshape(stoch_shape[0], stoch_shape[1], -1)
        else:
            raise ValueError(f"Unexpected stoch shape: {stoch_shape}")
        
        return stoch_flat
    
    def decode(
        self,
        wm_params: WorldModelParams,
        state: WorldModelState,
    ) -> Dict[str, Any]:
        """Decode state to observation distributions.
        
        STORM reuses DreamerV3's decoder, which expects 'deter' and 'stoch'.
        For STORM, we only have stochastic latents, so we use them for both.
        """
        # For STORM, we use the stochastic latents as both components
        stoch_flat = state.stoch.reshape(state.stoch.shape[0], -1)
        
        state_dict = {
            'deter': stoch_flat,  # Use stochastic as "deter" component
            'stoch': state.stoch,  # Keep stochastic component
        }
        return self.decoder.apply(wm_params.decoder, state_dict)
    
    def kl_divergence(
        self,
        wm_params: WorldModelParams,
        post: WorldModelState,
        prior: WorldModelState,
    ) -> jnp.ndarray:
        """Compute KL divergence between posterior and prior.
        
        Args:
            wm_params: World model parameters (not used but matches interface)
            post: Posterior StormState
            prior: Prior StormState
            
        Returns:
            KL divergence dict with 'dyn' and 'rep' losses
        """
        # States are directly passed, not wrapped in dicts
        post_state = post
        prior_state = prior
        
        # Reshape logits to distributions
        post_logits = post_state.logits.reshape(-1, self.stoch_dim, self.classes)
        prior_logits = prior_state.logits.reshape(-1, self.stoch_dim, self.classes)
        
        # Create distributions
        post_dist = OneHotDist(post_logits, unimix=self.unimix)
        prior_dist = OneHotDist(prior_logits, unimix=self.unimix)
        
        # Dynamics loss: D(sg(post) || prior) - gradients to prior (transformer)
        post_sg_logits = jax.lax.stop_gradient(post_logits)
        post_sg_dist = OneHotDist(post_sg_logits, unimix=self.unimix)
        dyn = post_sg_dist.kl_divergence(prior_dist)
        
        # Representation loss: D(post || sg(prior)) - gradients to post (encoder)
        prior_sg_logits = jax.lax.stop_gradient(prior_logits)
        prior_sg_dist = OneHotDist(prior_sg_logits, unimix=self.unimix)
        rep = post_dist.kl_divergence(prior_sg_dist)
        
        # Apply free nats
        if self.free_nats > 0:
            dyn = jnp.maximum(dyn, self.free_nats)
            rep = jnp.maximum(rep, self.free_nats)
        
        return {'dyn': dyn, 'rep': rep}
    
    def predict_reward(
        self,
        wm_params: WorldModelParams,
        feat: jnp.ndarray,
    ) -> Any:
        """Predict reward from features."""
        return self.reward_head.apply(wm_params.reward, feat)
    
    def predict_continuation(
        self,
        wm_params: WorldModelParams,
        feat: jnp.ndarray,
    ) -> Any:
        """Predict continuation (1 - done) from features."""
        return self.cont_head.apply(wm_params.continuation, feat)
