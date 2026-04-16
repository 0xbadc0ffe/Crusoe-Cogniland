"""MLP policy implementation for DreamerV3 and similar agents.

This module provides a simple feedforward MLP policy using MLPHead for both
actor and critic networks. This is the standard architecture for DreamerV3, STORM,
and other MLP-based world model agents.
"""

from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp

from cogniland.agents.policy.base import Policy
from cogniland.agents.state import PolicyParams
from cogniland.agents.commons.networks.mlp import MLPHead


class MLPPolicy(Policy):
    """Simple MLP policy for actor-critic agents.

    Uses MLPHead for both actor and critic networks with configurable
    architectures. This is the standard policy for DreamerV3.

    Note: Normalizers are NOT initialized here - the caller (agent) is
          responsible for creating and setting normalizers with appropriate
          configs (e.g., separate configs for return/value/advantage).

    Args:
        action_space: Number of discrete actions
        actor_config: Configuration dict for actor MLPHead
                     (e.g., {'layers': 3, 'units': 1024, 'dist': 'onehot'})
        critic_config: Configuration dict for critic MLPHead
                      (e.g., {'layers': 3, 'units': 1024, 'dist': 'symexp_twohot'})
    """

    def __init__(
        self,
        action_space: int,
        actor_config: Dict,
        critic_config: Dict,
    ):
        self.action_space = action_space
        self.actor_config = actor_config
        self.critic_config = critic_config

        # Create Flax modules (specs only, no params)
        self.actor = MLPHead(
            output_shape=(action_space,),
            **actor_config
        )
        self.critic = MLPHead(
            output_shape=(1,),  # Scalar value
            **critic_config
        )

    def init_params(
        self,
        rng: jax.random.PRNGKey,
        feat_dim: int,
        action_space: int,
    ) -> PolicyParams:
        """Initialize all policy parameters.

        Args:
            rng: Random key for initialization
            feat_dim: Feature dimension from world model
            action_space: Number of discrete actions (must match self.action_space)

        Returns:
            PolicyParams with actor, critic, slow_critic, and normalizers
        """
        assert action_space == self.action_space, \
            f"Action space mismatch: {action_space} != {self.action_space}"

        rng_actor, rng_critic = jax.random.split(rng)

        # Dummy input for initialization
        dummy_feat = jnp.zeros((1, feat_dim), dtype=jnp.float32)

        # Initialize actor
        actor_params = self.actor.init(rng_actor, dummy_feat, training=False)

        # Initialize critic
        critic_params = self.critic.init(rng_critic, dummy_feat, training=False)

        # Copy critic params for slow critic (EMA target)
        slow_critic_params = jax.tree.map(lambda x: x.copy(), critic_params)

        # Normalizers are NOT initialized here - caller is responsible
        # This allows agents to provide their own normalizer configs
        # (e.g., DreamerV3 uses separate configs for return/value/advantage)
        return PolicyParams(
            actor=actor_params,
            critic=critic_params,
            slow_critic=slow_critic_params,
            normalizers={},  # Empty - caller will set these
        )

    def apply_actor(
        self,
        actor_params: Any,
        features: jnp.ndarray,
        training: bool = False,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Any:
        """Apply actor network to get action distribution.

        Args:
            actor_params: Actor parameters
            features: Features from world model [B, F]
            training: Whether in training mode (affects unimix)
            rng: Random key (unused for MLPHead)

        Returns:
            Action distribution (OneHot or Categorical)
        """
        return self.actor.apply(actor_params, features, training=training)

    def apply_critic(
        self,
        critic_params: Any,
        features: jnp.ndarray,
        training: bool = False,
    ) -> Any:
        """Apply critic network to get value distribution.

        Args:
            critic_params: Critic parameters (can be critic or slow_critic)
            features: Features from world model [B, F]
            training: Whether in training mode

        Returns:
            Value distribution (TwoHot, MSE, etc.)
        """
        return self.critic.apply(critic_params, features, training=training)


# Export public API
__all__ = [
    'MLPPolicy',
]
