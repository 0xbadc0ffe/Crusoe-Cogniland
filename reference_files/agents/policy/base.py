"""Policy protocol for actor-critic agents.

This module defines the Policy protocol that all policy implementations must follow.
Similar to the WorldModel protocol, this provides a clean abstraction over different
policy architectures (MLP, ConvLSTM, etc.).

Design principles:
    1. Parameterless: Policy instances don't store parameters (Flax functional style)
    2. Architecture agnostic: Works with MLP, LSTM, Transformer policies
    3. Composable: Can be used with any WorldModel implementation
    4. JIT-friendly: All methods are pure functions

Usage pattern:
    # 1. Create policy instance (architecture only)
    policy = MLPPolicy(actor_config, critic_config)

    # 2. Initialize parameters
    policy_params = policy.init_params(rng, feat_dim, action_space)
    # Returns PolicyParams(actor, critic, slow_critic, normalizers)

    # 3. Apply networks
    actor_dist = policy.apply_actor(policy_params.actor, features, training=True, rng=rng)
    critic_dist = policy.apply_critic(policy_params.critic, features, training=False)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import jax
import jax.numpy as jnp

from cl.agents.state import PolicyParams


class Policy(ABC):
    """Abstract base class for policy implementations.

    A Policy defines the architecture for both actor (policy) and critic (value)
    networks. It handles parameter initialization and provides apply methods.

    Implementations:
        - MLPPolicy: Simple feedforward MLP for DreamerV3, STORM
        - ConvLSTMPolicy: Recurrent policy for IRIS (future)
        - TransformerPolicy: Transformer-based policy (future)
    """

    @abstractmethod
    def init_params(
        self,
        rng: jax.random.PRNGKey,
        feat_dim: int,
        action_space: int,
    ) -> PolicyParams:
        """Initialize all policy parameters.

        This includes:
            - actor: Actor network parameters
            - critic: Critic network parameters
            - slow_critic: Copy of critic parameters (EMA target)
            - normalizers: Dict with 'return', 'value', 'advantage' normalizers

        Args:
            rng: Random key for initialization
            feat_dim: Feature dimension from world model
            action_space: Number of discrete actions

        Returns:
            PolicyParams with all initialized parameters
        """
        pass

    @abstractmethod
    def apply_actor(
        self,
        actor_params: Any,
        features: jnp.ndarray,
        training: bool = False,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Any:
        """Apply actor network to get action distribution.

        Args:
            actor_params: Actor parameters (from PolicyParams.actor)
            features: Features from world model [B, F]
            training: Whether in training mode (for dropout, etc.)
            rng: Random key (for stochastic layers, if any)

        Returns:
            Action distribution (e.g., OneHot, Categorical)
        """
        pass

    @abstractmethod
    def apply_critic(
        self,
        critic_params: Any,
        features: jnp.ndarray,
        training: bool = False,
    ) -> Any:
        """Apply critic network to get value distribution.

        Args:
            critic_params: Critic parameters (from PolicyParams.critic or .slow_critic)
            features: Features from world model [B, F]
            training: Whether in training mode

        Returns:
            Value distribution (e.g., TwoHot, MSE)
        """
        pass


# Export public API
__all__ = [
    'Policy',
]
