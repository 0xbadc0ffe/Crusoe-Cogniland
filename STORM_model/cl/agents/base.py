"""Base class for continual RL agents in JAX"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from omegaconf import OmegaConf


@struct.dataclass
class AgentState:
    """Base agent state for JAX agents (can be extended by subclasses)"""
    train_steps: int = 0
    # Subclasses can add: params, opt_state, rng, etc.


class ContinualAgent(ABC):
    """
    Base class for continual RL agents using JAX.

    Designed to work with both synchronous (PPO) and asynchronous (IMPALA) training.
    Agents handle their own rollout collection following CleanBA/PureJaxRL patterns.

    For continual learning across environments with varying action spaces:
    - Agents are initialized with max_actions (max across all environments)
    - Agents can use action masking when current environment has fewer actions
    - Subclasses can override set_environment() to handle environment-specific setup
    """

    def __init__(self, config: OmegaConf, obs_space: tuple, act_space: int):
        """
        Initialize agent.

        Args:
            config: OmegaConf configuration
            obs_space: Observation space shape (can be max across envs for flexibility)
            act_space: Number of discrete actions (typically max_actions for CRL)
        """
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space  # Max actions across all environments
        self.current_num_actions = act_space  # Can be updated via set_environment()

    @abstractmethod
    def init(self, rng: jax.random.PRNGKey) -> AgentState:
        """
        Initialize agent state (params, optimizer, etc.).

        Args:
            rng: JAX random key

        Returns:
            Initial agent state
        """
        pass

    @abstractmethod
    def select_action(
        self,
        state: AgentState,
        obs: jnp.ndarray,
        rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, AgentState]:
        """
        Select action(s) for given observation(s).

        Args:
            state: Current agent state
            obs: Observation(s) - shape depends on environment
            rng: JAX random key

        Returns:
            actions: Selected action(s)
            new_state: Updated agent state
        """
        pass

    @abstractmethod
    def train(
        self,
        state: AgentState,
        env: Any,
        rng: jax.random.PRNGKey,
        num_train_frames: int,
    ) -> Tuple[AgentState, Dict[str, Any]]:
        """
        Train agent on environment for specified number of frames.

        This is the main training loop - agent owns rollout collection and updates.
        Subclasses implement either synchronous (PPO) or asynchronous (IMPALA) style.

        Args:
            state: Current agent state
            env: Environment instance
            rng: JAX random key
            num_train_frames: Number of frames to train for

        Returns:
            new_state: Updated agent state after training
            metrics: Training metrics dict (aggregated over entire training run)
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        state: AgentState,
        env: Any,
        rng: jax.random.PRNGKey,
        num_eval_frames: int,
    ) -> Dict[str, Any]:
        """
        Evaluate agent on environment for specified number of frames.

        Args:
            state: Current agent state
            env: Environment instance
            rng: JAX random key
            num_eval_frames: Number of frames to evaluate for

        Returns:
            metrics: Evaluation metrics dict
        """
        pass

    def on_environment_end(self, state: AgentState, env_name: str = None) -> AgentState:
        """
        [OPTIONAL] Called when training on an environment ends.
        Can be used to reset environment-specific components.

        Args:
            state: Current agent state
            env_name: Name of the environment

        Returns:
            Updated agent state
        """
        return state

    def set_environment(
        self, state: AgentState, env_name: str = None, num_actions: int = None
    ) -> AgentState:
        """
        [OPTIONAL] Called when starting a new environment.
        Can be used to load task-specific parameters or update action masking.

        Args:
            state: Current agent state
            env_name: Name of the environment
            num_actions: Number of actions in the new environment (for action masking)

        Returns:
            Updated agent state
        """
        if num_actions is not None:
            self.current_num_actions = num_actions
        return state

    @abstractmethod
    def state_from_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        runtime_state: Any,
    ) -> AgentState:
        """
        Convert checkpoint data to AgentState.

        Orbax loads checkpoints as nested dictionaries. This method converts
        them back to the agent's specific dataclass structures.

        Args:
            checkpoint_data: Dictionary with saved parameters/optimizer state
            runtime_state: Current runtime state (buffer, recurrent state, etc.)

        Returns:
            AgentState with proper dataclass types
        """
        pass

    def reset(self, state: AgentState, rng: jax.random.PRNGKey) -> AgentState:
        """
        Reset agent state between tasks/environments.

        This should reinitialize optimizer states, running statistics, etc.
        while optionally preserving learned parameters and replay buffers
        for continual learning.

        Args:
            state: Current agent state
            rng: JAX random key for reinitialization

        Returns:
            Reset agent state (default: fresh initialization)
        """
        # Default implementation: full reinitialization
        # Subclasses can override to preserve parameters/buffers
        return self.init(rng)
