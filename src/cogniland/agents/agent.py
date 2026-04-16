"""Agent dataclass for functional agent pattern.

This module defines the Agent dataclass that holds all agent methods.
Factory functions return Agent instances instead of subclassing ContinualAgent.

Example usage:
    from cogniland.agents.storm import make_storm

    # Create agent (factory returns Agent dataclass)
    agent = make_storm(config, obs_space, act_space)

    # Initialize state
    state = agent.init(rng)

    # Train
    state, metrics = agent.train(state, env, rng, num_frames)

    # Evaluate
    eval_metrics = agent.evaluate(state, env, rng, num_frames)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp


# Type aliases for clarity
PRNGKey = jax.random.PRNGKey


def _identity_set_environment(state: Any, env_name: str = None, num_actions: int = None, task_id: int = None) -> Any:
    """Default set_environment: update current_num_actions and task_id in RuntimeState.

    This is the default implementation used when an agent doesn't provide
    a custom set_environment method. It handles the common case of updating
    the action space and task identifier for a new environment.

    Args:
        state: Current AgentState
        env_name: Name of the environment (optional, for logging)
        num_actions: Number of actions in the new environment
        task_id: Task index for replay buffer tracking

    Returns:
        Updated AgentState with new current_num_actions/current_task_id in RuntimeState
    """
    if hasattr(state, 'runtime') and state.runtime is not None:
        new_runtime = state.runtime
        if num_actions is not None:
            new_runtime = new_runtime.replace(current_num_actions=jnp.array(num_actions))
        if task_id is not None:
            new_runtime = new_runtime.replace(current_task_id=jnp.array(task_id, dtype=jnp.int32))
        if new_runtime is not state.runtime:
            return state.replace(runtime=new_runtime)
    return state


def _identity_on_environment_end(state: Any, env_name: str = None) -> Any:
    """Default on_environment_end: no-op.

    This is the default implementation used when an agent doesn't provide
    a custom on_environment_end method. Most agents don't need special
    cleanup when transitioning between environments.

    Args:
        state: Current AgentState
        env_name: Name of the environment that just ended

    Returns:
        Unchanged AgentState
    """
    return state


@dataclass
class Agent:
    """Container for agent functions. NOT a PyTree - just a namespace.

    This dataclass holds references to the agent's methods. The methods
    themselves close over the agent's configuration and modules (networks,
    buffers, optimizers) which are immutable bindings from the factory.

    The Agent dataclass replaces the ContinualAgent class hierarchy with
    a more functional approach that works better with JAX's transformations.

    Required Methods:
        init: Initialize agent state from random key
        train: Train agent for specified number of frames
        evaluate: Evaluate agent for specified number of frames
        select_action: Select action given observation
        state_from_checkpoint: Restore state from checkpoint data

    Optional Methods (with defaults):
        reset: Reset agent state (defaults to calling init)
        set_environment: Called when switching environments
        on_environment_end: Called when leaving an environment

    Optional Methods (for specific agent types):
        train_pure: Pure training function for vmap (PR4)
        get_latents: Extract latent representations (world model agents)

    Metadata:
        obs_space: Observation space specification
        action_space: Maximum number of actions across all environments

    Example:
        @register_agent('my_agent')
        def make_my_agent(config, obs_space, act_space) -> Agent:
            # Setup closures
            network = MyNetwork(...)
            optimizer = optax.adam(config.lr)

            def init(rng):
                params = network.init(rng, dummy_input)
                return AgentState(train_state=..., runtime=...)

            def train(state, env, rng, num_frames, progress_bar=None, checkpoint_callback=None):
                # Training logic
                return new_state, metrics

            # ... other methods ...

            return Agent(
                init=init,
                train=train,
                evaluate=evaluate,
                select_action=select_action,
                state_from_checkpoint=state_from_checkpoint,
                obs_space=obs_space,
                action_space=act_space,
            )
    """

    # === CORE METHODS (required) ===
    init: Callable[[PRNGKey], Any]
    """Initialize agent state.

    Args:
        rng: JAX random key for initialization

    Returns:
        Initial AgentState with train_state and runtime
    """

    train: Callable
    """Train agent for specified number of frames.

    Signature: (state, env, rng, num_frames, progress_bar=None, checkpoint_callback=None) -> (state, metrics)

    Args:
        state: Current AgentState
        env: Environment instance
        rng: JAX random key
        num_frames: Number of environment frames to train for
        progress_bar: Optional tqdm progress bar to update
        checkpoint_callback: Optional callback for checkpointing

    Returns:
        Tuple of (new_state, metrics_dict)
    """

    evaluate: Callable
    """Evaluate agent for specified number of frames.

    Signature: (state, env, rng, num_frames, progress_bar=None) -> metrics

    Args:
        state: Current AgentState
        env: Environment instance
        rng: JAX random key
        num_frames: Number of environment frames to evaluate for
        progress_bar: Optional tqdm progress bar

    Returns:
        Dictionary of evaluation metrics
    """

    select_action: Callable
    """Select action given observation.

    Signature: (state, obs, rng, is_first=None, prev_action=None, training=False) -> (actions, state)

    Args:
        state: Current AgentState
        obs: Observation (dict or array)
        rng: JAX random key
        is_first: Optional episode reset flags
        prev_action: Optional previous actions (for recurrent agents)
        training: Whether in training mode

    Returns:
        Tuple of (actions, updated_state)
    """

    state_from_checkpoint: Callable[[Dict, Any], Any]
    """Restore AgentState from checkpoint data.

    Args:
        checkpoint_data: Dictionary loaded from checkpoint (contains 'train_state')
        runtime_state: Current RuntimeState to merge with checkpoint

    Returns:
        Restored AgentState
    """

    # === RESET (optional - defaults to calling init) ===
    reset: Callable = None
    """Reset agent state for new episode/environment.

    Signature: (state, rng) -> AgentState

    If not provided, defaults to calling init(rng), which fully
    reinitializes the agent. Override this if you need to preserve
    certain state across resets (e.g., replay buffer, learned params).

    Args:
        state: Current AgentState
        rng: JAX random key

    Returns:
        Reset AgentState
    """

    # === ENVIRONMENT HOOKS (with defaults) ===
    set_environment: Callable = field(default=_identity_set_environment)
    """Called when switching to a new environment.

    Signature: (state, env_name=None, num_actions=None) -> state

    Updates the current_num_actions in RuntimeState for action masking.
    Override if you need additional environment-specific setup.

    Args:
        state: Current AgentState
        env_name: Name of the new environment
        num_actions: Number of actions in the new environment

    Returns:
        Updated AgentState
    """

    on_environment_end: Callable = field(default=_identity_on_environment_end)
    """Called when leaving an environment.

    Signature: (state, env_name=None) -> state

    Default is a no-op. Override if you need cleanup when
    transitioning between environments.

    Args:
        state: Current AgentState
        env_name: Name of the environment being left

    Returns:
        Updated AgentState
    """

    # === OPTIONAL (for PR4 vmap support) ===
    train_pure: Optional[Callable] = None
    """Pure training function for vmap across seeds.

    Signature: (state, env_state, rng, num_updates) -> (state, env_state, metrics)

    This is an optional method for PR4 multi-seed training. It provides
    a pure functional interface that can be vmapped across seeds.
    """

    # === OPTIONAL (for world model agents) ===
    get_latents: Optional[Callable] = None
    """Extract latent representations from the world model state.

    Signature: (state) -> latents

    For world model agents (STORM, DreamerV3), this extracts the
    latent state representations for analysis/visualization.
    Note: Call select_action first to update wm_state with latest observation.
    """

    # === OPTIONAL (for testing/debugging/analysis) ===
    buffer: Optional[Any] = None
    """Replay buffer instance for direct access.

    Exposed for testing and debugging purposes. Allows tests to sample
    batches directly via buffer.sample(state.runtime.buffer_state, rng).
    """

    world_model: Optional[Any] = None
    """World model instance for direct access.

    Exposed for notebooks and analysis. Allows calling wm.initial_state(),
    wm.encode(), wm.observe(), etc. directly with params from state.train_state.params.wm.
    """

    train_step: Optional[Callable] = None
    """Single training step function for direct access.

    Signature: (state, batch) -> (state, metrics)

    Exposed for testing and debugging. Allows tests to run individual
    training steps and inspect metrics without running the full train loop.
    """

    # === METADATA (immutable, set at creation) ===
    obs_space: Any = None
    """Observation space specification (dict or tuple)."""

    action_space: int = 0
    """Maximum number of actions across all environments."""

    def __post_init__(self):
        """Set default reset to call init if not provided."""
        if self.reset is None:
            # Default: full reinitialization
            # Use object.__setattr__ because dataclass is immutable after init
            object.__setattr__(self, 'reset', lambda state, rng: self.init(rng))


# Type alias for factory functions
AgentFactory = Callable[[Any, Any, int], Agent]
"""Type alias for agent factory functions.

Factory functions have signature:
    (config: OmegaConf, obs_space: Any, act_space: int) -> Agent
"""


__all__ = [
    'Agent',
    'AgentFactory',
    '_identity_set_environment',
    '_identity_on_environment_end',
]
