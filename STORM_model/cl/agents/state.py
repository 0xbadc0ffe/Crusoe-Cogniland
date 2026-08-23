"""State dataclasses for all agent types (world model-based and model-free).

This module defines unified state structures following DreamerV3 conventions.

ALL AGENTS use the same AgentState structure:
    1. wm_params: World model parameters (None for model-free agents)
    2. policy_params: Policy/actor-critic parameters
    3. opt: Optimizer state
    4. runtime: Runtime bookkeeping (buffer, counters, RNG, wm_state)

Benefits:
    - Consistent naming across all agents
    - Clear ownership: Parameters vs. optimizer vs. runtime
    - Easy serialization: Save params + opt, discard runtime
    - Testable: Each component can be tested independently
    - JIT-friendly: Pass only what you need across boundaries

Usage patterns:
    # World model agent (DreamerV3)
    AgentState(
        wm_params=WorldModelParams(...),
        policy_params=PolicyParams(actor=..., critic=..., ...),
        opt=OptState(...),
        runtime=RuntimeState(buffer_state=..., wm_state=..., ...),
    )

    # Model-free agent with replay (CLEAR)
    AgentState(
        wm_params=None,  # No world model
        policy_params=PolicyParams(actor=..., critic=..., ...),
        opt=OptState(...),
        runtime=RuntimeState(buffer_state=..., wm_state=None, ...),
    )

    # Model-free agent without replay (PPO, IMPALA)
    AgentState(
        wm_params=None,  # No world model
        policy_params=PolicyParams(actor=..., critic=..., ...),
        opt=OptState(...),
        runtime=RuntimeState(buffer_state=None, wm_state=None, ...),
    )
"""

from typing import Any, Dict, Optional
import chex
import jax
import jax.numpy as jnp

# Import WorldModelParams from base.py (single source of truth)
from cl.agents.world_models.base import WorldModelParams, WorldModelState
from cl.agents.commons.replay_buffer import ReplayBufferState


@chex.dataclass
class PolicyParams:
    """Parameters for policy/value networks (trainable).

    For world model agents:
        - actor: Policy network that operates on world model features
        - critic: Value network that operates on world model features
        - slow_critic: EMA target critic for stable TD learning
        - normalizers: Running statistics for returns, values, advantages

    For model-free agents:
        - actor: Policy network that operates on observations (processed with CNN/MLP)
        - critic: Value network that operates on observations (processed with CNN/MLP)
        - slow_critic: Not typically used (can be None)
        - normalizers: Not typically used (can be empty dict)

    Attributes:
        actor: Actor network parameters (policy π(a|s))
        critic: Critic network parameters (value V(s))
        slow_critic: Slow critic parameters (EMA target for TD learning)
        normalizers: Dictionary of normalizer parameters
                    (e.g., {"returns": {...}, "values": {...}, "advantages": {...}})

    Note: slow_critic is an exponential moving average (EMA) of critic parameters,
          used as a stable target for temporal difference learning (reduces oscillations).
    """
    actor: Any
    critic: Any
    slow_critic: Any  # EMA target (can be None for some agents)
    normalizers: Dict[str, Any]  # retnorm, valnorm, advnorm


@chex.dataclass
class OptState:
    """Optimizer state (dynamic, trainable).

    Contains the internal state of the optimizer (e.g., momentum, adaptive
    learning rates, gradient accumulators). This is updated by the optimizer
    during training and needs to be saved for checkpoint resumption.

    Attributes:
        opt_state: Optimizer-specific state (e.g., optax's ScaleByAdamState)

    Note: The specific structure depends on which optimizer is used (Adam, LaProp, etc.).
          This is an opaque object from the optimizer's perspective.
    """
    opt_state: Any


@chex.dataclass
class RuntimeState:
    """Runtime bookkeeping for all agents (dynamic, non-trainable).

    Contains ephemeral state that doesn't need to be saved in checkpoints.
    This includes counters, RNG keys, replay buffer contents, and optionally
    the current world model latent state (for world model agents).

    Attributes:
        buffer_state: Replay buffer state (ReplayBufferState or None)
                     - None for agents without replay (PPO, IMPALA)
                     - ReplayBufferState for agents with replay (DreamerV3, CLEAR)
        wm_state: Current world model latent state (WorldModelState or None)
                 - WorldModelState for world model agents (DreamerV3)
                 - None for model-free agents (PPO, IMPALA, CLEAR)
        step: Environment interaction steps [scalar]
        train_steps: Training/optimization steps [scalar]
        rng: JAX random key for reproducibility

    JIT hygiene notes:
        - All counters use jnp.ndarray (not Python int) for JIT compatibility
        - Counters are wrapped in jnp.array() during initialization
        - This prevents shape/type changes across JIT compilations

    IMPORTANT JIT INVARIANT:
        NEVER pass full AgentState (or RuntimeState) into jax.jit functions!
        Because buffer_state contains large NumPy arrays, JAX will try to move
        them to GPU, causing "GPU out of memory" issues.

        ✅ CORRECT: Pass only what you need
            @jax.jit
            def train_step(wm_params, policy_params, opt_state, batch, rng):
                ...

        ❌ WRONG: Don't pass full state
            @jax.jit
            def train_step(agent_state, batch):  # buffer_state goes to GPU!
                ...

        Keep buffer operations (sampling, adding) OUTSIDE jitted functions.

    Note: Runtime state is typically NOT saved in checkpoints (it's ephemeral).
          Exception: You might want to save step counters for logging continuity.
    """
    buffer_state: Optional[ReplayBufferState]  # None for agents without replay
    wm_state: Optional[WorldModelState]  # None for model-free agents
    step: jnp.ndarray      # Environment steps
    train_steps: jnp.ndarray  # Training/optimization steps
    rng: jax.random.PRNGKey


@chex.dataclass
class AgentState:
    """Complete agent state for all agent types (world model and model-free).

    This is the unified top-level state structure that combines all components.
    It provides a clean separation between:
        - Trainable state (train_state) - holds params, optimizer, and agent-specific state
        - Runtime bookkeeping (runtime) - ephemeral state like buffers, RNG, counters

    Attributes:
        train_state: Agent-specific TrainState (polymorphic)
                    - DreamerV3TrainState for DreamerV3
                    - flax.training.train_state.TrainState for PPO/IMPALA (for now)
                    - Future: PPOTrainState, IMPALATrainState, STORMTrainState, etc.
        runtime: Runtime bookkeeping (RuntimeState) - buffer, counters, RNG, wm_state

    Benefits over previous manual structure:
        - Impossible to forget to add a param to the optimizer dict
        - Agent-specific logic (like EMA) is encapsulated in the TrainState
        - Cleaner training loops (no manual dictionary packing/unpacking)
        - Type-safe access to agent-specific components

    Usage examples:

        # World model agent (DreamerV3)
        from cl.agents.world_models.dreamerv3 import DreamerV3TrainState, DreamerV3Params

        train_state = DreamerV3TrainState.create(
            apply_fn=None,
            params=DreamerV3Params(wm=wm_params, policy=policy_params),
            tx=optimizer,
            slow_critic=initial_slow_critic,
            normalizers=initial_normalizers,
            slow_critic_rate=0.02,
        )

        state = AgentState(
            train_state=train_state,
            runtime=RuntimeState(
                buffer_state=replay_buffer.get_state(),
                wm_state=initial_latent_state,
                step=jnp.array(0),
                train_steps=jnp.array(0),
                rng=rng,
            ),
        )

        # Model-free agent (PPO, IMPALA)
        # For now, use flax.training.train_state.TrainState directly
        # Future: Create PPOTrainState, IMPALATrainState similar to DreamerV3TrainState
        state = AgentState(
            train_state=train_state,  # Standard TrainState
            runtime=RuntimeState(
                buffer_state=None,  # No replay buffer for PPO
                wm_state=None,      # No world model state
                step=jnp.array(0),
                train_steps=jnp.array(0),
                rng=rng,
            ),
        )

    Checkpoint structure:
        checkpoint = {
            "_version": "v2",  # Note: version bump due to structure change
            "train_state": train_state,  # TrainState is a PyTree
            "config": {...},
        }
        # Note: runtime is NOT saved (it's ephemeral)

    Note: This structure is designed to be:
          - Serializable (for checkpointing)
          - Composable (different agents use different TrainStates)
          - JIT-friendly (clear boundaries for compilation)
          - Testable (each component can be mocked independently)
          - Unified (same AgentState structure for all agent types)
    """
    train_state: Any  # Polymorphic - agent-specific TrainState (DreamerV3TrainState, etc.)
    runtime: RuntimeState


# Export public API
__all__ = [
    'WorldModelParams',  # Re-export from base.py for convenience
    'WorldModelState',   # Re-export from base.py for convenience
    'PolicyParams',
    'OptState',
    'RuntimeState',
    'AgentState',
]