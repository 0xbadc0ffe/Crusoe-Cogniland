"""DreamerV3-specific state definitions.

This module defines the parameter structure and TrainState for DreamerV3.
By keeping this separate from the general state.py, we make it clear that
this is specific to DreamerV3's architecture.
"""

from typing import Any, Dict
import chex
import jax.tree_util as jtu
from flax.training import train_state
from flax import struct

from cl.agents.world_models.base import WorldModelParams
from cl.agents.state import PolicyParams


@chex.dataclass
class DreamerV3Params:
    """Composite parameter structure for DreamerV3.

    This creates the Single Source of Truth for trainable parameters in DreamerV3.
    All gradient computations operate on this structure, eliminating manual
    dictionary packing/unpacking in the training loop.

    Attributes:
        wm: World model parameters (encoder, decoder, dynamics, reward, continuation)
        policy: Policy parameters (actor, critic - but NOT slow_critic or normalizers)

    Note: slow_critic and normalizers are NOT stored here because they are NOT
          updated via gradients:
          - slow_critic: Updated via EMA (exponential moving average)
          - normalizers: Running statistics (updated from batch statistics)

          These are stored in DreamerV3TrainState instead.
    """
    wm: WorldModelParams
    policy: PolicyParams  # Contains actor and critic params ONLY


class DreamerV3TrainState(train_state.TrainState):
    """Training state for DreamerV3.

    Extends Flax's TrainState to:
    1. Hold composite DreamerV3Params (world model + policy)
    2. Store non-gradient state (slow_critic, normalizers)
    3. Encapsulate EMA update logic in apply_gradients()

    Attributes:
        params: DreamerV3Params (trainable via gradients)
        tx: Optax optimizer
        opt_state: Optimizer internal state
        step: Training step counter
        slow_critic: EMA target critic (updated via exponential moving average)
        normalizers: Running statistics (return, value, advantage normalizers)
        slow_critic_rate: EMA interpolation rate (e.g., 0.02 = 2% new, 98% old)

    Benefits over manual management:
        - Impossible to forget to add a new param to the optimizer
        - EMA update is encapsulated in the state class
        - Normalizers are bundled with the state they belong to
        - Cleaner training loop (no dictionary shuffling)
    """
    # Override params type annotation for better type checking
    params: DreamerV3Params

    # Non-gradient state (not optimized, updated separately)
    slow_critic: Any  # EMA target critic
    normalizers: Dict[str, Any]  # {return, value, advantage} normalizers

    # Hyperparameters (not part of PyTree)
    slow_critic_rate: float = struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, **kwargs):
        """Apply gradients and update slow critic via EMA.

        This method:
        1. Calls parent's apply_gradients to update params and opt_state
        2. Automatically updates slow_critic via exponential moving average

        Args:
            grads: Gradients (must match DreamerV3Params structure)
            **kwargs: Additional arguments for parent's apply_gradients

        Returns:
            New DreamerV3TrainState with updated params, opt_state, and slow_critic
        """
        # 1. Standard Optax update (params + opt_state)
        new_state = super().apply_gradients(grads=grads, **kwargs)

        # 2. Automatic EMA update for slow critic
        # slow_critic = (1 - rate) * old_slow + rate * new_critic
        new_slow_critic = jtu.tree_map(
            lambda slow, fast: (1 - self.slow_critic_rate) * slow + self.slow_critic_rate * fast,
            self.slow_critic,
            new_state.params.policy.critic  # Access critic from PolicyParams
        )

        # Return updated state with new slow critic
        return new_state.replace(slow_critic=new_slow_critic)

    @classmethod
    def create(cls, *, apply_fn, params, tx, slow_critic, normalizers, slow_critic_rate, **kwargs):
        """Create a new DreamerV3TrainState.

        Args:
            apply_fn: Not used (kept for compatibility with TrainState.create)
            params: DreamerV3Params (composite world model + policy params)
            tx: Optax optimizer
            slow_critic: Initial EMA target critic (typically copy of params.policy.critic)
            normalizers: Initial normalizer states (return, value, advantage)
            slow_critic_rate: EMA interpolation rate (e.g., 0.02)
            **kwargs: Additional arguments for TrainState (if any)

        Returns:
            Initialized DreamerV3TrainState
        """
        # Initialize optimizer state
        opt_state = tx.init(params)

        # Create the TrainState with all fields (standard + custom)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            slow_critic=slow_critic,
            normalizers=normalizers,
            slow_critic_rate=slow_critic_rate,
            **kwargs,
        )


# Export public API
__all__ = [
    'DreamerV3Params',
    'DreamerV3TrainState',
]
