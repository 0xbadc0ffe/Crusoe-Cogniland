"""STORM-specific state definitions.

This module defines the state structure for STORM's Transformer State-Space Model (TSSM).
Includes STORM-specific params and train state classes for clean decoupling from DreamerV3.
"""

from typing import List, Optional, Dict, Any
import chex
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training import train_state
from flax import struct

from cogniland.agents.world_models.base import WorldModelState, WorldModelParams
from cogniland.agents.state import PolicyParams


@chex.dataclass
class TSSMState(WorldModelState):
    """Latent state for Transformer State-Space Model (TSSM).

    Unlike RSSM which has {stoch, deter}, TSSM has {stoch, hidden, kv_cache}.

    Attributes:
        stoch: Stochastic categorical samples, shape [B, stoch, classes] where
               each of the 'stoch' categorical distributions has 'classes' options
        hidden: Transformer hidden state, shape [B, hidden_dim]
        kv_cache: List of key-value cache dicts per transformer layer.
                  Each dict has {'keys': [B, max_cache_len, hidden_dim], 'values': [B, max_cache_len, hidden_dim]}
                  Fixed-size rolling buffer for JAX scan compatibility
        cache_step: Number of tokens in cache (for rolling buffer pointer), shape [B]
        logits: Optional logits for KL computation, shape [B, stoch, classes]
    """
    stoch: jnp.ndarray       # [B, stoch, classes] - categorical samples
    hidden: jnp.ndarray      # [B, hidden_dim] - transformer hidden state
    kv_cache: List[Dict[str, jnp.ndarray]]  # List of {'keys': [B, max_len, D], 'values': [B, max_len, D]}
    cache_step: jnp.ndarray  # [B] - number of tokens in cache (for rolling buffer)

    # Optional: include logits for KL computation (used in observe())
    logits: Optional[jnp.ndarray] = None  # [B, stoch, classes]


@chex.dataclass
class STORMParams:
    """Composite parameter structure for STORM.

    This creates the Single Source of Truth for trainable parameters in STORM.
    All gradient computations operate on this structure.

    Attributes:
        wm: World model parameters (encoder, decoder, TSSM, reward, continuation)
        policy: Policy parameters (actor, critic - but NOT slow_critic or normalizers)

    Note: slow_critic and normalizers are NOT stored here because they are NOT
          updated via gradients:
          - slow_critic: Updated via EMA (exponential moving average)
          - normalizers: Running statistics (updated from batch statistics)

          These are stored in STORMTrainState instead.
    """
    wm: WorldModelParams
    policy: PolicyParams  # Contains actor and critic params ONLY


@chex.dataclass
class STORMJTPParams:
    """Composite parameter structure for STORM-JTP (includes fetch head)."""
    wm: WorldModelParams
    policy: PolicyParams
    fetch: Any  # Fetch head params


class STORMTrainState(train_state.TrainState):
    """Training state for STORM.

    Extends Flax's TrainState to:
    1. Hold composite STORMParams (world model + policy)
    2. Store non-gradient state (slow_critic, normalizers)
    3. Encapsulate EMA update logic in apply_gradients()

    Attributes:
        params: STORMParams (trainable via gradients)
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
    params: STORMParams

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
            grads: Gradients (must match STORMParams structure)
            **kwargs: Additional arguments for parent's apply_gradients

        Returns:
            New STORMTrainState with updated params, opt_state, and slow_critic
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
        """Create a new STORMTrainState.

        Args:
            apply_fn: Not used (kept for compatibility with TrainState.create)
            params: STORMParams (composite world model + policy params)
            tx: Optax optimizer
            slow_critic: Initial EMA target critic (typically copy of params.policy.critic)
            normalizers: Initial normalizer states (return, value, advantage)
            slow_critic_rate: EMA interpolation rate (e.g., 0.02)
            **kwargs: Additional arguments for TrainState (if any)

        Returns:
            Initialized STORMTrainState
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


class STORMJTPTrainState(STORMTrainState):
    """TrainState variant for STORM-JTP (params include fetch head)."""
    params: STORMJTPParams


# Export public API
__all__ = [
    'TSSMState',
    'STORMParams',
    'STORMTrainState',
    'STORMJTPParams',
    'STORMJTPTrainState',
]
