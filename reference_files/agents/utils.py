"""Utility classes and functions for RL agents."""
from typing import Dict, Any, Tuple, Optional

import jax

from cl.shared import setup_logger

logger = setup_logger(__name__)


def sg(x):
    """Stop gradient utility.

    A common shorthand for jax.lax.stop_gradient that's used throughout
    DreamerV3 and other agents to prevent gradients from flowing
    through specific tensors.

    Args:
        x: JAX array or PyTree

    Returns:
        Same as input but with gradients stopped

    Example:
        # Prevent actor-critic from backpropagating into world model
        features = sg(world_model_features)
        action = policy(features)
    """
    return jax.lax.stop_gradient(x)


def count_parameters(params: Any) -> int:
    """
    Count the total number of parameters in a JAX PyTree.

    Args:
        params: JAX parameter PyTree (nested dict of arrays)

    Returns:
        Total number of parameters

    Example:
        >>> state = agent.init(rng)
        >>> total = count_parameters(state.encoder_params)
        >>> print(f"Encoder has {total:,} parameters")
    """
    leaves = jax.tree_util.tree_leaves(params)
    return sum(x.size for x in leaves if hasattr(x, 'size'))


def count_parameters_by_component(
    params_dict: Dict[str, Any],
    show_breakdown: bool = True
) -> Tuple[int, Dict[str, int]]:
    """
    Count parameters for each component in a parameter dictionary.

    Args:
        params_dict: Dictionary mapping component names to parameter PyTrees
        show_breakdown: If True, print breakdown to stdout

    Returns:
        Tuple of (total_params, component_counts_dict)

    Example:
        >>> state = agent.init(rng)
        >>> components = {
        ...     'encoder': state.encoder_params,
        ...     'decoder': state.decoder_params,
        ...     'rssm': state.rssm_params,
        ... }
        >>> total, breakdown = count_parameters_by_component(components)
    """
    breakdown = {}
    total = 0

    for name, params in params_dict.items():
        count = count_parameters(params)
        breakdown[name] = count
        total += count

    if show_breakdown:
        print("Parameter Breakdown:")
        print("-" * 50)
        for name, count in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / total if total > 0 else 0
            print(f"  {name:20s}: {_format_number(count):>10s} ({pct:5.1f}%)")
        print("-" * 50)
        print(f"  {'TOTAL':20s}: {_format_number(total):>10s}")
        print()

    return total, breakdown


def get_agent_parameter_count(
    agent_state,
    trainable_keys: Optional[list] = None,
    show_breakdown: bool = True
) -> Tuple[int, int, Dict[str, int]]:
    """
    Count trainable and non-trainable parameters in an agent state.

    In JAX, "trainable" parameters are those passed to the optimizer.
    For DreamerV3, slow_critic is updated via EMA, not gradients.

    Args:
        agent_state: Agent state containing parameter trees
        trainable_keys: List of state keys that are trainable. If None, assumes
                       all except 'slow_critic_params' and 'buffer_state'
        show_breakdown: If True, print breakdown to stdout

    Returns:
        Tuple of (trainable_params, non_trainable_params, all_components_dict)

    Example:
        >>> state = agent.init(rng)
        >>> trainable, non_trainable, breakdown = get_agent_parameter_count(state)
        >>> print(f"Trainable: {trainable:,}, Non-trainable: {non_trainable:,}")
    """
    # Handle PPO and policy-gradient agents (with train_state)
    if hasattr(agent_state, 'train_state') and hasattr(agent_state.train_state, 'params'):
        # All parameters in train_state.params are trainable
        trainable_total = count_parameters(agent_state.train_state.params)
        non_trainable_total = 0
        all_components = {'network': trainable_total}
    else:
        # Handle DreamerV3 and world model agents (with _params attributes)
        # Get all attributes that end with '_params'
        param_attrs = {
            k: v for k, v in vars(agent_state).items()
            if k.endswith('_params')
        }

        # Default trainable keys (everything except slow_critic)
        if trainable_keys is None:
            trainable_keys = [
                k for k in param_attrs.keys()
                if 'slow_critic' not in k.lower()
            ]

        # Count parameters
        all_components = {}
        trainable_total = 0
        non_trainable_total = 0

        for key, params in param_attrs.items():
            count = count_parameters(params)
            all_components[key] = count

            if key in trainable_keys:
                trainable_total += count
            else:
                non_trainable_total += count

    if show_breakdown:
        print("=" * 60)
        print("AGENT PARAMETER COUNT")
        print("=" * 60)
        print()

        # Show trainable parameters
        print("Trainable Parameters (updated by optimizer):")
        print("-" * 60)
        trainable_components = {k: v for k, v in all_components.items() if k in trainable_keys}
        for name, count in sorted(trainable_components.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / trainable_total if trainable_total > 0 else 0
            # Clean up name for display
            display_name = name.replace('_params', '')
            print(f"  {display_name:20s}: {_format_number(count):>10s} ({pct:5.1f}%)")
        print("-" * 60)
        print(f"  {'Subtotal':20s}: {_format_number(trainable_total):>10s}")
        print()

        # Show non-trainable parameters if any
        if non_trainable_total > 0:
            print("Non-Trainable Parameters (EMA updated):")
            print("-" * 60)
            non_trainable_components = {k: v for k, v in all_components.items() if k not in trainable_keys}
            for name, count in sorted(non_trainable_components.items(), key=lambda x: x[1], reverse=True):
                display_name = name.replace('_params', '')
                print(f"  {display_name:20s}: {_format_number(count):>10s}")
            print("-" * 60)
            print(f"  {'Subtotal':20s}: {_format_number(non_trainable_total):>10s}")
            print()

        # Total
        total = trainable_total + non_trainable_total
        print("=" * 60)
        print(f"{'TOTAL PARAMETERS':20s}: {_format_number(total):>10s}")
        print("=" * 60)
        print()

        # Memory estimate
        param_memory_gb = (total * 4) / 1e9  # float32
        optimizer_memory_gb = (trainable_total * 4 * 2) / 1e9  # 2 states (momentum + variance)
        print("Estimated memory:")
        print(f"  Parameters (float32): {param_memory_gb:.2f} GB")
        print(f"  Optimizer state: {optimizer_memory_gb:.2f} GB")
        print(f"  Total: {param_memory_gb + optimizer_memory_gb:.2f} GB")
        print()

    return trainable_total, non_trainable_total, all_components


def _format_number(n: int) -> str:
    """Format number with K/M/B suffix."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return f"{n:,}"


class RatioTracker:
    """
    Tracks training ratio to determine how many gradient updates to do at each step.

    Based on embodied.when.Ratio from DreamerV3 codebase.
    Used to implement the training schedule where gradient_steps / env_steps = ratio.

    Example:
        ratio_tracker = RatioTracker(0.5)  # Train every 2 steps on average

        for step in range(100):
            # ... collect environment data ...
            num_updates = ratio_tracker(step)
            for _ in range(num_updates):
                # ... perform gradient update ...

    Args:
        ratio: Desired ratio of gradient updates to environment steps.
               - ratio = 0.5 means 1 gradient update per 2 env steps
               - ratio = 2.0 means 2 gradient updates per 1 env step
    """

    def __init__(self, ratio: float):
        self._ratio = ratio
        self._prev = None
        self._zero_updates_count = 0  # Track consecutive zero updates
        self._last_step = None  # Track if step is incrementing

    def __call__(self, step: int) -> int:
        """
        Calculate how many gradient updates should be performed at this step.

        Args:
            step: Current environment step number

        Returns:
            Number of gradient updates to perform (0, 1, 2, ...)
        """
        step = int(step)

        if self._ratio == 0:
            return 0

        if self._ratio < 0:
            return 1

        # First call - initialize and return 1 update
        if self._prev is None:
            self._prev = step
            self._last_step = step
            return 1

        # BUG DETECTION: Check if step counter is stuck (not incrementing)
        if self._last_step is not None and step == self._last_step:
            logger.warning(
                f"CRITICAL BUG: RatioTracker called with same step={step} multiple times! "
                f"This means the step counter is not being incremented between calls. "
                f"Training will stall with 0 gradient updates. "
                f"Check that runtime.step is being incremented properly in the training loop."
            )

        # Calculate how many updates are "due" based on the ratio
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio

        # BUG DETECTION: Warn if we're consistently returning 0 updates
        if repeats == 0:
            self._zero_updates_count += 1
            if self._zero_updates_count >= 50:
                logger.warning(
                    f"RatioTracker has returned 0 updates for 50 consecutive calls! "
                    f"Current step={step}, prev={self._prev}, ratio={self._ratio}. "
                    f"This may indicate a bug where the step counter is not incrementing properly."
                )
                self._zero_updates_count = 0  # Reset to avoid spam
        else:
            self._zero_updates_count = 0

        self._last_step = step
        return repeats
