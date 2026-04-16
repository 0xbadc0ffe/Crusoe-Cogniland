"""Policy implementations for world model-based agents.

This package provides policy training algorithms that can operate on top of
world models. The key idea is to use world models for imagination/planning,
then train policies on imagined trajectories.

Key components:
    - base.py: Policy protocol (abstract interface)
    - mlp.py: MLP policy implementation (for DreamerV3, STORM)
    - actor_critic.py: Imagination-based actor-critic training
    - ppo_policy.py: PPO-style policy (future)
    - utils.py: Shared utilities (future)

Design philosophy:
    1. World-model agnostic: Works with any WorldModel implementation
    2. Functional: Pure functions, no state
    3. Reusable: Shared across different world model architectures
    4. Efficient: Designed for JIT compilation
"""

from .base import Policy
from .mlp import MLPPolicy
from .actor_critic import PolicyFn, imagine_trajectory, imag_loss, repl_loss

__all__ = [
    # Policy classes
    'Policy',
    'MLPPolicy',
    # Actor-critic functions
    'PolicyFn',
    'imagine_trajectory',
    'imag_loss',
    'repl_loss',
]
