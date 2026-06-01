"""EnvState + EnvParams pytrees for the pure-JAX bridge_tunnel_commit env.

Mirrors ``bridge_tunnel_jax/state.py`` with two additions for the commitment
mechanic:

* ``EnvParams.ctg`` is now ``(N, 3, H, W)`` — three commitment-indexed static
  cost-to-go fields (none / build / mine), matching
  ``BridgeTunnelCommitEnv._compute_all_ctg``. The PBRS potential is
  ``φ = −ctg[commit]`` so the active field switches when the agent commits.
* ``EnvParams.category`` is ``(N,)`` int32 (0=balanced, 1=lakes, 2=rocky) — a
  label carried only for logging / eval (not used by the dynamics).
* ``EnvState.commit`` is the per-episode irreversible commitment slot.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class EnvParams:
    """Per-env static parameters; carries the precomputed map dataset.

    Shapes:
        terrain:   (N, H, W)    int8
        spawn:     (N, 2)       int32
        target:    (N, 2)       int32
        goal_mask: (N, H, W)    bool
        ctg:       (N, 3, H, W) float32  — commit-indexed [none, build, mine]
        category:  (N,)         int32    — 0 balanced / 1 lakes / 2 rocky
    """
    terrain: jax.Array        # (N, H, W) int8
    spawn: jax.Array          # (N, 2) int32
    target: jax.Array         # (N, 2) int32
    goal_mask: jax.Array      # (N, H, W) bool
    ctg: jax.Array            # (N, 3, H, W) float32
    category: jax.Array       # (N,) int32
    max_steps: int = struct.field(pytree_node=False, default=800)
    view_size: int = struct.field(pytree_node=False, default=21)
    slack_penalty: float = struct.field(pytree_node=False, default=-0.01)
    reach_bonus: float = struct.field(pytree_node=False, default=1.0)
    shaping_coef: float = struct.field(pytree_node=False, default=0.01)
    build_cost: float = struct.field(pytree_node=False, default=0.05)
    gamma: float = struct.field(pytree_node=False, default=0.99)

    @property
    def num_maps(self) -> int:
        return int(self.terrain.shape[0])

    @property
    def height(self) -> int:
        return int(self.terrain.shape[1])

    @property
    def width(self) -> int:
        return int(self.terrain.shape[2])

    @classmethod
    def from_map_arrays(
        cls,
        terrain: np.ndarray,
        spawn: np.ndarray,
        target: np.ndarray,
        goal_mask: np.ndarray,
        ctg: np.ndarray,
        category: np.ndarray,
        *,
        max_steps: int = 800,
        view_size: int = 21,
        slack_penalty: float = -0.01,
        reach_bonus: float = 1.0,
        shaping_coef: float = 0.01,
        build_cost: float = 0.05,
        gamma: float = 0.99,
    ) -> "EnvParams":
        assert ctg.ndim == 4 and ctg.shape[1] == 3, \
            f"ctg must be (N,3,H,W), got {ctg.shape}"
        return cls(
            terrain=jnp.asarray(terrain, dtype=jnp.int8),
            spawn=jnp.asarray(spawn, dtype=jnp.int32),
            target=jnp.asarray(target, dtype=jnp.int32),
            goal_mask=jnp.asarray(goal_mask, dtype=bool),
            ctg=jnp.asarray(ctg, dtype=jnp.float32),
            category=jnp.asarray(category, dtype=jnp.int32),
            max_steps=int(max_steps),
            view_size=int(view_size),
            slack_penalty=float(slack_penalty),
            reach_bonus=float(reach_bonus),
            shaping_coef=float(shaping_coef),
            build_cost=float(build_cost),
            gamma=float(gamma),
        )


@struct.dataclass
class EnvState:
    """Per-episode dynamic state."""
    map_idx: jax.Array       # () int32 — which map this episode samples from
    terrain: jax.Array       # (H, W) int8 — mutable (BUILD/MINE edit it)
    agent_r: jax.Array       # () int32
    agent_c: jax.Array       # () int32
    facing: jax.Array        # () int32 in {0,1,2,3}
    commit: jax.Array        # () int32 in {0 none, 1 build, 2 mine}
    step_count: jax.Array    # () int32
