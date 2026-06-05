"""EnvState + EnvParams for the pure-JAX bridge_tunnel env (both variants).

``EnvParams.commit`` is a STATIC flag (pytree_node=False) selecting the variant:

* ``commit=False`` (bt): ``ctg`` is ``(N, H, W)``; ``category`` unused.
* ``commit=True`` (btc): ``ctg`` is ``(N, 3, H, W)`` commit-indexed; ``category``
  is ``(N,)`` int32 (0 balanced / 1 lakes / 2 rocky).

``from_map_arrays`` infers ``commit`` from ``ctg.ndim`` (4 → commit), so the same
constructor serves both variants and both parity tests' ``from_map_arrays(**arrays)``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class EnvParams:
    terrain: jax.Array        # (N, H, W) int8
    spawn: jax.Array          # (N, 2) int32
    target: jax.Array         # (N, 2) int32
    goal_mask: jax.Array      # (N, H, W) bool
    ctg: jax.Array            # bt (N,H,W) | btc (N,3,H,W) float32
    category: jax.Array       # (N,) int32 (btc; zeros for bt)
    commit: bool = struct.field(pytree_node=False, default=False)
    max_steps: int = struct.field(pytree_node=False, default=800)
    view_size: int = struct.field(pytree_node=False, default=21)
    slack_penalty: float = struct.field(pytree_node=False, default=-0.01)
    reach_bonus: float = struct.field(pytree_node=False, default=1.0)
    shaping_coef: float = struct.field(pytree_node=False, default=0.01)
    build_cost: float = struct.field(pytree_node=False, default=0.0)
    commit_cost: float = struct.field(pytree_node=False, default=0.05)
    illegal_penalty: float = struct.field(pytree_node=False, default=0.02)
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
    def from_map_arrays(cls, terrain, spawn, target, goal_mask, ctg, category=None,
                        *, max_steps=800, view_size=21, slack_penalty=-0.01,
                        reach_bonus=1.0, shaping_coef=0.01, build_cost=0.0,
                        commit_cost=0.05, illegal_penalty=0.02, gamma=0.99):
        ctg = np.asarray(ctg)
        commit = (ctg.ndim == 4)        # (N,3,H,W) ⇒ commit variant
        if category is None:
            category = np.zeros((terrain.shape[0],), dtype=np.int32)
        return cls(
            terrain=jnp.asarray(terrain, jnp.int8),
            spawn=jnp.asarray(spawn, jnp.int32),
            target=jnp.asarray(target, jnp.int32),
            goal_mask=jnp.asarray(goal_mask, bool),
            ctg=jnp.asarray(ctg, jnp.float32),
            category=jnp.asarray(category, jnp.int32),
            commit=bool(commit), max_steps=int(max_steps), view_size=int(view_size),
            slack_penalty=float(slack_penalty), reach_bonus=float(reach_bonus),
            shaping_coef=float(shaping_coef), build_cost=float(build_cost),
            commit_cost=float(commit_cost), illegal_penalty=float(illegal_penalty),
            gamma=float(gamma),
        )


@struct.dataclass
class EnvState:
    map_idx: jax.Array       # () int32
    terrain: jax.Array       # (H, W) int8 — mutable
    agent_r: jax.Array       # () int32
    agent_c: jax.Array       # () int32
    facing: jax.Array        # () int32
    commit: jax.Array        # () int32 (0/1/2; stays 0 for bt)
    step_count: jax.Array    # () int32
