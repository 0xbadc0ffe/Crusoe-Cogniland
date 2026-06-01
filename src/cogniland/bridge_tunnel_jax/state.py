"""EnvState + EnvParams pytrees for the pure-JAX bridge_tunnel env.

Both are ``flax.struct.dataclass`` so they're JAX pytree compatible.
``EnvParams`` carries the precomputed natural-map dataset (terrain, spawn,
goal mask, and the *static-terrain* min-action cost-to-go field used by the
PBRS potential — see ``BridgeTunnelEnv._compute_ctg``). ``EnvState`` is the
per-episode dynamic side; it carries the **mutable** terrain grid because
PLACE (water→wood) and MINE (rock→grass) edit tiles in place.

The PyTorch oracle computes the ctg ONCE at reset from the initial terrain
and never recomputes it after a build/mine — so the PBRS potential is a
static lookup into ``ctg`` indexed by the agent's position. We replicate
that exactly: ``ctg`` lives in EnvParams (per map), indexed by ``map_idx``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class EnvParams:
    """Per-env static parameters; carries the precomputed natural-map dataset.

    Shapes:
        terrain:   (N, H, W) int8   — initial tile ids per cell
        spawn:     (N, 2)    int32  — (r, c) agent spawn
        target:    (N, 2)    int32  — representative target cell (for info)
        goal_mask: (N, H, W) bool   — every TARGET cell on the goal wall/door
        ctg:       (N, H, W) float32 — static-terrain min-action cost-to-go
                                       (Dijkstra) used as the PBRS potential
                                       φ = −ctg. Unreachable → BIG sentinel.
    """
    terrain: jax.Array        # (N, H, W) int8
    spawn: jax.Array          # (N, 2) int32
    target: jax.Array         # (N, 2) int32
    goal_mask: jax.Array      # (N, H, W) bool
    ctg: jax.Array            # (N, H, W) float32
    max_steps: int = struct.field(pytree_node=False, default=800)
    view_size: int = struct.field(pytree_node=False, default=21)
    slack_penalty: float = struct.field(pytree_node=False, default=-0.01)
    reach_bonus: float = struct.field(pytree_node=False, default=3.0)
    shaping_coef: float = struct.field(pytree_node=False, default=0.015)
    build_cost: float = struct.field(pytree_node=False, default=0.0)
    gamma: float = struct.field(pytree_node=False, default=0.997)

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
        *,
        max_steps: int = 800,
        view_size: int = 21,
        slack_penalty: float = -0.01,
        reach_bonus: float = 3.0,
        shaping_coef: float = 0.015,
        build_cost: float = 0.0,
        gamma: float = 0.997,
    ) -> "EnvParams":
        # The PyTorch _compute_ctg uses INF = H*W*4 for unreachable cells (a
        # plain int32, not float inf) — keep that exact sentinel so the shaping
        # numbers match bit-for-bit on reachable cells and agree on the rare
        # unreachable ones.
        return cls(
            terrain=jnp.asarray(terrain, dtype=jnp.int8),
            spawn=jnp.asarray(spawn, dtype=jnp.int32),
            target=jnp.asarray(target, dtype=jnp.int32),
            goal_mask=jnp.asarray(goal_mask, dtype=bool),
            ctg=jnp.asarray(ctg, dtype=jnp.float32),
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
    terrain: jax.Array       # (H, W) int8 — mutable (PLACE/MINE edit it)
    agent_r: jax.Array       # () int32
    agent_c: jax.Array       # () int32
    facing: jax.Array        # () int32 in {0,1,2,3}
    step_count: jax.Array    # () int32
