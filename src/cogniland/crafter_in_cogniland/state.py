"""EnvState + EnvParams pytrees for the crafter_in_cogniland env.

Both are ``flax.struct.dataclass`` so they're JAX pytree compatible —
JIT, vmap, and lax.scan can traverse them transparently. EnvParams is
the "static" side of the env (pre-generated map dataset + scalar
options); EnvState is the per-episode dynamic side.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class EnvParams:
    """Per-env static parameters; carries the loaded map dataset.

    Built once at startup by ``maps.load_map_arrays(...)`` +
    ``EnvParams.from_map_arrays(...)``. Lives on-device, so the env
    factory closes over it via ``default_params`` rather than passing it
    every step.

    Shapes:
        terrain: (N_maps, H, W) int8  — tile ids per cell
        spawn:   (N_maps, 2)    int32 — (r, c) agent spawn
        target:  (N_maps, 2)    int32 — (r, c) target cell
        ctg_none / ctg_raft / ctg_harness: (N_maps, H, W) float32
            — Dijkstra unit-cost distance to target under each item. Used
              by the PBRS shaping reward; inf for unreachable cells. The
              env switches between the three on `active_object` change.
        map_type: (N_maps,) int8 — 0=balanced, 1=lake, 2=rocky
    """
    terrain: jax.Array       # (N, H, W) int8
    spawn: jax.Array         # (N, 2) int32
    target: jax.Array        # (N, 2) int32
    ctg_none: jax.Array      # (N, H, W) float32
    ctg_raft: jax.Array      # (N, H, W) float32
    ctg_harness: jax.Array   # (N, H, W) float32
    map_type: jax.Array      # (N,) int8
    max_steps: int = struct.field(pytree_node=False, default=1000)
    view_size: int = struct.field(pytree_node=False, default=21)

    @property
    def num_maps(self) -> int:
        return int(self.terrain.shape[0])

    @property
    def map_size(self) -> int:
        return int(self.terrain.shape[-1])

    @classmethod
    def from_map_arrays(
        cls,
        terrain: np.ndarray,
        spawn: np.ndarray,
        target: np.ndarray,
        ctg_none: np.ndarray,
        ctg_raft: np.ndarray,
        ctg_harness: np.ndarray,
        map_type: np.ndarray,
        *,
        max_steps: int = 1000,
        view_size: int = 21,
    ) -> "EnvParams":
        # Sanitize inf in ctg arrays (Dijkstra returns inf for unreachable)
        # — replace with a finite "big" so JAX arithmetic doesn't NaN.
        BIG = np.float32(1e6)
        ctg_none = np.where(np.isfinite(ctg_none), ctg_none, BIG).astype(np.float32)
        ctg_raft = np.where(np.isfinite(ctg_raft), ctg_raft, BIG).astype(np.float32)
        ctg_harness = np.where(np.isfinite(ctg_harness), ctg_harness, BIG).astype(np.float32)
        return cls(
            terrain=jnp.asarray(terrain, dtype=jnp.int8),
            spawn=jnp.asarray(spawn, dtype=jnp.int32),
            target=jnp.asarray(target, dtype=jnp.int32),
            ctg_none=jnp.asarray(ctg_none, dtype=jnp.float32),
            ctg_raft=jnp.asarray(ctg_raft, dtype=jnp.float32),
            ctg_harness=jnp.asarray(ctg_harness, dtype=jnp.float32),
            map_type=jnp.asarray(map_type, dtype=jnp.int8),
            max_steps=int(max_steps),
            view_size=int(view_size),
        )


@struct.dataclass
class EnvState:
    """Per-episode dynamic state."""
    map_idx: jax.Array       # () int32 — which map this episode samples from
    agent_r: jax.Array       # () int32
    agent_c: jax.Array       # () int32
    facing: jax.Array        # () int32  in {0,1,2,3}
    active_object: jax.Array # () int32  0=NONE, 1=RAFT, 2=HARNESS
    step_count: jax.Array    # () int32
    last_ctg: jax.Array      # () float32 — ctg under `active_object` at agent_pos
