"""flax.struct EnvParams / EnvState for the JAX MemoryEnv."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from . import constants as C
from .maps import build_geometry


@struct.dataclass
class EnvState:
    agent_x: jax.Array        # () int32  (column)
    agent_y: jax.Array        # () int32  (row)
    agent_dir: jax.Array      # () int32  facing (0=E,1=S,2=W,3=N)
    cue_type: jax.Array       # () int32  in [0,4)  (CUE_TYPES order)
    door_green_top: jax.Array  # () bool  green door at the top? (else bottom)
    cue_x: jax.Array          # () int32  cue cell column
    cue_y: jax.Array          # () int32  cue cell row
    taken_branch: jax.Array   # () int32  BRANCH_NONE/UP/DOWN
    selected_door: jax.Array  # () int32  DOOR_NONE/SEL_GREEN/SEL_BLUE
    step_count: jax.Array     # () int32
    prev_phi: jax.Array       # () float32  (PBRS potential at last step)
    terminated: jax.Array     # () bool  (stepped on a door)
    done: jax.Array           # () bool  (terminated | truncated)


@struct.dataclass
class EnvParams:
    # pytree arrays
    base_terrain: jax.Array      # (H, W) int8 — walls + carved corridors (no cue/doors)
    cue_probs: jax.Array         # (4,) float32 — sampling probs over CUE_TYPES
    # static geometry anchors
    height: int = struct.field(pytree_node=False, default=7)
    width: int = struct.field(pytree_node=False, default=27)
    my: int = struct.field(pytree_node=False, default=3)
    row_up: int = struct.field(pytree_node=False, default=1)
    row_lo: int = struct.field(pytree_node=False, default=5)
    row_room_up: int = struct.field(pytree_node=False, default=2)
    row_room_lo: int = struct.field(pytree_node=False, default=4)
    row_door_top: int = struct.field(pytree_node=False, default=1)
    row_door_bot: int = struct.field(pytree_node=False, default=5)
    x_precue_start: int = struct.field(pytree_node=False, default=1)
    x_room_start: int = struct.field(pytree_node=False, default=6)
    x_room_end: int = struct.field(pytree_node=False, default=9)
    x_pre_end: int = struct.field(pytree_node=False, default=14)
    x_branch_start: int = struct.field(pytree_node=False, default=15)
    x_branch_end: int = struct.field(pytree_node=False, default=18)
    x_post_start: int = struct.field(pytree_node=False, default=19)
    x_post_end: int = struct.field(pytree_node=False, default=23)
    x_doorcol: int = struct.field(pytree_node=False, default=24)
    # static episode / reward params
    view_size: int = struct.field(pytree_node=False, default=5)
    max_steps: int = struct.field(pytree_node=False, default=200)
    step_penalty: float = struct.field(pytree_node=False, default=0.0)
    branch_bonus: float = struct.field(pytree_node=False, default=0.5)
    wrong_branch_penalty: float = struct.field(pytree_node=False, default=0.0)
    success_reward: float = struct.field(pytree_node=False, default=0.5)
    wrong_door_reward: float = struct.field(pytree_node=False, default=0.0)
    shaping_coef: float = struct.field(pytree_node=False, default=0.01)
    # curriculum: prob that the door colours are randomised per episode. 1.0 =
    # fully random (target task, conditional colour->door); 0.0 = fixed (green
    # always top) so colour->door is unconditional, like the branch.
    door_random_prob: float = struct.field(pytree_node=False, default=1.0)

    @classmethod
    def from_config(cls, *, cue_distribution="factorized", custom_cues=None,
                    custom_weights=None, max_steps=200, view_size=5,
                    center_wall_thickness=3, pre_cue_steps=1,
                    pre_branch_corridor_len=5, branch_len=4,
                    post_branch_corridor_len=5, step_penalty=0.0,
                    branch_bonus=0.5, wrong_branch_penalty=0.0, success_reward=0.5,
                    wrong_door_reward=0.0, shaping_coef=0.01,
                    door_random_prob=1.0) -> "EnvParams":
        g = build_geometry(
            pre_cue_steps=pre_cue_steps, pre_branch_corridor_len=pre_branch_corridor_len,
            branch_len=branch_len, post_branch_corridor_len=post_branch_corridor_len,
            view_size=view_size, center_wall_thickness=center_wall_thickness)
        probs = _cue_probs(cue_distribution, custom_cues, custom_weights)
        return cls(
            base_terrain=jnp.asarray(g["base_terrain"], dtype=jnp.int8),
            cue_probs=jnp.asarray(probs, dtype=jnp.float32),
            height=g["height"], width=g["width"], my=g["my"],
            row_up=g["row_up"], row_lo=g["row_lo"],
            row_room_up=g["row_room_up"], row_room_lo=g["row_room_lo"],
            row_door_top=g["row_door_top"], row_door_bot=g["row_door_bot"],
            x_precue_start=g["x_precue_start"], x_room_start=g["x_room_start"],
            x_room_end=g["x_room_end"], x_pre_end=g["x_pre_end"],
            x_branch_start=g["x_branch_start"], x_branch_end=g["x_branch_end"],
            x_post_start=g["x_post_start"], x_post_end=g["x_post_end"],
            x_doorcol=g["x_doorcol"], view_size=view_size, max_steps=max_steps,
            step_penalty=step_penalty, branch_bonus=branch_bonus,
            wrong_branch_penalty=wrong_branch_penalty, door_random_prob=door_random_prob,
            success_reward=success_reward, wrong_door_reward=wrong_door_reward,
            shaping_coef=shaping_coef)


def _cue_probs(cue_distribution, custom_cues, custom_weights) -> np.ndarray:
    """Sampling distribution over CUE_TYPES (mirrors _sample_cue)."""
    idx = {n: i for i, n in enumerate(C.CUE_TYPES)}
    p = np.zeros(4, dtype=np.float64)
    if cue_distribution == "factorized":
        p[:] = 0.25
    elif cue_distribution == "entangled":
        p[idx["green_up"]] = p[idx["blue_down"]] = 0.5
    elif cue_distribution == "custom":
        if not custom_cues:
            raise ValueError("cue_distribution='custom' requires custom_cues")
        w = (np.asarray(custom_weights, float) if custom_weights is not None
             else np.ones(len(custom_cues)))
        w = w / w.sum()
        for name, wi in zip(custom_cues, w):
            if name not in idx:
                raise ValueError(f"unknown cue {name!r}")
            p[idx[name]] += wi
    else:
        raise ValueError(f"bad cue_distribution={cue_distribution!r}")
    return p
