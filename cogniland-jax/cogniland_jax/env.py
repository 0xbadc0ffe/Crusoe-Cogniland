"""Cogniland as a Gymnax-style pure-JAX Environment.

Obs (dict pytree per step):
    "minimap":        int8   [45, 45]     tile-class ids (see constants.TILE_*)
    "scalars":        float32 [6]         [compass_x, compass_y, tile_cls/9,
                                          hp/hp_max, wood/wood_max, tool/3]
    "task_embedding": float32 [7]         one-hot task id

Action: ``Discrete(8)`` — 0..3 cardinal moves, 4 forage, 5..7 craft
raft/rope/shoes.

Reward:
    r = -step_penalty
      + reach_bonus · [reached YES or NO]
      + shaping_coef · (ctg_prev - ctg_curr)   # Dijkstra cost-to-go on drain graph
      + hp_coef     · (hp_curr - hp_prev)      # HP delta (healing/drain)
      - death_penalty · [died]

Termination: HP≤0, deadly-tile step, reach YES or NO target, or
step_count ≥ params.max_steps (truncation). No auto-reset inside
``step_env`` — training code wraps with an auto-reset helper (Craftax
semantics).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from cogniland_jax import constants as C
from cogniland_jax.dynamics import (
    compute_ctg,
    env_step_core,
    sample_map_and_spawn_target,
)
from cogniland_jax.render import build_obs
from cogniland_jax.state import EnvParams, EnvState


class CognilandEnv(environment.Environment[EnvState, EnvParams]):
    """Gymnax-compatible Cogniland environment."""

    def __init__(self, default_params: EnvParams | None = None):
        super().__init__()
        self._default_params = default_params

    @property
    def default_params(self) -> EnvParams:
        if self._default_params is None:
            raise RuntimeError(
                "CognilandEnv needs an EnvParams with map arrays. Use "
                "`cogniland_jax.maps.load_map_arrays(...)` + "
                "`EnvParams.from_map_arrays(...)` then pass to the env."
            )
        return self._default_params

    @property
    def name(self) -> str:
        return "Cogniland-v0"

    @property
    def num_actions(self) -> int:
        return C.NUM_ACTIONS

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(C.NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict({
            "minimap": spaces.Box(
                low=0, high=C.NUM_TILE_CLASSES - 1,
                shape=(C.MINIMAP_DIAMETER, C.MINIMAP_DIAMETER),
                dtype=jnp.int8,
            ),
            "scalars": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=jnp.float32),
            "task_embedding": spaces.Box(
                low=0.0, high=1.0, shape=(C.TASK_EMBEDDING_DIM,), dtype=jnp.float32,
            ),
        })

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict({})  # internal use only

    # ── Lifecycle ────────────────────────────────────────────────────

    def reset_env(self, key: jax.Array, params: EnvParams) -> tuple[dict, EnvState]:
        key, k_search, k_task = jax.random.split(key, 3)
        max_euclid = C.MAX_EUCLID_BY_DIFFICULTY[jnp.clip(params.difficulty, 0, 2)]
        map_idx, sr, sc, yr, yc, nr, nc = sample_map_and_spawn_target(
            k_search, params.terrain_idx, max_euclid,
        )
        terrain_idx_map = params.terrain_idx[map_idx]
        berry_mask_map = params.berry_mask[map_idx]
        mid_r = yr
        mid_c = yc + C.TARGET_GAP // 2
        ctg = compute_ctg(terrain_idx_map, berry_mask_map, mid_r, mid_c)

        state = EnvState(
            pos_r=sr, pos_c=sc,
            hp=C.INIT_HP, wood=jnp.int32(0), tool=jnp.int32(C.TOOL_NONE),
            consec_grass=jnp.int32(0), steps=jnp.int32(0),
            map_idx=map_idx,
            spawn_r=sr, spawn_c=sc,
            yes_r=yr, yes_c=yc,
            no_r=nr, no_c=nc,
            mid_r=mid_r, mid_c=mid_c,
            ctg=ctg, ctg_spawn=ctg[sr, sc],
            task_id=jax.random.randint(k_task, (), 0, C.TASK_EMBEDDING_DIM),
            terminated=jnp.bool_(False),
            last_action=jnp.int32(-1),
            crafted_this_step=jnp.int32(0),
        )
        return build_obs(state, params), state

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[dict, EnvState, jax.Array, jax.Array, dict]:
        action = jnp.asarray(action, dtype=jnp.int32)
        new_state, reward, done, info = env_step_core(state, action, params)
        obs = build_obs(new_state, params)
        return obs, new_state, reward, done, info

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams | None = None,
    ) -> tuple[dict, EnvState, jax.Array, jax.Array, dict]:
        """Pure step — no auto-reset on done (Craftax semantics).

        The gymnax base ``Environment.step`` auto-resets via
        ``jax.lax.select`` on obs/state. ``lax.select`` can't broadcast
        over our dict obs, so we override to return the raw terminal
        transition. Training code is expected to wrap with an
        AutoResetEnvWrapper that uses ``jax.tree.map`` + ``lax.select``.
        """
        if params is None:
            params = self.default_params
        return self.step_env(key, state, action, params)

    def reset(
        self,
        key: jax.Array,
        params: EnvParams | None = None,
    ) -> tuple[dict, EnvState]:
        if params is None:
            params = self.default_params
        return self.reset_env(key, params)
