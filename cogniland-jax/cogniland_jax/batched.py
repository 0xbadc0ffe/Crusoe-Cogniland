"""Batched wrapper around ``CognilandEnv`` matching the legacy numpy API.

``cogniland.envs.env.CognilandEnv`` + ``MultiTaskEnvWrapper`` expose:

    env.num_envs
    env.action_space()                  # int
    env.observation_space()             # dict
    env.set_tasks(task_ids)
    env.reset()                         # dict with {minimap, scalars, task_embedding}
    env.step(actions_np)                # (obs, rewards, dones, info)
    env.spawn_distance_schedule         # property (None here)
    env.set_spawn_distance_range(...)   # no-op here

This wrapper reproduces the same surface on top of a pure-JAX Gymnax env,
handling auto-reset + episode bookkeeping internally. The trainer and
PPO-RNN agent need **zero** changes to consume it.

Task handling: task_ids are held on the wrapper (not on the per-env state)
and injected into the ``task_embedding`` obs field on every step. Task 0
is the only live task; the success flag is `reached YES or NO`. Tasks 1-6
are stubs (success=0) — keep the code path identical to the legacy env.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from cogniland_jax import constants as C
from cogniland_jax.env import CognilandEnv
from cogniland_jax.state import EnvParams, EnvState


_BIOME_NAMES = np.array(
    ["balanced", "archipelago", "grassland", "highland"], dtype=object,
)


def _blend_dict(done: jax.Array, step_branch: dict, reset_branch: dict) -> dict:
    """Per-field blend of two dicts of jnp arrays using a [B] bool mask."""
    def _sel(s: jax.Array, r: jax.Array) -> jax.Array:
        shape = (-1,) + (1,) * (s.ndim - 1)
        return jnp.where(done.reshape(shape), r, s)
    return jax.tree.map(_sel, reset_branch, step_branch)


class JaxBatchedEnv:
    """Numpy-API wrapper around a batched+auto-reset Gymnax-style env."""

    def __init__(
        self,
        env: CognilandEnv,
        params: EnvParams,
        num_envs: int,
        seed: int = 0,
        gamma: float = 0.99,
    ):
        self._env = env
        self._params = params
        self._num_envs = int(num_envs)
        self._gamma = float(gamma)
        self._rng = jax.random.PRNGKey(int(seed))

        # task_ids held on the wrapper (not on state) — simpler to update.
        self._task_ids = np.zeros(self._num_envs, dtype=np.int32)

        # Per-env episode accumulators (numpy — keeps agent code untouched).
        self._episode_returns = np.zeros(self._num_envs, dtype=np.float32)
        self._episode_returns_disc = np.zeros(self._num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self._num_envs, dtype=np.int32)
        self._episode_steps = np.zeros(self._num_envs, dtype=np.int32)

        self._state: Optional[EnvState] = None

        # Pre-build jitted vmapped reset/step.
        self._reset_vmap = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
        self._step_vmap = jax.jit(
            jax.vmap(env.step, in_axes=(0, 0, 0, None))
        )

    # ── API properties ────────────────────────────────────────────────

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def spawn_distance_schedule(self) -> None:
        return None

    def set_spawn_distance_range(self, lo: int, hi: int) -> None:
        """No-op — difficulty is a JAX-level `EnvParams` field, not a
        curriculum knob on this wrapper. Kept for trainer compatibility."""
        return

    def action_space(self) -> int:
        return C.NUM_ACTIONS

    def observation_space(self) -> dict:
        return {
            "minimap": (C.MINIMAP_DIAMETER, C.MINIMAP_DIAMETER),
            "scalars": (6,),
        }

    def set_tasks(self, task_ids: np.ndarray) -> None:
        self._task_ids = np.asarray(task_ids, dtype=np.int32).reshape(-1)

    # ── Core lifecycle ────────────────────────────────────────────────

    def _inject_task_ids(self, state):
        """Overwrite state.task_id with the wrapper's task_ids.

        The underlying env's ``reset_env`` samples ``task_id`` randomly to
        keep the pytree shape stable for jit/vmap; the wrapper owns the
        authoritative task assignment (via ``set_tasks``), so we inject
        it after every reset and before every step. ``env.step_env`` reads
        ``state.task_id`` to compute the task-specific reward bonus.
        """
        return state.replace(task_id=jnp.asarray(self._task_ids, dtype=jnp.int32))

    def reset(self) -> dict:
        self._rng, k = jax.random.split(self._rng)
        keys = jax.random.split(k, self._num_envs)
        obs, self._state = self._reset_vmap(keys, self._params)
        self._state = self._inject_task_ids(self._state)
        self._episode_returns[:] = 0.0
        self._episode_returns_disc[:] = 0.0
        self._episode_lengths[:] = 0
        self._episode_steps[:] = 0
        return self._obs_to_numpy(obs)

    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, dict[str, Any]]:
        actions_jax = jnp.asarray(actions, dtype=jnp.int32)
        self._rng, k_step, k_reset = jax.random.split(self._rng, 3)
        step_keys = jax.random.split(k_step, self._num_envs)
        reset_keys = jax.random.split(k_reset, self._num_envs)

        # The wrapper, not the env, owns task assignment — inject before
        # stepping so reward/info use the right task id.
        self._state = self._inject_task_ids(self._state)
        obs_step, state_step, rewards, dones, info = self._step_vmap(
            step_keys, self._state, actions_jax, self._params,
        )
        obs_reset, state_reset = self._reset_vmap(reset_keys, self._params)
        self._state = _blend_dict(dones, state_step, state_reset)
        # Re-inject post-blend so auto-reset branches keep the right task.
        self._state = self._inject_task_ids(self._state)
        obs = _blend_dict(dones, obs_step, obs_reset)

        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=bool)

        # Episode accumulation.
        gamma_t = np.power(self._gamma, self._episode_steps.astype(np.float32),
                           dtype=np.float32)
        self._episode_returns_disc += gamma_t * rewards_np
        self._episode_returns += rewards_np
        self._episode_lengths += 1
        self._episode_steps += 1

        returned_returns = np.where(dones_np, self._episode_returns, 0.0).astype(np.float32)
        returned_returns_d = np.where(dones_np, self._episode_returns_disc, 0.0).astype(np.float32)
        returned_lengths = np.where(dones_np, self._episode_lengths, 0).astype(np.int32)

        reached_yes = np.asarray(info["reached_yes"], dtype=bool)
        reached_no = np.asarray(info["reached_no"], dtype=bool)
        # Per-task success comes from the env — wrapper just casts for the
        # existing agent/trainer numpy-API contract.
        task_success = np.asarray(info["task_success"], dtype=np.float32)

        biome_ids = np.asarray(info["biome_id"], dtype=np.int32)
        biomes = _BIOME_NAMES[np.clip(biome_ids, 0, len(_BIOME_NAMES) - 1)]

        info_np: dict[str, Any] = {
            "returned_episode": dones_np,
            "returned_episode_returns": returned_returns,
            "returned_episode_returns_discounted": returned_returns_d,
            "returned_episode_lengths": returned_lengths,
            "returned_episode_berry_forages": np.zeros(self._num_envs, dtype=np.int32),
            "task_success": task_success,
            "task_rewards": rewards_np,
            "reached": (reached_yes | reached_no),
            "reached_yes": reached_yes,
            "reached_no": reached_no,
            "alive": ~np.asarray(info["died"], dtype=bool),
            "crafted": np.asarray(info["crafted"], dtype=np.int32),
            "biome": biomes,
            "hp_prev": np.asarray(info["hp_prev"], dtype=np.float32),
            "hp_curr": np.asarray(info["hp_curr"], dtype=np.float32),
            "ctg_prev": np.asarray(info["ctg_prev"], dtype=np.float32),
            "ctg_curr": np.asarray(info["ctg_curr"], dtype=np.float32),
        }

        # Reset per-env accumulators after reporting.
        if dones_np.any():
            self._episode_returns[dones_np] = 0.0
            self._episode_returns_disc[dones_np] = 0.0
            self._episode_lengths[dones_np] = 0
            self._episode_steps[dones_np] = 0

        return self._obs_to_numpy(obs), rewards_np, dones_np, info_np

    # ── Helpers ───────────────────────────────────────────────────────

    def _obs_to_numpy(self, obs: dict[str, jax.Array]) -> dict[str, np.ndarray]:
        # Keep minimap on-device (JAX array) — the PPO-RNN agent calls
        # ``jnp.asarray`` on it, which is a no-op when already on-device.
        # Scalars we copy to host; task_embedding we rebuild from the
        # wrapper's own task_ids so set_tasks actually takes effect.
        scalars = np.asarray(obs["scalars"])
        te = np.eye(C.TASK_EMBEDDING_DIM, dtype=np.float32)[self._task_ids]
        return {
            "minimap": obs["minimap"],
            "scalars": scalars,
            "task_embedding": te,
        }
