"""
TODO: To use dreamer and storm we need to convert the environment to jax - we won't need this file anymore

Adapter for reference agents (DreamerV3, STORM) that expect a specific API.

Wraps MultiTaskEnvWrapper and presents a state-based interface where step()
returns a state object compatible with the reference agent training loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cogniland.envs.multitask_wrapper import MultiTaskEnvWrapper


@dataclass
class EnvObservation:
    """Container for environment observations, accessed as dict-like."""
    _data: dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def __contains__(self, key: str) -> bool:
        return key in self._data


@dataclass
class EnvState:
    """State of the environment at a single timestep.

    Compatible with the reference agents' expectations:
        env_state.observation -> dict-like observations
        env_state.reward -> float array [B]
        env_state.is_done() -> bool array [B]
        env_state.is_termination() -> bool array [B]
        env_state.t -> int array [B] (step counter within episode)
    """
    observation: EnvObservation
    reward: np.ndarray  # [B]
    _done: np.ndarray   # [B]
    t: np.ndarray       # [B] step counter within episode

    def is_done(self) -> np.ndarray:
        return self._done

    def is_termination(self) -> np.ndarray:
        # For now, all dones are terminations (no truncation distinction)
        return self._done


@dataclass
class GymAdapterState:
    """Full state returned by GymAdapter.step() and .reset().

    Contains env_state plus episode tracking info.
    """
    env_state: EnvState
    returned_episode_returns: np.ndarray  # [B]
    returned_episode_lengths: np.ndarray  # [B]
    returned_episode: np.ndarray          # [B] bool
    task_success: np.ndarray              # [B] float32, task-aware success flag
    timestep: np.ndarray                  # [B] global timestep counter


class GymAdapter:
    """Adapter wrapping MultiTaskEnvWrapper for reference agent compatibility.

    The reference agents (DreamerV3, STORM) expect:
        state = env.reset(rngs)
        state = env.step(state, actions)
        state.env_state.observation[modality]
        state.env_state.reward
        state.env_state.is_done()
        state.returned_episode_returns
    """

    def __init__(self, env: MultiTaskEnvWrapper):
        self._env = env
        self._episode_returns = np.zeros(env.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(env.num_envs, dtype=np.int32)
        self._timestep = np.zeros(env.num_envs, dtype=np.int32)
        self._step_counter = np.zeros(env.num_envs, dtype=np.int32)

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    def observation_space(self) -> dict:
        return self._env.observation_space()

    def action_space(self) -> int:
        return self._env.action_space()

    def reset(self, rngs: Any = None) -> GymAdapterState:
        """Reset all envs. rngs is accepted for API compat but ignored."""
        obs = self._env.reset()
        B = self._env.num_envs
        self._episode_returns = np.zeros(B, dtype=np.float32)
        self._episode_lengths = np.zeros(B, dtype=np.int32)
        self._step_counter = np.zeros(B, dtype=np.int32)

        env_state = EnvState(
            observation=EnvObservation(_data=obs),
            reward=np.zeros(B, dtype=np.float32),
            _done=np.zeros(B, dtype=bool),
            t=np.zeros(B, dtype=np.int32),
        )
        return GymAdapterState(
            env_state=env_state,
            returned_episode_returns=np.zeros(B, dtype=np.float32),
            returned_episode_lengths=np.zeros(B, dtype=np.int32),
            returned_episode=np.zeros(B, dtype=bool),
            task_success=np.zeros(B, dtype=np.float32),
            timestep=self._timestep.copy(),
        )

    def step(self, state: GymAdapterState, actions: np.ndarray) -> GymAdapterState:
        """Step all envs.

        Args:
            state: previous GymAdapterState (used for API compat)
            actions: int array [B]

        Returns:
            new GymAdapterState
        """
        obs, rewards, dones, info = self._env.step(actions)

        self._episode_returns += rewards
        self._episode_lengths += 1
        self._timestep += 1
        self._step_counter += 1

        # Capture returns for done envs before reset
        returned_episode = info["returned_episode"]
        returned_returns = np.where(
            returned_episode, self._episode_returns, 0.0
        )
        returned_lengths = np.where(
            returned_episode, self._episode_lengths, 0
        )
        # Task-aware per-episode success flag from the wrapper. Only meaningful
        # on steps where ``returned_episode`` is True.
        task_success_raw = info.get(
            "task_success", np.zeros_like(rewards, dtype=np.float32)
        )
        task_success = np.where(returned_episode, task_success_raw, 0.0).astype(
            np.float32
        )

        # Reset tracking for done envs (env auto-resets internally)
        if returned_episode.any():
            self._episode_returns[returned_episode] = 0.0
            self._episode_lengths[returned_episode] = 0
            self._step_counter[returned_episode] = 0

        env_state = EnvState(
            observation=EnvObservation(_data=obs),
            reward=rewards,
            _done=dones,
            t=self._step_counter.copy(),
        )
        return GymAdapterState(
            env_state=env_state,
            returned_episode_returns=returned_returns,
            returned_episode_lengths=returned_lengths,
            returned_episode=returned_episode,
            task_success=task_success,
            timestep=self._timestep.copy(),
        )
