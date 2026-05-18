"""Minimal Flashbax replay sanity checks for the DreamerV3 add/sample contract."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import flashbax as fbx


class _Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    is_first: jnp.ndarray
    is_last: jnp.ndarray
    is_terminal: jnp.ndarray


def test_flashbax_add_pattern_preserves_env_local_sequences():
    num_envs = 2
    batch_size = 3
    seq_len = 4

    buffer = fbx.make_trajectory_buffer(
        add_batch_size=num_envs,
        sample_batch_size=batch_size,
        sample_sequence_length=seq_len,
        period=1,
        min_length_time_axis=seq_len,
        max_length_time_axis=16,
    )

    dummy = _Transition(
        obs=jnp.zeros((1,), dtype=jnp.int32),
        action=jnp.zeros((1,), dtype=jnp.int32),
        reward=jnp.zeros((), dtype=jnp.int32),
        is_first=jnp.zeros((), dtype=bool),
        is_last=jnp.zeros((), dtype=bool),
        is_terminal=jnp.zeros((), dtype=bool),
    )
    state = buffer.init(dummy)

    for t in range(6):
        transition = _Transition(
            obs=jnp.array([[100 * t + e] for e in range(num_envs)], dtype=jnp.int32),
            action=jnp.array([[10 * t + e] for e in range(num_envs)], dtype=jnp.int32),
            reward=jnp.array([1000 * t + e for e in range(num_envs)], dtype=jnp.int32),
            is_first=jnp.array([t == 0] * num_envs),
            is_last=jnp.zeros((num_envs,), dtype=bool),
            is_terminal=jnp.zeros((num_envs,), dtype=bool),
        )
        state = buffer.add(state, jax.tree.map(lambda x: x[:, None, ...], transition))

    stored_obs = np.asarray(state.experience.obs[..., 0])
    np.testing.assert_array_equal(stored_obs[0, :6], np.array([0, 100, 200, 300, 400, 500]))
    np.testing.assert_array_equal(stored_obs[1, :6], np.array([1, 101, 201, 301, 401, 501]))

    sample = buffer.sample(state, jax.random.PRNGKey(0)).experience
    obs = np.asarray(sample.obs[..., 0])
    action = np.asarray(sample.action[..., 0])
    reward = np.asarray(sample.reward)

    # Every sampled row should stay within one env stream and advance by exactly one time step.
    np.testing.assert_array_equal(np.diff(obs, axis=1), np.full((batch_size, seq_len - 1), 100))
    np.testing.assert_array_equal(np.diff(action, axis=1), np.full((batch_size, seq_len - 1), 10))
    np.testing.assert_array_equal(np.diff(reward, axis=1), np.full((batch_size, seq_len - 1), 1000))
    np.testing.assert_array_equal(np.diff(obs % 100, axis=1), np.zeros((batch_size, seq_len - 1)))
