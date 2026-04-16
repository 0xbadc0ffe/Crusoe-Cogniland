"""
R2D2-specific replay buffers with LSTM hidden state storage.

Extends the base sequence buffers to store (h, c) hidden states per transition.
"""
from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .sequence import ReplayBuffer, ReservoirReplayBuffer, ReplayBufferState


class R2D2ReplayBuffer(ReplayBuffer):
    """
    Replay buffer for R2D2 with LSTM hidden state storage.

    Extends ReplayBuffer to store (h, c) hidden states per transition.
    When sampling sequences, returns the hidden state at sequence START
    for burn-in reconstruction during training.
    """

    def __init__(
        self,
        capacity: int = 1000000,
        obs_shapes: Dict[str, Tuple[int, ...]] = None,
        action_dim: int = 4,
        batch_size: int = 16,
        batch_length: int = 64,
        seed: int = 0,
        num_envs: int = 1,
        lstm_size: int = 512,
    ):
        """
        Initialize R2D2 replay buffer.

        Args:
            capacity: Maximum buffer size (total transitions)
            obs_shapes: Dictionary of observation shapes
            action_dim: Action dimension (1 for discrete actions)
            batch_size: Batch size for sampling
            batch_length: Length of sequences to sample
            seed: Random seed
            num_envs: Number of parallel environments
            lstm_size: Size of LSTM hidden state
        """
        super().__init__(
            capacity=capacity,
            obs_shapes=obs_shapes,
            action_dim=action_dim,
            batch_size=batch_size,
            batch_length=batch_length,
            seed=seed,
            num_envs=num_envs,
        )
        self.lstm_size = lstm_size

    def init(self, rng: Optional[jax.random.PRNGKey] = None) -> ReplayBufferState:
        """Initialize buffer state with hidden state arrays."""
        state = super().init(rng)

        # Add hidden state arrays
        state.data['hidden_h'] = np.zeros((self.capacity, self.lstm_size), dtype=np.float32)
        state.data['hidden_c'] = np.zeros((self.capacity, self.lstm_size), dtype=np.float32)

        return state

    def add_batch(
        self,
        state: ReplayBufferState,
        transitions: Dict[str, jnp.ndarray],
    ) -> ReplayBufferState:
        """
        Add a batch of transitions including hidden states.

        Args:
            state: Current buffer state
            transitions: Dictionary with arrays including 'hidden_h' and 'hidden_c'

        Returns:
            Updated buffer state
        """
        # Get batch size and compute indices (same as parent)
        batch_size = int(next(iter(transitions.values())).shape[0])
        assert batch_size == self.num_envs, f"Batch size {batch_size} must match num_envs {self.num_envs}"

        time_idx = int(state.write_idx)
        start_idx = (time_idx * self.num_envs) % self.capacity
        indices = np.arange(start_idx, start_idx + self.num_envs) % self.capacity

        # Store hidden states
        if 'hidden_h' in transitions:
            hidden_h = transitions['hidden_h']
            if isinstance(hidden_h, jnp.ndarray):
                hidden_h = np.array(hidden_h)
            state.data['hidden_h'][indices] = hidden_h

        if 'hidden_c' in transitions:
            hidden_c = transitions['hidden_c']
            if isinstance(hidden_c, jnp.ndarray):
                hidden_c = np.array(hidden_c)
            state.data['hidden_c'][indices] = hidden_c

        # Call parent to store other data and update indices
        return super().add_batch(state, transitions)

    def sample(
        self,
        state: ReplayBufferState,
        rng: jax.random.PRNGKey,
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample sequences with hidden states at sequence START.

        Returns batch with:
            - obs_*: (batch_size, batch_length, *obs_shape)
            - action: (batch_size, batch_length, action_dim)
            - reward: (batch_size, batch_length)
            - is_first, is_last, is_terminal: (batch_size, batch_length)
            - hidden_h: (batch_size, lstm_size) - h at sequence START
            - hidden_c: (batch_size, lstm_size) - c at sequence START
        """
        # Split RNG for env and time sampling
        rng_env, rng_time = jax.random.split(rng)

        valid_timesteps = int(state.valid_size) // self.num_envs

        if valid_timesteps < self.batch_length:
            raise ValueError(
                f"Not enough timesteps for sampling: have {valid_timesteps}, "
                f"need {self.batch_length}."
            )

        # Sample (env, start_time) pairs
        env_indices = jax.random.randint(rng_env, (self.batch_size,), 0, self.num_envs)
        max_start_time = valid_timesteps - self.batch_length
        start_times = jax.random.randint(rng_time, (self.batch_size,), 0, max_start_time + 1)

        # Compute sequence indices
        env_np = np.array(env_indices)
        start_np = np.array(start_times)
        time_offsets = np.arange(self.batch_length)

        time_idx = start_np[:, None] + time_offsets[None, :]
        seq_idxs = (time_idx * self.num_envs + env_np[:, None]) % self.capacity

        # Gather sequences
        batch = {}

        for key in self.obs_shapes:
            obs_key = f'obs_{key}'
            obs_data = state.data[obs_key][seq_idxs]
            obs_data = jnp.array(obs_data)
            # Keep original dtype - preprocessing functions handle type conversion and normalization
            batch[obs_key] = obs_data

        batch['action'] = jnp.array(state.data['action'][seq_idxs])
        batch['reward'] = jnp.array(state.data['reward'][seq_idxs])
        batch['is_first'] = jnp.array(state.data['is_first'][seq_idxs])
        batch['is_last'] = jnp.array(state.data['is_last'][seq_idxs])
        batch['is_terminal'] = jnp.array(state.data['is_terminal'][seq_idxs])

        # Get hidden states at sequence START (first timestep of each sequence)
        start_idxs = seq_idxs[:, 0]  # (batch_size,)
        batch['hidden_h'] = jnp.array(state.data['hidden_h'][start_idxs])
        batch['hidden_c'] = jnp.array(state.data['hidden_c'][start_idxs])

        return batch


class R2D2ReservoirReplayBuffer(ReservoirReplayBuffer):
    """
    R2D2 replay buffer with reservoir sampling for continual learning.

    Extends ReservoirReplayBuffer with LSTM hidden state storage.
    Performs reservoir sampling at the sequence level to maintain temporal coherence.
    """

    def __init__(
        self,
        capacity: int = 1000000,
        obs_shapes: Dict[str, Tuple[int, ...]] = None,
        action_dim: int = 4,
        batch_size: int = 16,
        batch_length: int = 64,
        seed: int = 0,
        num_envs: int = 1,
        lstm_size: int = 512,
    ):
        """
        Initialize R2D2 reservoir replay buffer.

        Args:
            capacity: Maximum buffer size (total transitions)
            obs_shapes: Dictionary of observation shapes
            action_dim: Action dimension
            batch_size: Batch size for sampling
            batch_length: Length of sequences to sample
            seed: Random seed
            num_envs: Number of parallel environments
            lstm_size: Size of LSTM hidden state
        """
        super().__init__(
            capacity=capacity,
            obs_shapes=obs_shapes,
            action_dim=action_dim,
            batch_size=batch_size,
            batch_length=batch_length,
            seed=seed,
            num_envs=num_envs,
        )
        self.lstm_size = lstm_size

    def init(self, rng: Optional[jax.random.PRNGKey] = None) -> ReplayBufferState:
        """Initialize buffer state with hidden state arrays."""
        state = super().init(rng)

        # Add hidden state arrays
        state.data['hidden_h'] = np.zeros((self.capacity, self.lstm_size), dtype=np.float32)
        state.data['hidden_c'] = np.zeros((self.capacity, self.lstm_size), dtype=np.float32)

        return state

    def add_batch(
        self,
        state: ReplayBufferState,
        transitions: Dict[str, jnp.ndarray],
    ) -> ReplayBufferState:
        """
        Add a batch of transitions including hidden states.

        Uses sequence-level reservoir sampling from parent class.
        """
        batch_size = int(next(iter(transitions.values())).shape[0])
        assert batch_size == self.num_envs

        curr_valid = int(state.valid_size)

        # If buffer not full, use FIFO - compute indices for hidden state storage
        if curr_valid < self.capacity:
            time_idx = int(state.write_idx)
            start_idx = (time_idx * self.num_envs) % self.capacity
            indices = np.arange(start_idx, start_idx + self.num_envs) % self.capacity

            # Store hidden states
            if 'hidden_h' in transitions:
                hidden_h = transitions['hidden_h']
                if isinstance(hidden_h, jnp.ndarray):
                    hidden_h = np.array(hidden_h)
                state.data['hidden_h'][indices] = hidden_h

            if 'hidden_c' in transitions:
                hidden_c = transitions['hidden_c']
                if isinstance(hidden_c, jnp.ndarray):
                    hidden_c = np.array(hidden_c)
                state.data['hidden_c'][indices] = hidden_c

            return super().add_batch(state, transitions)

        # Buffer is full - reservoir sampling mode
        # IMPORTANT: Call parent FIRST to get correct seq_target (decided at seq_step=0)
        # Then write hidden states to the correct slot
        original_seq_step = int(state.seq_step)
        new_state = super().add_batch(state, transitions)

        # Now get the correct seq_target from the updated state
        seq_target = int(new_state.seq_target)

        # Store hidden states if we're keeping this sequence
        if seq_target >= 0:
            target_time_start = seq_target * self.batch_length
            time_idx = target_time_start + original_seq_step
            start_idx = (time_idx * self.num_envs) % self.capacity
            indices = np.arange(start_idx, start_idx + self.num_envs) % self.capacity

            if 'hidden_h' in transitions:
                hidden_h = transitions['hidden_h']
                if isinstance(hidden_h, jnp.ndarray):
                    hidden_h = np.array(hidden_h)
                new_state.data['hidden_h'][indices] = hidden_h

            if 'hidden_c' in transitions:
                hidden_c = transitions['hidden_c']
                if isinstance(hidden_c, jnp.ndarray):
                    hidden_c = np.array(hidden_c)
                new_state.data['hidden_c'][indices] = hidden_c

        return new_state

    def sample(
        self,
        state: ReplayBufferState,
        rng: jax.random.PRNGKey,
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample sequences with hidden states at sequence START.

        Respects sequence-slot boundaries for reservoir sampling.
        """
        rng_env, rng_seq = jax.random.split(rng)

        valid_timesteps = int(state.valid_size) // self.num_envs

        if valid_timesteps < self.batch_length:
            raise ValueError(
                f"Not enough timesteps for sampling: have {valid_timesteps}, "
                f"need {self.batch_length}."
            )

        # Sample environment indices
        env_indices = jax.random.randint(rng_env, (self.batch_size,), 0, self.num_envs)

        # Sample sequence slots (aligned to batch_length boundaries)
        num_available_slots = min(valid_timesteps // self.batch_length, self.num_sequences)

        # Protect active slot if incomplete
        active_slot = int(state.seq_target)
        active_incomplete = (int(state.seq_step) != 0 and active_slot >= 0)

        seq_slot_indices = jax.random.randint(rng_seq, (self.batch_size,), 0, num_available_slots)

        if active_incomplete:
            seq_slot_np = np.array(seq_slot_indices)
            mask = (seq_slot_np == active_slot)
            if np.any(mask):
                seq_slot_np[mask] = (active_slot + 1) % num_available_slots
                seq_slot_indices = jnp.array(seq_slot_np)

        # Compute indices
        start_times = np.array(seq_slot_indices) * self.batch_length
        env_np = np.array(env_indices)
        time_offsets = np.arange(self.batch_length)

        time_idx = start_times[:, None] + time_offsets[None, :]
        seq_idxs = (time_idx * self.num_envs + env_np[:, None]) % self.capacity

        # Gather sequences
        batch = {}

        for key in self.obs_shapes:
            obs_key = f'obs_{key}'
            obs_data = state.data[obs_key][seq_idxs]
            obs_data = jnp.array(obs_data)
            # Keep original dtype - preprocessing functions handle type conversion and normalization
            batch[obs_key] = obs_data

        batch['action'] = jnp.array(state.data['action'][seq_idxs])
        batch['reward'] = jnp.array(state.data['reward'][seq_idxs])
        batch['is_first'] = jnp.array(state.data['is_first'][seq_idxs])
        batch['is_last'] = jnp.array(state.data['is_last'][seq_idxs])
        batch['is_terminal'] = jnp.array(state.data['is_terminal'][seq_idxs])

        # Get hidden states at sequence START
        start_idxs = seq_idxs[:, 0]
        batch['hidden_h'] = jnp.array(state.data['hidden_h'][start_idxs])
        batch['hidden_c'] = jnp.array(state.data['hidden_c'][start_idxs])

        return batch
