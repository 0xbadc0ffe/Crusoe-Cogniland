"""
Fixed Replay buffer that maintains temporal coherence with parallel environments.
"""
from typing import Dict, Tuple, Optional, Any

import jax
import jax.numpy as jnp
import numpy as np
import chex


@chex.dataclass
class ReplayBufferState:
    """State for the replay buffer."""

    # Data arrays - store transitions in circular buffer
    data: Dict[str, np.ndarray]  # All transition data

    # Buffer management
    write_idx: jnp.ndarray      # Current write position (in time steps, not transitions)
    valid_size: jnp.ndarray     # Number of valid entries
    total_steps: jnp.ndarray    # Total steps added (for stats)
    rng: jax.random.PRNGKey     # RNG key

    # Sequence-level reservoir sampling state (for ReservoirReplayBuffer)
    seq_step: jnp.ndarray       # Current step within sequence (0 to batch_length-1)
    seq_target: jnp.ndarray     # Target sequence slot (-1 = skip)


class ReplayBuffer:
    """
    FIFO Replay buffer for DreamerV3 with proper temporal coherence for parallel environments.

    The key fix: When sampling sequences, we stride by num_envs to ensure each sequence
    comes from a single environment's trajectory.

    Storage pattern remains: [env0_t0, env1_t0, ..., envN_t0, env0_t1, env1_t1, ..., envN_t1, ...]
    Sampling pattern: For env e, sample indices [t*N+e, (t+1)*N+e, (t+2)*N+e, ...] where N=num_envs
    """

    def __init__(
        self,
        capacity: int = 1000000,
        obs_shapes: Dict[str, Tuple[int, ...]] = None,
        action_dim: int = 4,
        batch_size: int = 16,
        batch_length: int = 64,
        seed: int = 0,
        num_envs: int = 1,  # Number of parallel environments - CRITICAL for correct sampling
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size (total transitions)
            obs_shapes: Dictionary of observation shapes
            action_dim: Action dimension
            batch_size: Batch size for sampling
            batch_length: Length of sequences to sample
            seed: Random seed
            num_envs: Number of parallel environments (MUST match the actual number used in training)
        """
        self.capacity = capacity
        self.obs_shapes = obs_shapes or {'image': (64, 64, 3)}
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.seed = seed
        self.num_envs = num_envs

        # Capacity should be divisible by num_envs for clean indexing
        if capacity % num_envs != 0:
            self.capacity = (capacity // num_envs) * num_envs
            print(f"Warning: Adjusted capacity to {self.capacity} to be divisible by num_envs={num_envs}")

        # Time capacity is how many timesteps we can store (each timestep has num_envs transitions)
        self.time_capacity = self.capacity // num_envs

        # Determine dtypes for each observation
        self.obs_dtypes = {}
        for key, shape in self.obs_shapes.items():
            # Use uint8 for images to save memory
            if len(shape) == 3 and shape[-1] in [1, 3, 4]:  # Image channels
                self.obs_dtypes[key] = jnp.uint8
            else:
                self.obs_dtypes[key] = jnp.float32

    def init(self, rng: Optional[jax.random.PRNGKey] = None) -> ReplayBufferState:
        """
        Initialize empty buffer state.

        Args:
            rng: Random key (optional, uses seed if not provided)

        Returns:
            Initial buffer state
        """
        if rng is None:
            rng = jax.random.PRNGKey(self.seed)

        data = {}

        # Use numpy arrays for CPU storage (mutable, support in-place updates)
        # JAX arrays are immutable and .at[].set() would copy the entire buffer!

        # Observations
        for key, shape in self.obs_shapes.items():
            dtype = self.obs_dtypes[key]
            # Use numpy for mutable in-place updates
            np_dtype = np.uint8 if dtype == jnp.uint8 else np.float32
            data[f'obs_{key}'] = np.zeros(
                (self.capacity,) + shape, dtype=np_dtype
            )

        # Actions, rewards, and flags (numpy arrays)
        data['action'] = np.zeros(
            (self.capacity, self.action_dim), dtype=np.float32
        )
        data['reward'] = np.zeros(self.capacity, dtype=np.float32)
        data['is_first'] = np.zeros(self.capacity, dtype=bool)
        data['is_last'] = np.zeros(self.capacity, dtype=bool)
        data['is_terminal'] = np.zeros(self.capacity, dtype=bool)
        data['task_id'] = np.full(self.capacity, -1, dtype=np.int32)

        return ReplayBufferState(
            data=data,
            write_idx=jnp.array(0, dtype=jnp.int32),  # This tracks TIME steps, not individual transitions
            valid_size=jnp.array(0, dtype=jnp.int32),
            total_steps=jnp.array(0, dtype=jnp.int32),
            rng=rng,
            seq_step=jnp.array(0, dtype=jnp.int32),
            seq_target=jnp.array(-1, dtype=jnp.int32),
        )

    def add_batch(
        self,
        state: ReplayBufferState,
        transitions: Dict[str, jnp.ndarray],
    ) -> ReplayBufferState:
        """
        Add a batch of transitions from parallel environments.

        This stores transitions as [env0_t, env1_t, ..., envN_t] at consecutive indices.
        This is the natural pattern from vectorized environments and is efficient for storage.

        Args:
            state: Current buffer state
            transitions: Dictionary with arrays of shape (num_envs, ...)

        Returns:
            Updated buffer state
        """
        # Get batch size from first array
        batch_size = int(next(iter(transitions.values())).shape[0])
        assert batch_size == self.num_envs, f"Batch size {batch_size} must match num_envs {self.num_envs}"

        # Cast JAX scalars to Python ints for buffer bookkeeping
        time_idx = int(state.write_idx)
        curr_valid = int(state.valid_size)
        curr_total = int(state.total_steps)

        # Calculate linear indices for all environments at current timestep
        start_idx = (time_idx * self.num_envs) % self.capacity
        indices = np.arange(start_idx, start_idx + self.num_envs) % self.capacity

        # Vectorized storage - all environments' transitions at once
        for key in self.obs_shapes:
            obs_key = f'obs_{key}'
            if obs_key in transitions:
                obs_value = transitions[obs_key]
                if isinstance(obs_value, jnp.ndarray):
                    obs_value = np.array(obs_value)
                state.data[obs_key][indices] = obs_value

        # Store other data
        if 'action' in transitions:
            action_value = transitions['action']
            if isinstance(action_value, jnp.ndarray):
                action_value = np.array(action_value)
            state.data['action'][indices] = action_value

        if 'reward' in transitions:
            reward_value = transitions['reward']
            if isinstance(reward_value, jnp.ndarray):
                reward_value = np.array(reward_value)
            state.data['reward'][indices] = reward_value

        for key in ['is_first', 'is_last', 'is_terminal']:
            if key in transitions:
                value = transitions[key]
                if isinstance(value, jnp.ndarray):
                    value = np.array(value)
                state.data[key][indices] = value

        if 'task_id' in transitions:
            task_id_value = transitions['task_id']
            if isinstance(task_id_value, jnp.ndarray):
                task_id_value = np.array(task_id_value)
            state.data['task_id'][indices] = task_id_value

        # Update indices - write_idx advances by 1 TIME STEP (not by num_envs)
        new_write_idx = (time_idx + 1) % self.time_capacity
        new_valid_size = min(curr_valid + self.num_envs, self.capacity)
        new_total_steps = curr_total + self.num_envs

        # Split RNG for next operation
        new_rng, _ = jax.random.split(state.rng)

        return state.replace(
            write_idx=jnp.array(new_write_idx, dtype=jnp.int32),
            valid_size=jnp.array(new_valid_size, dtype=jnp.int32),
            total_steps=jnp.array(new_total_steps, dtype=jnp.int32),
            rng=new_rng,
        )

    def sample(
        self,
        state: ReplayBufferState,
        rng: jax.random.PRNGKey,
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample a batch of sequences from the buffer.

        Note: We sample with stride=num_envs to ensure each sequence
        comes from a single environment's trajectory.

        Returns sequences in shape (batch_size, batch_length, ...).

        Args:
            state: Current buffer state
            rng: Random key for sampling

        Returns:
            Dictionary with sampled sequences
        """
        # Split RNG for env and time sampling
        rng_env, rng_time = jax.random.split(rng)

        # Calculate how many complete timesteps we have (cast to Python int)
        valid_timesteps = int(state.valid_size) // self.num_envs

        # We need at least batch_length timesteps to sample a sequence
        if valid_timesteps < self.batch_length:
            # Not enough data yet - raise error instead of silently returning zeros
            raise ValueError(
                f"Not enough timesteps for sampling: have {valid_timesteps}, "
                f"need {self.batch_length}. Consider increasing pretrain steps."
            )

        # Sample random (env, start_time) pairs
        # Each defines a sequence from env at times [start_time, start_time+1, ..., start_time+batch_length-1]

        # Sample environment indices
        env_indices = jax.random.randint(
            rng_env,
            (self.batch_size,),
            minval=0,
            maxval=self.num_envs,
        )

        # Sample starting timesteps
        max_start_time = valid_timesteps - self.batch_length
        start_times = jax.random.randint(
            rng_time,
            (self.batch_size,),
            minval=0,
            maxval=max_start_time + 1,
        )

        # Fully vectorized index computation (no Python loop!)
        # Convert (env, time) pairs to linear indices with proper stride
        env_np = np.array(env_indices)    # (B,)
        start_np = np.array(start_times)  # (B,)
        time_offsets = np.arange(self.batch_length)  # (T,)

        # (B, T) time indices
        time_idx = start_np[:, None] + time_offsets[None, :]

        # (B, T) linear indices
        # Each env's trajectory is separated by stride=num_envs
        seq_idxs = (time_idx * self.num_envs + env_np[:, None]) % self.capacity

        # Gather sequences from numpy arrays
        batch = {}

        # Gather observations (convert numpy to JAX)
        for key in self.obs_shapes:
            obs_key = f'obs_{key}'
            obs_data = state.data[obs_key][seq_idxs]  # numpy array

            # Convert to JAX array
            obs_data = jnp.array(obs_data)

            # Keep original dtype - preprocessing functions handle type conversion and normalization
            batch[obs_key] = obs_data

        # Gather other data (convert numpy to JAX)
        batch['action'] = jnp.array(state.data['action'][seq_idxs])
        batch['reward'] = jnp.array(state.data['reward'][seq_idxs])
        batch['is_first'] = jnp.array(state.data['is_first'][seq_idxs])
        batch['is_last'] = jnp.array(state.data['is_last'][seq_idxs])
        batch['is_terminal'] = jnp.array(state.data['is_terminal'][seq_idxs])

        return batch

    def sample_flat(
        self,
        state: ReplayBufferState,
        rng: jax.random.PRNGKey,
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample sequences and return in flattened format.

        This is useful for models that process flattened batches.
        Returns data in shape (batch_size * batch_length, ...).

        Args:
            state: Current buffer state
            rng: Random key for sampling

        Returns:
            Dictionary with flattened data
        """
        # Sample sequences
        batch = self.sample(state, rng)

        # Flatten batch and time dimensions
        B, T = self.batch_size, self.batch_length
        flat_batch = {}

        for key, value in batch.items():
            if key.startswith('obs_'):
                # Get observation shape
                obs_key = key[4:]  # Remove 'obs_' prefix
                obs_shape = self.obs_shapes[obs_key]
                # (B, T, *shape) -> (B*T, *shape)
                flat_batch[key] = value.reshape((B * T,) + obs_shape)
            elif key == 'action':
                # (B, T, A) -> (B*T, A)
                flat_batch[key] = value.reshape(B * T, self.action_dim)
            else:
                # (B, T) -> (B*T,)
                flat_batch[key] = value.reshape(B * T)

        return flat_batch

    def stats(self, state: ReplayBufferState) -> Dict[str, Any]:
        """
        Get buffer statistics.

        Args:
            state: Current buffer state

        Returns:
            Dictionary with buffer stats
        """
        valid_timesteps = int(state.valid_size) // self.num_envs
        return {
            'size': int(state.valid_size),
            'capacity': self.capacity,
            'total_steps': int(state.total_steps),
            'write_idx': int(state.write_idx),
            'valid_timesteps': valid_timesteps,
            'full': bool(state.valid_size == self.capacity),
        }


class ReservoirReplayBuffer(ReplayBuffer):
    """
    Replay buffer with reservoir sampling for continual learning.

    Note: This performs reservoir sampling at the *sequence* level, not timestep level,
    to maintain temporal coherence. This ensures that sampled sequences are always
    temporally contiguous, even after reservoir sampling kicks in across task boundaries.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Number of sequences we can store
        # Each sequence takes batch_length timesteps
        self.num_sequences = self.time_capacity // self.batch_length
        if self.time_capacity % self.batch_length != 0:
            # Adjust capacity to be divisible by sequence length
            self.num_sequences = self.time_capacity // self.batch_length
            self.time_capacity = self.num_sequences * self.batch_length
            self.capacity = self.time_capacity * self.num_envs
            print(f"Warning: Adjusted capacity to {self.capacity} to fit {self.num_sequences} sequences of length {self.batch_length}")

    def add_batch(
        self,
        state: ReplayBufferState,
        transitions: Dict[str, jnp.ndarray],
    ) -> ReplayBufferState:
        """
        Add batch using sequence-level reservoir sampling.

        This adds one timestep at a time to build up full sequences.
        Once a sequence is complete, we decide whether to keep it via reservoir sampling.

        Args:
            state: Current buffer state
            transitions: Dictionary with arrays of shape (num_envs, ...)

        Returns:
            Updated buffer state
        """
        batch_size = int(next(iter(transitions.values())).shape[0])
        assert batch_size == self.num_envs, f"Batch size {batch_size} must match num_envs {self.num_envs}"

        # Cast JAX scalars to Python ints
        curr_valid = int(state.valid_size)
        curr_total = int(state.total_steps)
        seq_step = int(state.seq_step)
        seq_target = int(state.seq_target)

        # If buffer not full, use standard FIFO
        if curr_valid < self.capacity:
            # During FIFO phase, reset sequence tracking since we're not using reservoir yet
            return super().add_batch(state, transitions).replace(
                seq_step=jnp.array(0, dtype=jnp.int32),
                seq_target=jnp.array(-1, dtype=jnp.int32),
            )

        # Buffer is full - use sequence-level reservoir sampling
        # Freeze valid_size and write_idx for clarity (they're no longer meaningful in reservoir mode)
        # valid_size stays at capacity, write_idx at 0
        frozen_valid_size = jnp.array(self.capacity, dtype=jnp.int32)
        frozen_write_idx = jnp.array(0, dtype=jnp.int32)

        # Calculate how many complete sequences we've seen so far (including the one we're building)
        total_sequences = curr_total // (self.num_envs * self.batch_length)

        # At the start of a new sequence, decide if we'll keep it
        if seq_step == 0:
            # Starting a new sequence - decide if we'll keep it
            rng, rng_uniform, rng_randint = jax.random.split(state.rng, 3)

            keep_prob = self.num_sequences / (total_sequences + 1)
            should_keep = jax.random.uniform(rng_uniform) < keep_prob

            if should_keep:
                # Choose which sequence slot to replace
                replace_seq = int(jax.random.randint(
                    rng_randint, (), minval=0, maxval=self.num_sequences
                ))
                seq_target = replace_seq
            else:
                seq_target = -1  # Skip this sequence

            state = state.replace(rng=rng)
        else:
            # Continue with decision made at start of sequence
            rng = state.rng

        # Check if we should write this timestep
        if seq_target >= 0:
            # Calculate where to write in the buffer
            target_time_start = seq_target * self.batch_length
            time_idx = target_time_start + seq_step
            start_idx = (time_idx * self.num_envs) % self.capacity
            indices = np.arange(start_idx, start_idx + self.num_envs) % self.capacity

            # Store all environments' transitions
            for key in self.obs_shapes:
                obs_key = f'obs_{key}'
                if obs_key in transitions:
                    obs_value = transitions[obs_key]
                    if isinstance(obs_value, jnp.ndarray):
                        obs_value = np.array(obs_value)
                    state.data[obs_key][indices] = obs_value

            if 'action' in transitions:
                action_value = transitions['action']
                if isinstance(action_value, jnp.ndarray):
                    action_value = np.array(action_value)
                state.data['action'][indices] = action_value

            for key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                if key in transitions:
                    value = transitions[key]
                    if isinstance(value, jnp.ndarray):
                        value = np.array(value)
                    state.data[key][indices] = value

            if 'task_id' in transitions:
                task_id_value = transitions['task_id']
                if isinstance(task_id_value, jnp.ndarray):
                    task_id_value = np.array(task_id_value)
                state.data['task_id'][indices] = task_id_value

        # Advance sequence step counter
        new_seq_step = (seq_step + 1) % self.batch_length

        # Update total steps
        new_total_steps = curr_total + self.num_envs

        return state.replace(
            write_idx=frozen_write_idx,
            valid_size=frozen_valid_size,
            total_steps=jnp.array(new_total_steps, dtype=jnp.int32),
            seq_step=jnp.array(new_seq_step, dtype=jnp.int32),
            seq_target=jnp.array(seq_target, dtype=jnp.int32),
            rng=rng,
        )

    def sample(
        self,
        state: ReplayBufferState,
        rng: jax.random.PRNGKey,
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample sequences respecting sequence-slot boundaries.

        With sequence-level reservoir sampling, the buffer contains sequence slots
        that may be from different tasks. We sample aligned to these slot boundaries
        to ensure temporal coherence within each sub-trajectory.

        This allows mixing tasks (e.g., 8 frames from Task 1 + 8 frames from Task 2),
        as long as each sub-trajectory is temporally coherent.

        Args:
            state: Current buffer state
            rng: Random key for sampling

        Returns:
            Dictionary with sampled sequences
        """
        # Split RNG for env and sequence sampling
        rng_env, rng_seq = jax.random.split(rng)

        # Calculate how many complete timesteps we have
        valid_timesteps = int(state.valid_size) // self.num_envs

        # We need at least batch_length timesteps to sample a sequence
        if valid_timesteps < self.batch_length:
            raise ValueError(
                f"Not enough timesteps for sampling: have {valid_timesteps}, "
                f"need {self.batch_length}. Consider increasing pretrain steps."
            )

        # Sample random (env, sequence_slot) pairs
        # Each sequence slot contains batch_length consecutive timesteps
        env_indices = jax.random.randint(
            rng_env,
            (self.batch_size,),
            minval=0,
            maxval=self.num_envs,
        )

        # Sample which sequence slot to start from
        # During FIFO filling, we have fewer complete sequences
        # After reservoir kicks in, we always have num_sequences slots
        num_available_slots = min(valid_timesteps // self.batch_length, self.num_sequences)

        # Protect the active slot if it's being written to
        # If seq_step != 0, the slot at seq_target is partially written and unsafe to sample
        active_slot = int(state.seq_target)
        active_incomplete = (int(state.seq_step) != 0 and active_slot >= 0)

        seq_slot_indices = jax.random.randint(
            rng_seq,
            (self.batch_size,),
            minval=0,
            maxval=num_available_slots,
        )

        if active_incomplete:
            # Shift any occurrences of the active slot to avoid sampling partially-written sequences
            # This slightly biases the distribution but only for 1 slot out of N
            seq_slot_np = np.array(seq_slot_indices)
            mask = (seq_slot_np == active_slot)
            if np.any(mask):
                # Send them to (active_slot + 1) % num_available_slots
                seq_slot_np[mask] = (active_slot + 1) % num_available_slots
                seq_slot_indices = jnp.array(seq_slot_np)

        # Convert to start times (each slot starts at slot_idx * batch_length)
        start_times = np.array(seq_slot_indices) * self.batch_length  # (B,)
        env_np = np.array(env_indices)  # (B,)
        time_offsets = np.arange(self.batch_length)  # (T,)

        # (B, T) time indices
        time_idx = start_times[:, None] + time_offsets[None, :]

        # (B, T) linear indices with stride=num_envs
        seq_idxs = (time_idx * self.num_envs + env_np[:, None]) % self.capacity

        # Gather sequences (same as base class)
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

        return batch
