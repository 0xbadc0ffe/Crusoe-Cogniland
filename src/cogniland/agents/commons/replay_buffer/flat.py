"""
Flat replay buffer for transition-level sampling (IQN, DQN, etc.).

Unlike ReplayBuffer which samples sequences for world-model agents,
FlatBuffer samples individual transitions (s, a, r, s', done).

Inspired by Flashbax's flat_buffer implementation.
"""
from typing import Dict, Tuple, Optional, Any

import jax
import jax.numpy as jnp
import numpy as np
import chex


@chex.dataclass
class FlatBufferState:
    """State for the flat replay buffer."""

    # Data arrays - store transitions in circular buffer
    obs: np.ndarray           # (capacity, *obs_shape)
    action: np.ndarray        # (capacity,) for discrete actions
    reward: np.ndarray        # (capacity,) - for 1-step, or n_step_return for n-step
    next_obs: np.ndarray      # (capacity, *obs_shape) - s_{t+1} for 1-step, or s_{t+n} for n-step
    done: np.ndarray          # (capacity,)

    # N-step support
    actual_n: np.ndarray      # (capacity,) - actual steps taken (may be < n if early terminal)

    # Buffer management
    write_idx: jnp.ndarray    # Current write position
    valid_size: jnp.ndarray   # Number of valid entries
    total_steps: jnp.ndarray  # Total steps added (for stats)
    rng: jax.random.PRNGKey   # RNG key

    # Reservoir sampling state
    reservoir_count: jnp.ndarray  # Total transitions seen (for reservoir probability)


class FlatBuffer:
    """
    FIFO Flat replay buffer for transition-level sampling.

    Stores individual transitions (obs, action, reward, next_obs, done)
    and samples random batches for off-policy learning.

    This is simpler than ReplayBuffer which maintains temporal coherence
    for sequence sampling. FlatBuffer is suitable for DQN-family algorithms.
    """

    def __init__(
        self,
        capacity: int = 131072,
        obs_shape: Tuple[int, ...] = (7, 7, 3),
        seed: int = 0,
    ):
        """
        Initialize flat replay buffer.

        Args:
            capacity: Maximum buffer size (number of transitions)
            obs_shape: Shape of observations (e.g., (7, 7, 3) for MiniGrid)
            seed: Random seed
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.seed = seed

        # Determine dtype for observations
        # Use uint8 for images to save memory
        if len(obs_shape) == 3 and obs_shape[-1] in [1, 3, 4]:
            self.obs_dtype = np.uint8
        else:
            self.obs_dtype = np.float32

    def init(self, rng: Optional[jax.random.PRNGKey] = None) -> FlatBufferState:
        """
        Initialize empty buffer state.

        Args:
            rng: Random key (optional, uses seed if not provided)

        Returns:
            Initial buffer state
        """
        if rng is None:
            rng = jax.random.PRNGKey(self.seed)

        # Use numpy arrays for CPU storage (mutable, support in-place updates)
        obs = np.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype)
        next_obs = np.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype)
        action = np.zeros(self.capacity, dtype=np.int32)
        reward = np.zeros(self.capacity, dtype=np.float32)
        done = np.zeros(self.capacity, dtype=bool)
        actual_n = np.ones(self.capacity, dtype=np.int32)  # Default n=1 for single-step

        return FlatBufferState(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            actual_n=actual_n,
            write_idx=jnp.array(0, dtype=jnp.int32),
            valid_size=jnp.array(0, dtype=jnp.int32),
            total_steps=jnp.array(0, dtype=jnp.int32),
            rng=rng,
            reservoir_count=jnp.array(0, dtype=jnp.int32),
        )

    def add(
        self,
        state: FlatBufferState,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        actual_n: np.ndarray = None,
    ) -> FlatBufferState:
        """
        Add a single transition to the buffer.

        Args:
            state: Current buffer state
            obs: Observation (obs_shape,)
            action: Action (scalar)
            reward: Reward (scalar) or n-step return
            next_obs: Next observation (obs_shape,) or n-step ahead observation
            done: Done flag (scalar)
            actual_n: Actual number of steps (default 1 for single-step)

        Returns:
            Updated buffer state
        """
        idx = int(state.write_idx)

        # Convert JAX arrays to numpy if needed
        if isinstance(obs, jnp.ndarray):
            obs = np.array(obs)
        if isinstance(next_obs, jnp.ndarray):
            next_obs = np.array(next_obs)
        if isinstance(action, jnp.ndarray):
            action = np.array(action)
        if isinstance(reward, jnp.ndarray):
            reward = np.array(reward)
        if isinstance(done, jnp.ndarray):
            done = np.array(done)
        if actual_n is None:
            actual_n = np.array(1, dtype=np.int32)
        elif isinstance(actual_n, jnp.ndarray):
            actual_n = np.array(actual_n)

        # Store transition
        state.obs[idx] = obs
        state.action[idx] = action
        state.reward[idx] = reward
        state.next_obs[idx] = next_obs
        state.done[idx] = done
        state.actual_n[idx] = actual_n

        # Update indices
        new_write_idx = (idx + 1) % self.capacity
        new_valid_size = min(int(state.valid_size) + 1, self.capacity)
        new_total_steps = int(state.total_steps) + 1

        return state.replace(
            write_idx=jnp.array(new_write_idx, dtype=jnp.int32),
            valid_size=jnp.array(new_valid_size, dtype=jnp.int32),
            total_steps=jnp.array(new_total_steps, dtype=jnp.int32),
        )

    def add_batch(
        self,
        state: FlatBufferState,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        actual_n: np.ndarray = None,
    ) -> FlatBufferState:
        """
        Add a batch of transitions to the buffer.

        Args:
            state: Current buffer state
            obs: Observations (batch_size, *obs_shape)
            action: Actions (batch_size,)
            reward: Rewards (batch_size,) or n-step returns
            next_obs: Next observations (batch_size, *obs_shape) or n-step ahead
            done: Done flags (batch_size,)
            actual_n: Actual steps per transition (batch_size,), default 1

        Returns:
            Updated buffer state
        """
        batch_size = obs.shape[0]

        # Convert JAX arrays to numpy if needed
        if isinstance(obs, jnp.ndarray):
            obs = np.array(obs)
        if isinstance(next_obs, jnp.ndarray):
            next_obs = np.array(next_obs)
        if isinstance(action, jnp.ndarray):
            action = np.array(action)
        if isinstance(reward, jnp.ndarray):
            reward = np.array(reward)
        if isinstance(done, jnp.ndarray):
            done = np.array(done)
        if actual_n is None:
            actual_n = np.ones(batch_size, dtype=np.int32)
        elif isinstance(actual_n, jnp.ndarray):
            actual_n = np.array(actual_n)

        # Calculate indices for circular buffer
        start_idx = int(state.write_idx)
        indices = np.arange(start_idx, start_idx + batch_size) % self.capacity

        # Store transitions
        state.obs[indices] = obs
        state.action[indices] = action
        state.reward[indices] = reward
        state.next_obs[indices] = next_obs
        state.done[indices] = done
        state.actual_n[indices] = actual_n

        # Update indices
        new_write_idx = (start_idx + batch_size) % self.capacity
        new_valid_size = min(int(state.valid_size) + batch_size, self.capacity)
        new_total_steps = int(state.total_steps) + batch_size

        return state.replace(
            write_idx=jnp.array(new_write_idx, dtype=jnp.int32),
            valid_size=jnp.array(new_valid_size, dtype=jnp.int32),
            total_steps=jnp.array(new_total_steps, dtype=jnp.int32),
        )

    def sample(
        self,
        state: FlatBufferState,
        rng: jax.random.PRNGKey,
        batch_size: int,
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample a batch of transitions from the buffer.

        Uses NumPy RNG for sampling to avoid host↔device sync overhead.
        The buffer is stored on CPU, so using JAX RNG would cause:
        1. device→host sync to get indices
        2. host→device transfer for batch data
        By using NumPy RNG, we eliminate step 1 and do a single device_put.

        Args:
            state: Current buffer state
            rng: Random key for sampling (converted to NumPy seed)
            batch_size: Number of transitions to sample

        Returns:
            Dictionary with sampled transitions:
                - obs: (batch_size, *obs_shape)
                - action: (batch_size,)
                - reward: (batch_size,)
                - next_obs: (batch_size, *obs_shape)
                - done: (batch_size,)
        """
        valid_size = int(state.valid_size)

        if valid_size < batch_size:
            raise ValueError(
                f"Not enough transitions for sampling: have {valid_size}, "
                f"need {batch_size}."
            )

        # Use NumPy RNG for CPU-side sampling (avoids device→host sync)
        # Convert JAX key to seed for NumPy Generator
        seed = int(jax.random.randint(rng, (), 0, 2**31 - 1))
        np_rng = np.random.Generator(np.random.PCG64(seed))
        indices = np_rng.integers(0, valid_size, size=batch_size)

        # Gather transitions on CPU
        obs_np = state.obs[indices]
        action_np = state.action[indices]
        reward_np = state.reward[indices]
        next_obs_np = state.next_obs[indices]
        done_np = state.done[indices]
        actual_n_np = state.actual_n[indices]

        # Keep original dtype - preprocessing functions handle type conversion and normalization

        # Single device_put for the entire batch (one host→device transfer)
        batch_np = {
            'obs': obs_np,
            'action': action_np,
            'reward': reward_np,
            'next_obs': next_obs_np,
            'done': done_np,
            'actual_n': actual_n_np,
        }
        return jax.device_put(jax.tree.map(jnp.asarray, batch_np))

    def can_sample(self, state: FlatBufferState, min_size: int) -> bool:
        """
        Check if buffer has enough data for sampling.

        Args:
            state: Current buffer state
            min_size: Minimum number of transitions needed

        Returns:
            True if buffer has at least min_size transitions
        """
        return int(state.valid_size) >= min_size

    def stats(self, state: FlatBufferState) -> Dict[str, Any]:
        """
        Get buffer statistics.

        Args:
            state: Current buffer state

        Returns:
            Dictionary with buffer stats
        """
        return {
            'size': int(state.valid_size),
            'capacity': self.capacity,
            'total_steps': int(state.total_steps),
            'write_idx': int(state.write_idx),
            'full': bool(state.valid_size == self.capacity),
        }


class ReservoirFlatBuffer(FlatBuffer):
    """
    Flat replay buffer with reservoir sampling for continual learning.

    Uses Algorithm R (Vitter, 1985) for uniform sampling from a stream
    of unbounded length. Once the buffer is full, each new transition
    has a probability of capacity/n of replacing a random existing transition,
    where n is the total number of transitions seen.

    This ensures that after seeing n transitions, each transition has
    an equal probability of being in the buffer.
    """

    def add(
        self,
        state: FlatBufferState,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        actual_n: np.ndarray = None,
    ) -> FlatBufferState:
        """
        Add a single transition using reservoir sampling.

        Args:
            state: Current buffer state
            obs: Observation
            action: Action
            reward: Reward
            next_obs: Next observation
            done: Done flag
            actual_n: Actual number of steps (default 1)

        Returns:
            Updated buffer state
        """
        curr_valid = int(state.valid_size)
        total_seen = int(state.reservoir_count)

        # Convert JAX arrays to numpy if needed
        if isinstance(obs, jnp.ndarray):
            obs = np.array(obs)
        if isinstance(next_obs, jnp.ndarray):
            next_obs = np.array(next_obs)
        if isinstance(action, jnp.ndarray):
            action = np.array(action)
        if isinstance(reward, jnp.ndarray):
            reward = np.array(reward)
        if isinstance(done, jnp.ndarray):
            done = np.array(done)
        if actual_n is None:
            actual_n = np.array(1, dtype=np.int32)
        elif isinstance(actual_n, jnp.ndarray):
            actual_n = np.array(actual_n)

        # If buffer not full, use standard FIFO
        if curr_valid < self.capacity:
            state = super().add(state, obs, action, reward, next_obs, done, actual_n)
            return state.replace(
                reservoir_count=jnp.array(total_seen + 1, dtype=jnp.int32),
            )

        # Buffer is full - use reservoir sampling
        rng, rng_prob, rng_idx = jax.random.split(state.rng, 3)

        # Probability of keeping this transition
        keep_prob = self.capacity / (total_seen + 1)
        should_keep = jax.random.uniform(rng_prob) < keep_prob

        if should_keep:
            # Choose random index to replace
            idx = int(jax.random.randint(rng_idx, (), minval=0, maxval=self.capacity))
            state.obs[idx] = obs
            state.action[idx] = action
            state.reward[idx] = reward
            state.next_obs[idx] = next_obs
            state.done[idx] = done
            state.actual_n[idx] = actual_n

        return state.replace(
            rng=rng,
            reservoir_count=jnp.array(total_seen + 1, dtype=jnp.int32),
            total_steps=jnp.array(int(state.total_steps) + 1, dtype=jnp.int32),
        )

    def add_batch(
        self,
        state: FlatBufferState,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        actual_n: np.ndarray = None,
    ) -> FlatBufferState:
        """
        Add a batch of transitions using reservoir sampling.

        For efficiency, processes the batch in two phases:
        1. FIFO phase: Fill buffer until full
        2. Reservoir phase: Apply reservoir sampling for remaining transitions

        Args:
            state: Current buffer state
            obs: Observations (batch_size, *obs_shape)
            action: Actions (batch_size,)
            reward: Rewards (batch_size,)
            next_obs: Next observations (batch_size, *obs_shape)
            done: Done flags (batch_size,)
            actual_n: Actual steps per transition (batch_size,), default 1

        Returns:
            Updated buffer state
        """
        batch_size = obs.shape[0]
        curr_valid = int(state.valid_size)
        total_seen = int(state.reservoir_count)

        # Convert JAX arrays to numpy if needed
        if isinstance(obs, jnp.ndarray):
            obs = np.array(obs)
        if isinstance(next_obs, jnp.ndarray):
            next_obs = np.array(next_obs)
        if isinstance(action, jnp.ndarray):
            action = np.array(action)
        if isinstance(reward, jnp.ndarray):
            reward = np.array(reward)
        if isinstance(done, jnp.ndarray):
            done = np.array(done)
        if actual_n is None:
            actual_n = np.ones(batch_size, dtype=np.int32)
        elif isinstance(actual_n, jnp.ndarray):
            actual_n = np.array(actual_n)

        # Phase 1: FIFO until buffer is full
        if curr_valid < self.capacity:
            space_left = self.capacity - curr_valid
            fifo_count = min(batch_size, space_left)

            if fifo_count > 0:
                start_idx = int(state.write_idx)
                indices = np.arange(start_idx, start_idx + fifo_count) % self.capacity

                state.obs[indices] = obs[:fifo_count]
                state.action[indices] = action[:fifo_count]
                state.reward[indices] = reward[:fifo_count]
                state.next_obs[indices] = next_obs[:fifo_count]
                state.done[indices] = done[:fifo_count]
                state.actual_n[indices] = actual_n[:fifo_count]

                new_write_idx = (start_idx + fifo_count) % self.capacity
                new_valid_size = curr_valid + fifo_count
                total_seen += fifo_count

                state = state.replace(
                    write_idx=jnp.array(new_write_idx, dtype=jnp.int32),
                    valid_size=jnp.array(new_valid_size, dtype=jnp.int32),
                )

            # Process remaining with reservoir sampling
            remaining_start = fifo_count
        else:
            remaining_start = 0

        # Phase 2: Reservoir sampling for remaining transitions
        if remaining_start < batch_size:
            rng = state.rng

            for i in range(remaining_start, batch_size):
                total_seen += 1
                rng, rng_prob, rng_idx = jax.random.split(rng, 3)

                keep_prob = self.capacity / total_seen
                should_keep = float(jax.random.uniform(rng_prob)) < keep_prob

                if should_keep:
                    idx = int(jax.random.randint(rng_idx, (), minval=0, maxval=self.capacity))
                    state.obs[idx] = obs[i]
                    state.action[idx] = action[i]
                    state.reward[idx] = reward[i]
                    state.next_obs[idx] = next_obs[i]
                    state.done[idx] = done[i]
                    state.actual_n[idx] = actual_n[i]

            state = state.replace(rng=rng)

        return state.replace(
            reservoir_count=jnp.array(total_seen, dtype=jnp.int32),
            total_steps=jnp.array(int(state.total_steps) + batch_size, dtype=jnp.int32),
        )


class SumTree:
    """
    Sum Tree data structure for O(log n) priority-based sampling.

    The tree is stored as a flat array where:
    - tree[0] is the root (sum of all priorities)
    - tree[2*i + 1] and tree[2*i + 2] are children of tree[i]
    - Leaf nodes store actual priorities
    - Internal nodes store sum of children

    For capacity N, we need 2*N - 1 nodes total.
    """

    @staticmethod
    def create(capacity: int) -> np.ndarray:
        """Create an empty sum tree array."""
        tree_size = 2 * capacity - 1
        return np.zeros(tree_size, dtype=np.float32)

    @staticmethod
    def update(tree: np.ndarray, capacity: int, idx: int, priority: float):
        """Update priority at leaf index and propagate up."""
        tree_idx = idx + capacity - 1
        change = priority - tree[tree_idx]
        tree[tree_idx] = priority

        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            tree[tree_idx] += change

    @staticmethod
    def update_batch(tree: np.ndarray, capacity: int, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for a batch of indices."""
        for idx, priority in zip(indices, priorities):
            SumTree.update(tree, capacity, int(idx), float(priority))

    @staticmethod
    def total(tree: np.ndarray) -> float:
        """Get total priority (root value)."""
        return tree[0]

    @staticmethod
    def sample(tree: np.ndarray, capacity: int, value: float) -> int:
        """Sample a leaf index based on cumulative priority."""
        idx = 0
        while idx < capacity - 1:
            left = 2 * idx + 1
            right = 2 * idx + 2
            if value <= tree[left]:
                idx = left
            else:
                value -= tree[left]
                idx = right
        return idx - (capacity - 1)

    @staticmethod
    def get(tree: np.ndarray, capacity: int, idx: int) -> float:
        """Get priority at leaf index."""
        return tree[idx + capacity - 1]


@chex.dataclass
class PERBufferState:
    """State for PER buffer (extends FlatBufferState with priority tree)."""
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray
    actual_n: np.ndarray
    write_idx: jnp.ndarray
    valid_size: jnp.ndarray
    total_steps: jnp.ndarray
    rng: jax.random.PRNGKey
    reservoir_count: jnp.ndarray
    sum_tree: np.ndarray
    max_priority: jnp.ndarray


class PERFlatBuffer:
    """
    Prioritized Experience Replay buffer with configurable insertion policy.

    Combines:
    - Insertion policy: FIFO or Reservoir (how to add/evict transitions)
    - Sampling policy: Priority-based (how to select transitions for training)

    References:
    - Schaul et al. (2015): https://arxiv.org/abs/1511.05952
    """

    def __init__(
        self,
        capacity: int = 131072,
        obs_shape: Tuple[int, ...] = (7, 7, 3),
        seed: int = 0,
        insertion_type: str = "fifo",
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
    ):
        """
        Initialize PER buffer.

        Args:
            capacity: Maximum buffer size
            obs_shape: Shape of observations
            seed: Random seed
            insertion_type: "fifo" or "reservoir"
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta_start: Initial IS weight exponent (anneals to 1)
            beta_frames: Number of frames to anneal beta over
            epsilon: Small constant added to priorities
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.seed = seed
        self.insertion_type = insertion_type
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self._frame_count = 0

        if len(obs_shape) == 3 and obs_shape[-1] in [1, 3, 4]:
            self.obs_dtype = np.uint8
        else:
            self.obs_dtype = np.float32

    def _get_beta(self) -> float:
        """Get current beta value (anneals from beta_start to 1)."""
        fraction = min(1.0, self._frame_count / self.beta_frames)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def init(self, rng: Optional[jax.random.PRNGKey] = None) -> PERBufferState:
        """Initialize empty PER buffer state."""
        if rng is None:
            rng = jax.random.PRNGKey(self.seed)

        return PERBufferState(
            obs=np.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype),
            action=np.zeros(self.capacity, dtype=np.int32),
            reward=np.zeros(self.capacity, dtype=np.float32),
            next_obs=np.zeros((self.capacity,) + self.obs_shape, dtype=self.obs_dtype),
            done=np.zeros(self.capacity, dtype=bool),
            actual_n=np.ones(self.capacity, dtype=np.int32),
            write_idx=jnp.array(0, dtype=jnp.int32),
            valid_size=jnp.array(0, dtype=jnp.int32),
            total_steps=jnp.array(0, dtype=jnp.int32),
            rng=rng,
            reservoir_count=jnp.array(0, dtype=jnp.int32),
            sum_tree=SumTree.create(self.capacity),
            max_priority=jnp.array(1.0, dtype=jnp.float32),
        )

    def add_batch(
        self,
        state: PERBufferState,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        actual_n: np.ndarray = None,
    ) -> PERBufferState:
        """Add a batch of transitions with max priority."""
        batch_size = obs.shape[0]

        if isinstance(obs, jnp.ndarray):
            obs = np.array(obs)
        if isinstance(next_obs, jnp.ndarray):
            next_obs = np.array(next_obs)
        if isinstance(action, jnp.ndarray):
            action = np.array(action)
        if isinstance(reward, jnp.ndarray):
            reward = np.array(reward)
        if isinstance(done, jnp.ndarray):
            done = np.array(done)
        if actual_n is None:
            actual_n = np.ones(batch_size, dtype=np.int32)
        elif isinstance(actual_n, jnp.ndarray):
            actual_n = np.array(actual_n)

        priority = float(state.max_priority) ** self.alpha

        if self.insertion_type == "fifo":
            return self._add_batch_fifo(state, obs, action, reward, next_obs, done, actual_n, priority)
        else:
            return self._add_batch_reservoir(state, obs, action, reward, next_obs, done, actual_n, priority)

    def _add_batch_fifo(self, state, obs, action, reward, next_obs, done, actual_n, priority):
        """Add batch using FIFO insertion."""
        batch_size = obs.shape[0]
        start_idx = int(state.write_idx)
        indices = np.arange(start_idx, start_idx + batch_size) % self.capacity

        state.obs[indices] = obs
        state.action[indices] = action
        state.reward[indices] = reward
        state.next_obs[indices] = next_obs
        state.done[indices] = done
        state.actual_n[indices] = actual_n

        for idx in indices:
            SumTree.update(state.sum_tree, self.capacity, idx, priority)

        return state.replace(
            write_idx=jnp.array((start_idx + batch_size) % self.capacity, dtype=jnp.int32),
            valid_size=jnp.array(min(int(state.valid_size) + batch_size, self.capacity), dtype=jnp.int32),
            total_steps=jnp.array(int(state.total_steps) + batch_size, dtype=jnp.int32),
        )

    def _add_batch_reservoir(self, state, obs, action, reward, next_obs, done, actual_n, priority):
        """Add batch using reservoir sampling insertion."""
        batch_size = obs.shape[0]
        curr_valid = int(state.valid_size)
        total_seen = int(state.reservoir_count)

        if curr_valid < self.capacity:
            space_left = self.capacity - curr_valid
            fifo_count = min(batch_size, space_left)

            if fifo_count > 0:
                start_idx = int(state.write_idx)
                indices = np.arange(start_idx, start_idx + fifo_count) % self.capacity

                state.obs[indices] = obs[:fifo_count]
                state.action[indices] = action[:fifo_count]
                state.reward[indices] = reward[:fifo_count]
                state.next_obs[indices] = next_obs[:fifo_count]
                state.done[indices] = done[:fifo_count]
                state.actual_n[indices] = actual_n[:fifo_count]

                for idx in indices:
                    SumTree.update(state.sum_tree, self.capacity, idx, priority)

                total_seen += fifo_count
                state = state.replace(
                    write_idx=jnp.array((start_idx + fifo_count) % self.capacity, dtype=jnp.int32),
                    valid_size=jnp.array(curr_valid + fifo_count, dtype=jnp.int32),
                )

            remaining_start = fifo_count
        else:
            remaining_start = 0

        if remaining_start < batch_size:
            rng = state.rng
            for i in range(remaining_start, batch_size):
                total_seen += 1
                rng, rng_prob, rng_idx = jax.random.split(rng, 3)
                if float(jax.random.uniform(rng_prob)) < self.capacity / total_seen:
                    idx = int(jax.random.randint(rng_idx, (), 0, self.capacity))
                    state.obs[idx] = obs[i]
                    state.action[idx] = action[i]
                    state.reward[idx] = reward[i]
                    state.next_obs[idx] = next_obs[i]
                    state.done[idx] = done[i]
                    state.actual_n[idx] = actual_n[i]
                    SumTree.update(state.sum_tree, self.capacity, idx, priority)
            state = state.replace(rng=rng)

        return state.replace(
            reservoir_count=jnp.array(total_seen, dtype=jnp.int32),
            total_steps=jnp.array(int(state.total_steps) + batch_size, dtype=jnp.int32),
        )

    def sample(
        self,
        state: PERBufferState,
        rng: jax.random.PRNGKey,
        batch_size: int,
    ) -> Tuple[Dict[str, jnp.ndarray], np.ndarray, jnp.ndarray]:
        """
        Sample a batch based on priorities.

        Returns:
            Tuple of (batch, indices, is_weights)
        """
        valid_size = int(state.valid_size)
        if valid_size < batch_size:
            raise ValueError(f"Not enough transitions: have {valid_size}, need {batch_size}.")

        # Guard against zero total priority (shouldn't happen with proper initialization)
        total_priority = max(SumTree.total(state.sum_tree), 1e-8)
        segment_size = total_priority / batch_size

        seed = int(jax.random.randint(rng, (), 0, 2**31 - 1))
        np_rng = np.random.Generator(np.random.PCG64(seed))

        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            value = np_rng.uniform(segment_size * i, segment_size * (i + 1))
            idx = SumTree.sample(state.sum_tree, self.capacity, value)
            # If tree returns invalid index (shouldn't happen), resample uniformly
            if idx < 0 or idx >= valid_size:
                idx = np_rng.integers(0, valid_size)
            indices[i] = idx
            priorities[i] = max(SumTree.get(state.sum_tree, self.capacity, idx), 1e-8)

        sampling_probs = priorities / total_priority
        beta = self._get_beta()
        is_weights = (valid_size * sampling_probs) ** (-beta)
        is_weights = is_weights / is_weights.max()

        self._frame_count += batch_size

        obs_np = state.obs[indices]
        next_obs_np = state.next_obs[indices]
        # Keep original dtype - preprocessing functions handle type conversion and normalization

        batch = jax.device_put({
            'obs': jnp.asarray(obs_np),
            'action': jnp.asarray(state.action[indices]),
            'reward': jnp.asarray(state.reward[indices]),
            'next_obs': jnp.asarray(next_obs_np),
            'done': jnp.asarray(state.done[indices]),
            'actual_n': jnp.asarray(state.actual_n[indices]),
        })

        return batch, indices, jax.device_put(jnp.asarray(is_weights.astype(np.float32)))

    def update_priorities(
        self,
        state: PERBufferState,
        indices: np.ndarray,
        priorities: np.ndarray,
    ) -> PERBufferState:
        """Update priorities after computing TD errors."""
        if isinstance(priorities, jnp.ndarray):
            priorities = np.array(priorities)
        if isinstance(indices, jnp.ndarray):
            indices = np.array(indices)

        priorities = np.maximum(priorities, self.epsilon)
        priorities_alpha = priorities ** self.alpha
        SumTree.update_batch(state.sum_tree, self.capacity, indices, priorities_alpha)

        new_max = max(float(state.max_priority), priorities.max())
        return state.replace(max_priority=jnp.array(new_max, dtype=jnp.float32))

    def can_sample(self, state: PERBufferState, min_size: int) -> bool:
        """Check if buffer has enough data for sampling."""
        return int(state.valid_size) >= min_size

    def stats(self, state: PERBufferState) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'size': int(state.valid_size),
            'capacity': self.capacity,
            'total_steps': int(state.total_steps),
            'insertion_type': self.insertion_type,
            'total_priority': SumTree.total(state.sum_tree),
            'max_priority': float(state.max_priority),
            'beta': self._get_beta(),
        }
