import os

import jax
import numpy as np
import psutil


def get_ram_usage() -> float:
    """Returns the current process's RAM usage in gigabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


class RNGManager:
    """
    Manages JAX RNG keys for reproducibility.

    JAX uses explicit PRNG key splitting for deterministic randomness.
    This is superior to PyTorch's global RNG state for reproducibility.
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: The random seed to use for reproducibility.
        """
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        # Checkpoint for evaluation (save/restore RNG state)
        self.checkpoint_rng = None

    def split(self, num: int = 2) -> tuple:
        """
        Split the current RNG key.

        Args:
            num: Number of keys to generate (default: 2)

        Returns:
            Tuple of RNG keys. First key updates self.rng, rest are returned.
        """
        keys = jax.random.split(self.rng, num)
        self.rng = keys[0]
        return keys[1:] if num == 2 else keys[1:]

    def get_key(self) -> jax.random.PRNGKey:
        """
        Get a new RNG key and advance the internal state.

        Returns:
            Fresh RNG key
        """
        self.rng, key = jax.random.split(self.rng)
        return key

    def checkpoint(self) -> None:
        """Checkpoint the current RNG state (for evaluation)."""
        self.checkpoint_rng = self.rng

    def restore(self) -> None:
        """Restore the RNG state from checkpoint."""
        if self.checkpoint_rng is not None:
            self.rng = self.checkpoint_rng
