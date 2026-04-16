import jax
import jax.numpy as jnp


class TaskSampler:
    """Assigns a task index to each parallel env at episode reset."""

    def __init__(self, num_tasks: int, num_envs: int, mode: str = "round_robin"):
        self.num_tasks = num_tasks
        self.num_envs = num_envs
        self.mode = mode
        self._counter = 0

    def sample(self, rng=None) -> jnp.ndarray:
        """Returns (num_envs,) int array of task indices."""
        if self.mode == "round_robin":
            tasks = jnp.array([(self._counter + i) % self.num_tasks
                               for i in range(self.num_envs)])
            self._counter += self.num_envs
            return tasks
        elif self.mode == "uniform_random":
            return jax.random.randint(rng, (self.num_envs,), 0, self.num_tasks)
        else:
            raise ValueError(f"Unknown task_sampling mode: {self.mode}")

    def fixed(self, task_id: int) -> jnp.ndarray:
        """All envs run the same task (used during eval)."""
        return jnp.full((self.num_envs,), task_id, dtype=jnp.int32)
