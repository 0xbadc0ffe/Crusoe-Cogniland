from collections import deque
from enum import Enum
import time
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from cl.shared import setup_logger

logger = setup_logger(__name__)


class MetricsTrackerMode(Enum):
    """Tracking mode for the metrics tracker."""

    TRAIN = "train"
    EVAL = "eval"

    def __str__(self) -> str:
        return self.value

    def __fspath__(self) -> str:
        """Make the enum compatible with os.path functions"""
        return self.value


class TaskMetricsTracker:
    def __init__(
        self,
        config: OmegaConf,
        num_parallel_envs: int,
        mode: str,
    ):
        """
        A class to track and calculate metrics during training or evaluation.

        Args:
            config (OmegaConf): Configuration with settings for the metrics.
            num_parallel_envs (int): Number of parallel environments (first batch dimension).
            mode (str): Mode of the metrics tracker, can be "train" or "eval".
        """
        self.config = config
        self.num_parallel_envs = num_parallel_envs

        try:
            self.mode = MetricsTrackerMode(mode)
        except ValueError:
            raise ValueError(
                f"Invalid metrics tracker mode: {mode}. "
                f"Must be one of {[m.value for m in MetricsTrackerMode]}"
            )

        self.log_interval_episodes = self.config.trainer.log_interval_episodes
        self.window_size = self.config.metrics_tracker.moving_avg_window_size
        self.defaults = {
            "rewards": self.config.metrics_tracker.get(
                "moving_avg_reward_default", 0.0
            ),
            "successes": self.config.metrics_tracker.get(
                "moving_avg_success_rate_default", 0.0
            ),
            "lengths": self.config.metrics_tracker.get(
                "moving_avg_length_default", 325
            ),
        }

        # Metrics definitions, used for metrics registration in wandb
        self.metrics_base = [
            "frame",
            "episode",
            "fps",
            "reward",
            "success",
            "length",
        ]
        self.metric_functions: dict[str, Callable] = {
            "moving_avg_reward": lambda: np.mean(self.episode_reward_history),
            "moving_avg_success_rate": lambda: np.mean(self.episode_success_history),
            "moving_avg_length": lambda: np.mean(self.episode_length_history),
        }

    @property
    def step_metric(self) -> str:
        """
        Returns the step metric name for logging to wandb.
        """
        if self.mode == MetricsTrackerMode.TRAIN:
            return f"{self.mode}_episode"
        else:
            return f"{self.mode}_set"

    @property
    def current_metric_prefix(self) -> str:
        """
        Returns the current metric prefix for logging to wandb.
        """
        return f"{self.mode}/{self.current_env_name}"

    def get_metric_names(self) -> list[str]:
        """
        Returns a list of metric names to be logged to wandb.
        The list is different for training and evaluation modes.
        """
        if self.mode == MetricsTrackerMode.TRAIN:
            return self.metrics_base + list(self.metric_functions.keys())
        else:
            # For evaluation, we only log the averaged base metrics
            return [f"avg_{name}" for name in self.metrics_base]

    def initialize(self, env_name: str) -> None:
        """
        Initializes the storage for the new environment.
        """
        self.current_env_name = env_name

        # Total counters
        self.env_total_frames: int = 0
        self.env_total_episodes: int = 0

        # Time tracking for FPS calculation
        self.last_time: float = time.time()
        self.fps: float = 0.0

        # Per-episode accumulators - reset at episode end
        self.current_episode_rewards = np.zeros(
            self.num_parallel_envs, dtype=np.float64
        )
        self.current_episode_lengths = np.zeros(self.num_parallel_envs, dtype=np.int64)

        # Rolling history for moving averages - train mode only
        self.episode_reward_history: deque[float] = deque(
            [self.defaults["rewards"]] * self.window_size, maxlen=self.window_size
        )
        self.episode_success_history: deque[bool] = deque(
            [self.defaults["successes"]] * self.window_size, maxlen=self.window_size
        )
        self.episode_length_history: deque[int] = deque(
            [self.defaults["lengths"]] * self.window_size, maxlen=self.window_size
        )

        # Store model output from the latest batch
        self.model_output_buffer: dict[str, Any] = {}

        # Store metrics to be logged
        self.metrics_buffer: list[dict] = []
        self.metrics_buffer_logging: deque[dict] = deque(
            maxlen=self.log_interval_episodes
        )

        # Episode buffer for visualization (stores pytrees)
        self.episode_stream = [[] for _ in range(self.num_parallel_envs)]
        self.episode_buffer = [[] for _ in range(self.num_parallel_envs)]

    def _add_frame_data_train(self, frame: dict) -> None:
        """Updates metrics for a single frame.

        Args:
            frame: Dict with keys 'reward', 'terminated', 'truncated' (JAX arrays)
        """
        # Convert JAX arrays to numpy for accumulation
        rewards = np.array(frame["reward"])
        self.current_episode_rewards += rewards
        self.current_episode_lengths += 1

        # Store the frame data in the episode buffer
        for idx in range(self.num_parallel_envs):
            # Extract single environment data
            frame_idx = {k: v[idx] if hasattr(v, '__getitem__') else v
                        for k, v in frame.items()}
            self.episode_stream[idx].append(frame_idx)

        # Determine which environments have finished
        terminated = np.array(frame["terminated"])
        truncated = np.array(frame["truncated"])
        finished_envs_mask = terminated | truncated
        finished_indices = np.where(finished_envs_mask)[0]

        for idx in finished_indices:
            self.env_total_episodes += 1

            # Store the finished episode data
            self.episode_buffer[idx] = self.episode_stream[idx].copy()
            self.episode_stream[idx] = []  # Reset the stream for the next episode

            # Extract final episode metrics (cumulative reward, success, length)
            cumulative_reward = self.current_episode_rewards[idx]
            success = int(cumulative_reward > 0)
            length = self.current_episode_lengths[idx]
            self.episode_reward_history.append(cumulative_reward)
            self.episode_success_history.append(success)
            self.episode_length_history.append(length)

            # Build the episode metrics dictionary
            episode_metrics = {
                "frame": self.env_total_frames,
                "episode": self.env_total_episodes,
                "fps": self.fps,
                "reward": cumulative_reward,
                "success": success,
                "length": length,
            }
            for metric_name, metric_func in self.metric_functions.items():
                episode_metrics[metric_name] = metric_func()

            # Store the episode metrics in the buffer to be logged
            self.metrics_buffer.append(episode_metrics)
            self.metrics_buffer_logging.append(episode_metrics)

            # Reset the current episode data
            self.current_episode_rewards[idx] = 0
            self.current_episode_lengths[idx] = 0

    def add_batch_data(self, batch: dict) -> None:
        """
        Updates metrics with batch data from vectorized environment.

        Args:
            batch: Dict with arrays of shape (num_envs, num_frames, ...)
        """
        # Assume batch has shape (num_envs, num_frames, ...)
        # Get batch dimensions from any array in the dict
        sample_array = next(iter(batch.values()))
        if hasattr(sample_array, 'shape') and len(sample_array.shape) >= 2:
            num_envs = sample_array.shape[0]
            num_frames = sample_array.shape[1]
        else:
            raise ValueError("Batch must have at least 2 dimensions (num_envs, num_frames)")

        effective_batch_size = num_envs * num_frames
        self.env_total_frames += effective_batch_size

        # Calculate the time since the last batch
        current_time = time.time()
        elapsed_time = current_time - self.last_time
        self.last_time = current_time
        self.fps = effective_batch_size / elapsed_time

        # Update the metrics for each frame in the batch
        for time_step in range(num_frames):
            # Extract single time step for all environments
            frame = {k: v[:, time_step] if hasattr(v, '__getitem__') else v
                    for k, v in batch.items()}
            self._add_frame_data_train(frame)

    def add_model_data(self, output: dict[str, Any]) -> None:
        """
        Updates the internal buffer with model output data.
        Metrics for now can be any numeric value,
        and will be repeated for all finished episodes in the batch.
        """
        # Filter out non-numeric values, e.g. "actions"
        model_output = {
            k: v for k, v in output.items() if isinstance(v, (float, int, np.number))
        }
        # Update buffers with the model output
        for metrics in self.metrics_buffer:
            metrics.update(model_output)
        for metrics in self.metrics_buffer_logging:
            metrics.update(model_output)

    def get_latest_metrics(self) -> list[dict[str, Any]]:
        """
        Returns accumulated metrics and resets the internal buffer.

        Returns:
            List of metric dictionaries from completed episodes, may be empty.
        """
        metrics_buffer = self.metrics_buffer.copy()
        self.metrics_buffer.clear()
        return metrics_buffer

    @staticmethod
    def _aggregate_metrics(
        buffer: list[dict[str, Any]],
        latest_prefixes: list[str] = None,
        skip_prefixes: list[str] = None,
    ) -> dict[str, Any]:
        """
        Aggregates metrics from a buffer of metric dictionaries.

        Args:
            buffer: List of metric dictionaries to aggregate
            latest_prefixes: List of prefixes for keys to take the latest value from
            skip_prefixes: List of prefixes for keys to skip from aggregation

        Returns:
            A dictionary with aggregated metrics.
        """
        if not buffer:
            logger.warning(
                "No metrics available for aggregation. Returning empty dict."
            )
            return {}

        # Normalize inputs
        latest_prefixes = latest_prefixes or []
        skip_prefixes = skip_prefixes or []

        aggregated_metrics = {}
        all_keys = sorted(buffer[0].keys())

        # Add latest values
        for key in all_keys:
            if any(key.startswith(prefix) for prefix in latest_prefixes):
                aggregated_metrics[key] = buffer[-1][key]

        # Determine keys to aggregate (exclude latest_keys, skip_keys, and keyword matches)
        keys_to_average = [
            key
            for key in all_keys
            if not any(key.startswith(prefix) for prefix in latest_prefixes)
            and not any(key.startswith(prefix) for prefix in skip_prefixes)
        ]

        # Calculate averages
        for key in keys_to_average:
            values = [entry[key] for entry in buffer if key in entry]
            if values:
                aggregated_metrics[f"avg_{key}"] = np.mean(values)

        return aggregated_metrics

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """
        Returns the aggregated metrics from the main metrics buffer.
        Used to calculate the final metrics after evaluation.
        """
        if self.mode != MetricsTrackerMode.EVAL:
            raise ValueError(
                "Aggregated metrics are only available in evaluation mode."
            )

        # Take the latest value for moving averages and skip other metrics
        latest_prefixes = []

        # Skip metrics unrelated to evaluation and moving averages
        # Average metrics over whole evaluation instead
        skip_prefixes = ["frame", "episode", "moving_avg"]

        return self._aggregate_metrics(
            buffer=self.metrics_buffer,
            latest_prefixes=latest_prefixes,
            skip_prefixes=skip_prefixes,
        )

    def get_aggregated_logging_metrics(self) -> dict[str, Any]:
        """
        Returns the aggregated metrics from the logging metrics buffer.
        Used to calculate average metrics for frame based logging during training.
        """
        # Take the latest value if already averaged or does not make sense to average
        latest_prefixes = ["frame", "episode", "fps", "moving_avg"]

        # Skip metrics that have moving averages
        skip_prefixes = ["reward", "success", "length"]

        return self._aggregate_metrics(
            buffer=list(self.metrics_buffer_logging),
            latest_prefixes=latest_prefixes,
            skip_prefixes=skip_prefixes,
        )

    def get_latest_episode_recording(self, env_idx: int) -> list[dict]:
        """
        Returns the latest episode recording for a given environment index.

        Returns:
            List of frame dicts for the episode
        """
        return self.episode_buffer[env_idx]


class GlobalMetricsTracker:
    def __init__(self, config: OmegaConf):
        self.config = config
        self.mode = MetricsTrackerMode.EVAL
        self.envs_dict = {
            env_id: env_name for env_id, env_name in enumerate(self.config.envs)
        }
        self.aggregate_metric_names = self.config.metrics_tracker.aggregate_metric_names
        self.metrics = {}
        # No env_name prefix for logging to terminal or wandb - use "aggregated" instead
        self.current_env_name = "aggregated"
        # Metrics definitions, used for metrics registration in wandb
        self.metrics_base = [
            "frame",
            "episode",
            "fps",
            "reward",
            "success",
            "length",
        ]

    def _get_env_id(self, env_name: str) -> int:
        """
        Maps environment name to its index in the envs_dict.

        Args:
            env_name (str): Name of the environment (i.e "task").

        Returns:
            int: Index of the environment in the envs_dict, or None if not found.
        """
        return next(
            (env_idx for env_idx, name in self.envs_dict.items() if name == env_name),
            None,
        )

    @property
    def step_metric(self) -> str:
        """
        Returns the step metric name for logging to wandb.
        """
        return f"{self.mode}_set"

    @property
    def current_metric_prefix(self) -> str:
        """
        Returns the current metric prefix for logging to wandb - static for global metrics.
        """
        return f"{self.mode}/{self.current_env_name}"

    def get_metric_names(self) -> list[str]:
        """
        Returns a list of metric names to be logged to wandb.
        The list is different for training and evaluation modes.
        """
        metric_names = self.metrics_base.copy()
        metric_names = [f"avg_{name}" for name in metric_names]
        # Add current training environment ID
        metric_names.append("current_training_env_id")
        # Add bt, ft, fg prefixes for backward and forward transfer, forgetting
        for prefix in ["bt_", "ft_", "fg_"]:
            metric_names.extend(
                [f"{prefix}{name}" for name in self.aggregate_metric_names]
            )
        return metric_names

    def add_task_metrics(
        self,
        aggregated_metrics: dict[str, Any],
        set_id: int,
        env_name: str,
    ) -> None:
        """
        Adds evaluated metrics for a task to the global metrics tracker.

        Args:
            aggregated_metrics (dict): Aggregated metrics for the task.
            set_id (int): ID of the evaluation set.
            env_name (str): Name of the environment (i.e "task").
        """
        if set_id not in self.metrics:
            self.metrics[set_id] = {}

        self.metrics[set_id][env_name] = aggregated_metrics

    @staticmethod
    def aggregate_metrics(
        set_metrics: dict[str, dict],
        envs: list[str],
        metric_names: list[str],
        function: Callable,
    ) -> dict[str, float]:
        """
        Aggregates metrics across multiple evaluations and environments.

        Args:
            set_metrics (dict): Dictionary containing metrics for each environment.
            envs (list): List of environment names to aggregate metrics from.
            metric_names (list): List of metric names to aggregate.
            function (Callable): Function to apply for aggregation (e.g., np.mean, np.max).

        Returns:
            dict: Dictionary with aggregated metrics.
        """
        return {
            name: function([set_metrics[env][name] for env in envs])
            for name in metric_names
        }

    def get_aggregated_metrics(
        self, current_set_id: int, current_env_idx: int, current_training_env: str
    ) -> dict[str, Any]:
        """
        Calculates the aggregated metrics for a specific set ID across all environments.

        Args:
            current_set_id (int): ID of the evaluation set.
            current_env_idx (int): Index of the current environment.
            current_training_env (str): Name of the current training environment.

        Returns:
            dict: Aggregated metrics for the specified set ID.
        """
        current_set_metrics = self.metrics[current_set_id]
        current_env_id = self._get_env_id(current_training_env)

        # Get all environments, previous and future environments
        all_envs = [env_name for _, env_name in self.envs_dict.items()]
        previous_envs = [
            env_name
            for env_id, env_name in self.envs_dict.items()
            if env_id < current_env_id
        ]
        future_envs = [
            env_name
            for env_id, env_name in self.envs_dict.items()
            if env_id > current_env_id
        ]

        # Initialize with current training environment idx
        aggregated_metrics = {"current_training_env_id": current_env_idx}

        # Average: all environments, including current training env
        base_metrics = self.aggregate_metrics(
            set_metrics=current_set_metrics,
            envs=all_envs,
            metric_names=self.aggregate_metric_names,
            function=np.mean,
        )
        aggregated_metrics.update(base_metrics)

        # Backward transfer: envs before current training env
        if previous_envs:
            bt_metrics = self.aggregate_metrics(
                set_metrics=current_set_metrics,
                envs=previous_envs,
                metric_names=self.aggregate_metric_names,
                function=np.mean,
            )
            aggregated_metrics.update(
                {f"bt_{name}": value for name, value in bt_metrics.items()}
            )

        # Forward transfer: envs after current training env
        if future_envs:
            ft_metrics = self.aggregate_metrics(
                set_metrics=current_set_metrics,
                envs=future_envs,
                metric_names=self.aggregate_metric_names,
                function=np.mean,
            )
            aggregated_metrics.update(
                {f"ft_{name}": value for name, value in ft_metrics.items()}
            )

        # Forgetting: best performance from previous evaluations minus current performance
        if current_env_id > 0:
            previous_sets_metrics = [
                metrics for sid, metrics in self.metrics.items() if sid < current_set_id
            ]
            for metric in self.aggregate_metric_names:
                deltas = [
                    # Find the best performance in previous sets
                    max(m[env][metric] for m in previous_sets_metrics)
                    # Subtract the current performance
                    - current_set_metrics[env][metric]
                    for env in previous_envs
                ]
                aggregated_metrics[f"fg_{metric}"] = np.mean(deltas)

        return aggregated_metrics
