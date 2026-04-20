"""Metrics tracking for training and evaluation."""

import time
from collections import defaultdict, deque
from enum import Enum

import numpy as np
from omegaconf import OmegaConf


class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"


class MetricsTracker:
    def __init__(
        self,
        config: OmegaConf,
        num_parallel_envs: int,
        mode: str,
        num_tasks: int = 1,
    ):
        self.config = config
        self.mode = Mode(mode)
        self.num_parallel_envs = num_parallel_envs
        self.num_tasks = int(num_tasks)
        self.window_size = config.metrics_tracker.moving_avg_window_size

        self.metrics_base = ["frame", "episode", "fps", "reward", "success", "length"]
        self.metric_functions = {
            "moving_avg_reward": lambda: float(np.mean(self.episode_reward_history)),
            "moving_avg_success_rate": lambda: float(
                np.mean(self.episode_success_history)
            ),
            "moving_avg_length": lambda: float(np.mean(self.episode_length_history)),
        }

    @property
    def step_metric(self) -> str:
        return "train_episode" if self.mode == Mode.TRAIN else "eval_set"

    @property
    def metric_prefix(self) -> str:
        return self.mode.value

    def get_metric_names(self) -> list[str]:
        if self.mode == Mode.TRAIN:
            return self.metrics_base + list(self.metric_functions.keys())
        return [f"avg_{n}" for n in self.metrics_base]

    def initialize(self):
        self.env_total_frames = 0
        self.env_total_episodes = 0
        self.fps = 0.0
        self.last_time = time.time()
        self.episode_reward_history = deque(
            [0.0] * self.window_size, maxlen=self.window_size
        )
        self.episode_success_history = deque(
            [0.0] * self.window_size, maxlen=self.window_size
        )
        self.episode_length_history = deque(
            [0] * self.window_size, maxlen=self.window_size
        )

        # Per-task rolling histories (empty deques — we don't seed with zeros
        # because absent data should not skew the mean toward 0 before any
        # episode of that task has finished).
        self.per_task_reward_history = {
            t: deque(maxlen=self.window_size) for t in range(self.num_tasks)
        }
        self.per_task_success_history = {
            t: deque(maxlen=self.window_size) for t in range(self.num_tasks)
        }
        self.per_task_length_history = {
            t: deque(maxlen=self.window_size) for t in range(self.num_tasks)
        }
        self.per_task_total_episodes = {t: 0 for t in range(self.num_tasks)}

        # Per-biome rolling histories. Biome strings are discovered at
        # runtime from the env's ``info['biome']``, so we use defaultdicts
        # seeded with empty deques sized to ``window_size``.
        _w = self.window_size
        self.per_biome_reward_history = defaultdict(lambda: deque(maxlen=_w))
        self.per_biome_success_history = defaultdict(lambda: deque(maxlen=_w))
        self.per_biome_length_history = defaultdict(lambda: deque(maxlen=_w))
        self.per_biome_total_episodes = defaultdict(int)
