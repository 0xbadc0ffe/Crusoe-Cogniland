"""Metrics tracking for training and evaluation."""

import time
from collections import deque
from enum import Enum

import numpy as np
from omegaconf import OmegaConf


class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"


class MetricsTracker:
    def __init__(self, config: OmegaConf, num_parallel_envs: int, mode: str):
        self.config = config
        self.mode = Mode(mode)
        self.num_parallel_envs = num_parallel_envs
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
