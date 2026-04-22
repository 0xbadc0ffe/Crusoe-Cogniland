"""Metrics tracking for training and evaluation.

Train path: holds lightweight counters (episode count, fps) and is the
authoritative place for episode totals. All per-episode values are logged
to W&B as raw scalars — no moving averages are maintained here; W&B's UI
smoothing handles that.

Eval path: accumulates every finished episode from one evaluation pass into
flat lists and exposes ``avg_*`` aggregates.
"""

import time
from collections import deque
from enum import Enum

import numpy as np
from omegaconf import OmegaConf

# Moving-average window (# of finished train episodes) used to smooth the
# noisy {0,1} success signal before logging.
TRAIN_SUCCESS_MA_WINDOW = 50


class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"


class MetricsTracker:
    def __init__(
        self,
        config: OmegaConf,
        num_parallel_envs: int,
        mode: str,
    ):
        self.config = config
        self.mode = Mode(mode)
        self.num_parallel_envs = num_parallel_envs

        # Train: per-episode scalars emitted as raw values plus a moving-average
        # success (noisy {0,1} otherwise). Eval aggregates these into means
        # across all episodes observed in one eval set.
        self.metrics_base = ["fps", "reward", "reward_discounted", "success", "length"]

    @property
    def step_metric(self) -> str:
        return "train_episode" if self.mode == Mode.TRAIN else "eval_set"

    @property
    def metric_prefix(self) -> str:
        return self.mode.value

    def get_metric_names(self) -> list[str]:
        if self.mode == Mode.TRAIN:
            return list(self.metrics_base)
        return [f"avg_{n}" for n in ("reward", "success", "length")] + ["episodes"]

    def initialize(self):
        self.env_total_frames = 0
        self.env_total_episodes = 0
        self.fps = 0.0
        self.last_time = time.time()

        # Eval buffers: every finished episode in one eval set lands here;
        # the trainer reduces them to means at log time. Train mode keeps
        # them too so legacy code that pushes into them (e.g. the trajectory
        # logger) keeps working, but the train path does not read them back.
        self.episode_reward_history: list[float] = []
        self.episode_length_history: list[int] = []
        self.episode_success_history: list[int] = []

        # Train-only rolling window for the success moving average.
        self.train_success_window: deque[int] = deque(
            maxlen=TRAIN_SUCCESS_MA_WINDOW
        )
