"""r2dreamer wrapper for cogniland's MemoryEnv (MiniGrid memory task).

Patterned after :mod:`envs.memorymaze`. Wraps
``cogniland.memory_env.make_memory_env``, resizes the native (56,56,3) RGB
partial view up to the standard Dreamer CNN input (64,64,3) with crisp
nearest-neighbour scaling, and exposes the gym Dict-obs contract the trainer
expects (``image`` + ``is_first``/``is_last``/``is_terminal``).

The training cue subset is selected from the r2dreamer task name:

  memory_2cue -> custom cues ["green_up", "blue_down"]            (entangled)
  memory_3cue -> custom cues ["green_up", "green_down", "blue_down"] (partial)
  memory_4cue -> cue_distribution="factorized"                   (all four)

``cogniland.memory_env`` only needs numpy + gymnasium + minigrid, so this
module can be imported in the lightweight r2dreamer env without pulling in the
rest of cogniland's heavy dependencies (set PYTHONPATH to the repo's ``src``).
"""
import os

import gymnasium as gym
import numpy as np

from cogniland.memory_env import make_memory_env, MemoryEnvConfig


def _env_overrides() -> dict:
    """Per-run MemoryEnvConfig overrides from MEMENV_* env vars (for experiment
    arms — reward shaping / branch gating — without changing code defaults)."""
    ov = {}
    g = os.environ.get
    for var, field, cast in (
        ("MEMENV_BRANCH_BONUS", "branch_bonus", float),
        ("MEMENV_SUCCESS_REWARD", "success_reward", float),
        ("MEMENV_WRONG_BRANCH_PENALTY", "wrong_branch_penalty", float),
        ("MEMENV_WRONG_DOOR_REWARD", "wrong_door_reward", float),
    ):
        if g(var) is not None:
            ov[field] = cast(g(var))
    for var, field in (("MEMENV_WRONG_BRANCH_TERMINATES", "wrong_branch_terminates"),
                       ("MEMENV_SUCCESS_REQUIRES_BRANCH", "success_requires_branch")):
        v = g(var)
        if v is not None:
            ov[field] = v.lower() in ("1", "true", "yes")
    return ov


# task-name -> MemoryEnvConfig cue settings (mirrors scripts/memory_env/datasets.py)
_CUE_SUBSETS = {
    "2cue": dict(cue_distribution="custom", custom_cues=["green_up", "blue_down"]),
    "3cue": dict(cue_distribution="custom",
                 custom_cues=["green_up", "green_down", "blue_down"]),
    "4cue": dict(cue_distribution="factorized"),
}


def _resize_nn(img, size):
    """Nearest-neighbour resize of an (H,W,3) uint8 array to ``size``=(H2,W2)."""
    h, w = img.shape[:2]
    h2, w2 = size
    if (h, w) == (h2, w2):
        return img
    ys = (np.arange(h2) * h // h2).clip(0, h - 1)
    xs = (np.arange(w2) * w // w2).clip(0, w - 1)
    return img[ys][:, xs]


class Memory(gym.Env):
    def __init__(self, task, size=(64, 64), seed=0):
        if task not in _CUE_SUBSETS:
            raise ValueError(
                f"unknown memory task {task!r}; expected one of {sorted(_CUE_SUBSETS)}"
            )
        cfg = MemoryEnvConfig(**_CUE_SUBSETS[task], **_env_overrides())
        self._env = make_memory_env(cfg)
        self._size = tuple(size)
        self._seed = int(seed)
        self._task = task

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        img_shape = self._size + (3,)
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        })

    @property
    def action_space(self):
        return gym.spaces.Discrete(self._env.action_space.n)

    def _obs(self, image, is_first, is_last, is_terminal):
        image = _resize_nn(np.asarray(image, dtype=np.uint8), self._size)
        return {
            "image": image,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    @staticmethod
    def _clean_info(info):
        # Drop the private minigrid handle (``_mg``) — it holds unpicklable
        # closures, and ParallelEnv pickles the step/reset result across the
        # worker process boundary. Keep only picklable scalar task labels.
        return {k: v for k, v in info.items() if not k.startswith("_")}

    def step(self, action):
        image, reward, terminated, truncated, info = self._env.step(int(action))
        done = bool(terminated or truncated)
        obs = self._obs(image, is_first=False, is_last=done, is_terminal=bool(terminated))
        return obs, float(reward), done, self._clean_info(info)

    def reset(self):
        image, _info = self._env.reset(seed=self._seed)
        # advance the seed so successive episodes in this worker differ while
        # staying deterministic and well below the held-out test range.
        self._seed += 1
        return self._obs(image, is_first=True, is_last=False, is_terminal=False)
