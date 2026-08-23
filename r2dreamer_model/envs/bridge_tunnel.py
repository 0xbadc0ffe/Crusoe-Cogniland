"""r2dreamer wrapper for cogniland's bridge_tunnel fork_wall task (BT-rules).

Patterned after :mod:`envs.memory`. ``cogniland.bridge_tunnel`` only needs
numpy + gymnasium, so this stays importable in the lightweight r2dreamer env
(PYTHONPATH must include the repo's ``src``).

Observation: the env's native ``{minimap: (V,V) int8, scalars: (n,) float32}``
dict is flattened into a single one-hot ``vector`` key (minimap one-hot over
tile ids, concatenated with scalars) -- the same encoding the pure-JAX
DreamerV3 baseline in this repo already uses successfully for this env (no
CNN; the minimap is symbolic, not a natural image). ``configs/env/*.yaml``
must route it through the MLP encoder (``encoder.mlp_keys: 'vector'``,
``cnn_keys: '$^'``).

``log_success`` follows the same convention as ``envs/metaworld.py``: 1.0 on
the step the correct door is reached, 0.0 otherwise, summed+clipped by the
trainer's eval loop into a per-episode success rate visible in TensorBoard
without a separate offline eval pass.

Task names (routed by ``config.task`` in ``envs/__init__.py``, suite
``bridgetunnel``):

  bridgetunnel_forkwall -> btc rules, commitment DISABLED (BT-rules fork_wall),
                           fork_wall=True, categories balanced/lakes/rocky.
                           Mirrors configs/bridge_tunnel/btc_ppo_forkwall_nocommit.yaml
                           (released PPO: 100% success).
"""
import os

import gymnasium as gym
import numpy as np

from cogniland.bridge_tunnel.env import BridgeTunnelEnv
from cogniland.bridge_tunnel.tiles import NUM_TILES


def _env_overrides() -> dict:
    """Per-run BridgeTunnelEnv overrides from BT_* env vars (experiment arms
    without touching code defaults), mirroring envs/memory.py's convention."""
    ov = {}
    g = os.environ.get
    for var, field, cast in (
        ("BT_REACH_BONUS", "reach_bonus", float),
        ("BT_SHAPING_COEF", "shaping_coef", float),
        ("BT_SLACK_PENALTY", "slack_penalty", float),
        ("BT_ILLEGAL_PENALTY", "illegal_penalty", float),
        ("BT_MAX_STEPS", "max_steps", int),
    ):
        if g(var) is not None:
            ov[field] = cast(g(var))
    return ov


# task-name -> BridgeTunnelEnv kwargs (mirrors configs/bridge_tunnel/*.yaml)
_TASKS = {
    "forkwall": dict(
        variant="btc", commit=False, fork_wall=True,
        categories=("balanced", "lakes", "rocky"),
        passage_half=1, wall_margin=1, mem_gap=16, shaping_gamma=1.0,
        size=32, width=64, view_size=21, max_steps=800,
        orientation="natural", tree_frac=0.03, goal_half=0,
        slack_penalty=-0.01, shaping_coef=0.015, reach_bonus=3.0,
        build_cost=0.0, commit_cost=0.05, illegal_penalty=0.02,
        gamma=0.99,
    ),
}


class BridgeTunnel(gym.Env):
    def __init__(self, task, size=None, seed=0):
        if task not in _TASKS:
            raise ValueError(
                f"unknown bridgetunnel task {task!r}; expected one of {sorted(_TASKS)}"
            )
        kwargs = dict(_TASKS[task])
        kwargs.update(_env_overrides())
        self._seed = int(seed)
        self._env = BridgeTunnelEnv(seed=self._seed, **kwargs)
        self._task = task
        # FIXED-MAP MODE: if BT_MAPS points at a pickled MapRecord pool, draw one
        # map per episode from it (all models train on the SAME dataset) instead
        # of generating a fresh procedural map per seed. Each worker samples with
        # its own seeded RNG so the stream is reproducible + well-mixed.
        self._pool = None
        _maps = os.environ.get("BT_MAPS")
        if _maps:
            from cogniland.bridge_tunnel.map_pool import MapPool
            self._pool = MapPool(_maps)
            self._pool_rng = np.random.default_rng(1000 + self._seed)
        self._view = int(self._env.view_size)
        self._n_scalars = int(self._env.n_scalars)
        self._vec_dim = self._view * self._view * NUM_TILES + self._n_scalars

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "vector": gym.spaces.Box(-np.inf, np.inf, (self._vec_dim,), np.float32),
            "log_success": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        })

    @property
    def action_space(self):
        return gym.spaces.Discrete(self._env.action_space.n)

    def _flatten(self, raw_obs):
        minimap = np.asarray(raw_obs["minimap"], dtype=np.int64)
        onehot = np.zeros((self._view, self._view, NUM_TILES), dtype=np.float32)
        rr, cc = np.indices((self._view, self._view))
        onehot[rr, cc, minimap] = 1.0
        return np.concatenate([
            onehot.reshape(-1),
            np.asarray(raw_obs["scalars"], dtype=np.float32),
        ])

    def _obs(self, raw_obs, success, is_first, is_last, is_terminal):
        return {
            "vector": self._flatten(raw_obs),
            "log_success": float(success),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    @staticmethod
    def _clean_info(info):
        return {k: v for k, v in info.items() if not k.startswith("_")}

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self._env.step(int(action))
        done = bool(terminated or truncated)
        obs = self._obs(raw_obs, info.get("reached_target", False),
                        is_first=False, is_last=done, is_terminal=bool(terminated))
        return obs, float(reward), done, self._clean_info(info)

    def reset(self):
        if self._pool is not None:
            self._env._fixed_record = self._pool.sample(self._pool_rng)
            raw_obs, _info = self._env.reset()
        else:
            raw_obs, _info = self._env.reset(seed=self._seed)
            self._seed += 1
        return self._obs(raw_obs, False, is_first=True, is_last=False, is_terminal=False)
