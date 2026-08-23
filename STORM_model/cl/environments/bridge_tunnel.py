"""bridge_tunnel adapter -- fork_wall (BT-rules) task, for the STORM/DreamerV3
world-model agents in this repo.

Wraps the numpy ``cogniland.bridge_tunnel.env.BridgeTunnelEnv`` (fork_wall=True,
commit=False) unmodified -- the pure-JAX port of bridge_tunnel doesn't have
fork_wall yet, so this reuses the exact env the released PPO fork_wall
no-commit agent was trained on (100% held-out success,
``configs/bridge_tunnel/btc_ppo_forkwall_nocommit.yaml``) and that
``r2dreamer_model/envs/bridge_tunnel.py`` already trains a Dreamer baseline
on. Episodes are drawn from the SAME fixed map pool
(``data/bridge_tunnel/forkwall6k/{train,test}.pkl``) so PPO / Dreamer / STORM
are compared on identical data, not independently-sampled procedural streams.

Observation: the native ``{minimap: (V,V) int8, scalars: (n,) float32}`` dict
is flattened into one one-hot ``vector`` key (minimap one-hot over tile ids,
concatenated with scalars) -- no CNN, the minimap is symbolic (same convention
used by the in-tree pure-JAX DreamerV3 baseline and by r2dreamer's wrapper).

Not JAX-jittable: the underlying env is plain numpy (scipy/opensimplex
mapgen), so this is stepped with a Python loop across ``num_envs`` independent
envs rather than ``jax.vmap``. To slot into the STORM/DreamerV3 agent's
training loop (which expects the Navix/Craftax calling convention:
``env.reset(rngs) -> state`` and ``env.step(state, action) -> state``, with
``state.env_state.{observation,reward,is_done(),is_termination(),t}`` and
outer ``state.{returned_episode_returns,returned_episode_lengths,timestep}``)
the auto-reset semantics mirror Navix's own ``Environment.step``: whether a
sub-env is stepped for real or reset (ignoring the given action) is decided by
that sub-env's OWN incoming ``done`` flag (from the state passed in), not by
a flag mutated as a side effect inside a single step call. That is what makes
``is_first = (t == 0) & (~done)`` correct: a fresh reset frame always has
``t=0, done=False``, and the one frame where ``done=True`` is exactly the true
terminal observation (not yet reset).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from cl.environments import register_environment

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from cogniland.bridge_tunnel.env import BridgeTunnelEnv  # noqa: E402
from cogniland.bridge_tunnel.map_pool import MapPool  # noqa: E402
from cogniland.bridge_tunnel.tiles import NUM_TILES  # noqa: E402

# Mirrors r2dreamer_model/envs/bridge_tunnel.py::_TASKS["forkwall"] and
# configs/bridge_tunnel/btc_ppo_forkwall_nocommit.yaml (released PPO: 100% success).
FORKWALL_KWARGS: Dict[str, Any] = dict(
    variant="btc", commit=False, fork_wall=True,
    categories=("balanced", "lakes", "rocky"),
    passage_half=1, wall_margin=1, mem_gap=16, shaping_gamma=1.0,
    size=32, width=64, view_size=21, max_steps=800,
    orientation="natural", tree_frac=0.03, goal_half=0,
    slack_penalty=-0.01, shaping_coef=0.015, reach_bonus=3.0,
    build_cost=0.0, commit_cost=0.05, illegal_penalty=0.02,
    gamma=0.99,
)

# Small single-target diagnostic task (no fork_wall, no MapPool -- each env
# just generates its own fresh procedural map every reset via BridgeTunnelEnv's
# own seed). Used to isolate "STORM/bridge_tunnel wiring is broken" from
# "fork_wall specifically is hard": run 1 (entropy 3e-4) and run 2 (entropy
# 0.01) both got exactly 0% success / 0 training successes on fork_wall.
EASY_KWARGS: Dict[str, Any] = dict(
    variant="bt", commit=False, fork_wall=False,
    size=16, width=32, view_size=21, max_steps=300,
    orientation="natural", tree_frac=0.03, goal_half=2,
    water_frac=0.14, rock_frac=0.14,
    slack_penalty=-0.01, shaping_coef=0.015, reach_bonus=3.0,
    build_cost=0.0, commit_cost=0.05, illegal_penalty=0.02,
    gamma=0.99, shaping_gamma=1.0,
)

# task name (env_name suffix after "/") -> (env kwargs, use a fixed MapPool?)
TASKS: Dict[str, tuple] = {
    "forkwall": (FORKWALL_KWARGS, True),
    "easy": (EASY_KWARGS, False),
}

_DEFAULT_MAPS_PATH = "data/bridge_tunnel/forkwall6k/train.pkl"


def _resolve_maps_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_REPO_ROOT / path)


class _Timestep:
    """Mimics Navix's ``Timestep`` surface (``.observation/.reward/.t``,
    ``.is_done()/.is_termination()``) -- the only attributes the STORM/DreamerV3
    agent code actually reads off ``state.env_state``."""

    __slots__ = ("t", "observation", "reward", "done", "terminal")

    def __init__(self, t, observation, reward, done, terminal):
        self.t = t
        self.observation = observation
        self.reward = reward
        self.done = done
        self.terminal = terminal

    def is_done(self):
        return self.done

    def is_termination(self):
        return self.terminal


class _LogState:
    """Mimics the Navix/purejaxwm ``LogEnvState`` wrapper surface."""

    __slots__ = (
        "env_state", "episode_returns", "episode_lengths",
        "returned_episode_returns", "returned_episode_lengths", "timestep",
    )

    def __init__(self, env_state, episode_returns, episode_lengths,
                 returned_episode_returns, returned_episode_lengths, timestep):
        self.env_state = env_state
        self.episode_returns = episode_returns
        self.episode_lengths = episode_lengths
        self.returned_episode_returns = returned_episode_returns
        self.returned_episode_lengths = returned_episode_lengths
        self.timestep = timestep


class BridgeTunnelVectorized:
    """fork_wall (BT-rules) task, vectorized over ``num_envs`` independent
    numpy envs each drawing episodes from a shared fixed map pool."""

    def __init__(self, env_name: str, config: OmegaConf, num_envs: int):
        self.env_name = env_name
        self.num_envs = int(num_envs)

        task_name = env_name.split("/")[-1] if "/" in env_name else "forkwall"
        task_kwargs, use_pool = TASKS.get(task_name, (FORKWALL_KWARGS, True))
        base_seed = int(config.get("seed", 0))

        self._use_pool = use_pool
        if use_pool:
            maps_path = _resolve_maps_path(config.env.get("maps_path", _DEFAULT_MAPS_PATH))
            self._pool = MapPool(maps_path)
            self._pool_rngs = [
                np.random.default_rng(10_000 + base_seed + i) for i in range(self.num_envs)
            ]
        else:
            self._next_seed = [base_seed + i * 1_000_003 for i in range(self.num_envs)]

        self._envs = [
            BridgeTunnelEnv(seed=base_seed + i, **task_kwargs)
            for i in range(self.num_envs)
        ]

        self._view = int(self._envs[0].view_size)
        self._n_scalars = int(self._envs[0].n_scalars)
        self._vec_dim = self._view * self._view * NUM_TILES + self._n_scalars

    def action_space(self) -> int:
        return 6

    def observation_space(self) -> Dict[str, tuple]:
        return {"vector": (self._vec_dim,)}

    def _flatten(self, raw_obs) -> np.ndarray:
        minimap = np.asarray(raw_obs["minimap"], dtype=np.int64)
        onehot = np.zeros((self._view, self._view, NUM_TILES), dtype=np.float32)
        rr, cc = np.indices((self._view, self._view))
        onehot[rr, cc, minimap] = 1.0
        return np.concatenate([
            onehot.reshape(-1),
            np.asarray(raw_obs["scalars"], dtype=np.float32),
        ])

    def _reset_one(self, i: int) -> np.ndarray:
        if self._use_pool:
            self._envs[i]._fixed_record = self._pool.sample(self._pool_rngs[i])
            raw_obs, _info = self._envs[i].reset()
        else:
            raw_obs, _info = self._envs[i].reset(seed=self._next_seed[i])
            self._next_seed[i] += 1
        return self._flatten(raw_obs)

    def reset(self, rngs: Optional[Any] = None) -> _LogState:
        obs = np.stack([self._reset_one(i) for i in range(self.num_envs)])
        timestep = _Timestep(
            t=jnp.zeros(self.num_envs, dtype=jnp.int32),
            observation={"vector": jnp.asarray(obs)},
            reward=jnp.zeros(self.num_envs, dtype=jnp.float32),
            done=jnp.zeros(self.num_envs, dtype=bool),
            terminal=jnp.zeros(self.num_envs, dtype=bool),
        )
        zf = jnp.zeros(self.num_envs, dtype=jnp.float32)
        zi = jnp.zeros(self.num_envs, dtype=jnp.int32)
        return _LogState(timestep, zf, zi, zf, zi, zi)

    def step(self, state: _LogState, action) -> _LogState:
        prev_done = np.asarray(state.env_state.done)
        prev_t = np.asarray(state.env_state.t)
        action_np = np.asarray(action)

        obs = np.zeros((self.num_envs, self._vec_dim), dtype=np.float32)
        reward = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        terminal = np.zeros(self.num_envs, dtype=bool)
        t_new = np.zeros(self.num_envs, dtype=np.int32)

        for i in range(self.num_envs):
            if prev_done[i]:
                # incoming state was already terminal -> reset (action is a
                # throwaway, matches Navix's `should_reset = step_type > 0`)
                obs[i] = self._reset_one(i)
            else:
                raw_obs, r, terminated, truncated, _info = self._envs[i].step(int(action_np[i]))
                obs[i] = self._flatten(raw_obs)
                reward[i] = r
                done[i] = bool(terminated or truncated)
                terminal[i] = bool(terminated)
                t_new[i] = prev_t[i] + 1

        timestep = _Timestep(
            t=jnp.asarray(t_new),
            observation={"vector": jnp.asarray(obs)},
            reward=jnp.asarray(reward),
            done=jnp.asarray(done),
            terminal=jnp.asarray(terminal),
        )

        prev_ep_ret = np.asarray(state.episode_returns)
        prev_ep_len = np.asarray(state.episode_lengths)
        prev_ret_ret = np.asarray(state.returned_episode_returns)
        prev_ret_len = np.asarray(state.returned_episode_lengths)
        prev_global_t = np.asarray(state.timestep)

        new_ep_ret = prev_ep_ret + reward
        new_ep_len = prev_ep_len + 1
        done_f = done.astype(np.float32)
        done_i = done.astype(np.int32)

        new_state = _LogState(
            env_state=timestep,
            episode_returns=jnp.asarray(new_ep_ret * (1 - done_f)),
            episode_lengths=jnp.asarray(new_ep_len * (1 - done_i)),
            returned_episode_returns=jnp.asarray(
                prev_ret_ret * (1 - done_f) + new_ep_ret * done_f
            ),
            returned_episode_lengths=jnp.asarray(
                prev_ret_len * (1 - done_i) + new_ep_len * done_i
            ),
            timestep=jnp.asarray(prev_global_t + 1),
        )
        return new_state


@register_environment("BridgeTunnel")
def make_bridge_tunnel_env(env_name: str, config: OmegaConf) -> BridgeTunnelVectorized:
    """Factory for bridge_tunnel tasks (see ``TASKS``).

    Args:
        env_name: e.g. ``"BridgeTunnel/forkwall"`` or ``"BridgeTunnel/easy"``.
        config: OmegaConf configuration (``config.env.num_parallel_envs``,
            optionally ``config.env.maps_path`` for pool-backed tasks).
    """
    num_envs = config.env.num_parallel_envs
    return BridgeTunnelVectorized(env_name=env_name, config=config, num_envs=num_envs)
