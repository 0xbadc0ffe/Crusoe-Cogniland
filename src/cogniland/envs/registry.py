"""Environment registry — creates configured env instances."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from cogniland.envs.env import CognilandEnv
from cogniland.envs.multitask_wrapper import MultiTaskEnvWrapper


def make_env(env_id: str, config: Any, train: bool = True):
    """Create an environment instance from config.

    Args:
        env_id: environment identifier (currently only "cogniland-v0")
        config: full config object (Hydra DictConfig or plain dict)
        train: if True, use train maps; else use val maps

    Returns:
        MultiTaskEnvWrapper wrapping a CognilandEnv
    """
    env_cfg = config.env if hasattr(config, "env") else config.get("env", {})

    if env_id in ("cogniland-jax", "cogniland-jax-v0"):
        # Use aliased names so the legacy ``CognilandEnv`` import at module
        # level isn't shadowed by a local binding when this branch is
        # parsed on the non-jax code path.
        from cogniland_jax import CognilandEnv as JaxCognilandEnv
        from cogniland_jax import EnvParams as JaxEnvParams
        from cogniland_jax import constants as C
        from cogniland_jax.batched import JaxBatchedEnv
        from cogniland_jax.maps import load_map_arrays

        if train:
            maps_path = env_cfg.get("train_maps", "data/maps/train.pt") \
                if isinstance(env_cfg, dict) else getattr(env_cfg, "train_maps", "data/maps/train.pt")
            num_envs = env_cfg.get("num_parallel_envs", 128) \
                if isinstance(env_cfg, dict) else getattr(env_cfg, "num_parallel_envs", 128)
        else:
            maps_path = env_cfg.get("val_maps", "data/maps/val.pt") \
                if isinstance(env_cfg, dict) else getattr(env_cfg, "val_maps", "data/maps/val.pt")
            num_envs = env_cfg.get("num_parallel_envs_eval", 16) \
                if isinstance(env_cfg, dict) else getattr(env_cfg, "num_parallel_envs_eval", 16)

        biome_filter = getattr(env_cfg, "biome_filter", None) \
            if not isinstance(env_cfg, dict) else env_cfg.get("biome_filter", None)
        if biome_filter is not None and hasattr(biome_filter, "__iter__"):
            biome_filter = list(biome_filter)

        difficulty_map = {"easy": 0, "medium": 1, "hard": 2}
        diff_raw = getattr(env_cfg, "difficulty", "hard") \
            if not isinstance(env_cfg, dict) else env_cfg.get("difficulty", "hard")
        difficulty = difficulty_map.get(str(diff_raw), C.DIFFICULTY_HARD)

        reward_cfg = config.reward if hasattr(config, "reward") else config.get("reward", {})
        def _rget(k, default):
            return reward_cfg.get(k, default) if isinstance(reward_cfg, dict) \
                else getattr(reward_cfg, k, default)
        arrays = load_map_arrays(maps_path, biome_filter=biome_filter)

        params = JaxEnvParams.from_map_arrays(
            **arrays,
            max_steps=jnp.int32(int(getattr(env_cfg, "max_steps", 1000)
                if not isinstance(env_cfg, dict) else env_cfg.get("max_steps", 1000))),
            reach_bonus=jnp.float32(float(_rget("reach_bonus", 150.0))),
            step_penalty=jnp.float32(float(_rget("step_penalty", 0.02))),
            shaping_coef=jnp.float32(float(_rget("shaping_coef", 0.3))),
            hp_coef=jnp.float32(float(_rget("hp_coef", 0.06))),
            death_penalty=jnp.float32(float(_rget("death_penalty", 0.0))),
            difficulty=jnp.int32(int(difficulty)),
        )
        env = JaxCognilandEnv(default_params=params)
        seed = config.seed if hasattr(config, "seed") else config.get("seed", 42)
        agent_cfg = config.agent if hasattr(config, "agent") else config.get("agent", {})
        gamma = getattr(agent_cfg, "gamma", 0.99) if not isinstance(agent_cfg, dict) \
            else agent_cfg.get("gamma", 0.99)
        return JaxBatchedEnv(
            env=env, params=params,
            num_envs=int(num_envs), seed=int(seed) + (0 if train else 1),
            gamma=float(gamma),
        )

    if env_id.startswith("minigrid"):
        from cogniland.envs.minigrid_env import BatchedMiniGridDoorKeyEnv

        if train:
            num_envs = (
                env_cfg.num_parallel_envs
                if hasattr(env_cfg, "num_parallel_envs")
                else env_cfg.get("num_parallel_envs", 32)
            )
        else:
            num_envs = (
                env_cfg.num_parallel_envs_eval
                if hasattr(env_cfg, "num_parallel_envs_eval")
                else env_cfg.get(
                    "num_parallel_envs_eval",
                    env_cfg.get("num_parallel_envs", 32),
                )
            )
        gym_id = (
            env_cfg.gym_id
            if hasattr(env_cfg, "gym_id")
            else env_cfg.get("gym_id", "MiniGrid-DoorKey-6x6-v0")
        )
        return BatchedMiniGridDoorKeyEnv(
            config, int(num_envs), env_id=gym_id,
        )

    if train:
        if hasattr(env_cfg, "train_maps"):
            maps_path = env_cfg.train_maps
        else:
            maps_path = env_cfg.get("train_maps", "data/maps/train.pt")
    else:
        if hasattr(env_cfg, "val_maps"):
            maps_path = env_cfg.val_maps
        else:
            maps_path = env_cfg.get("val_maps", "data/maps/val.pt")

    if train:
        if hasattr(env_cfg, "num_parallel_envs"):
            num_envs = env_cfg.num_parallel_envs
        else:
            num_envs = env_cfg.get("num_parallel_envs", 32)
    else:
        if hasattr(env_cfg, "num_parallel_envs_eval"):
            num_envs = env_cfg.num_parallel_envs_eval
        elif isinstance(env_cfg, dict):
            num_envs = env_cfg.get(
                "num_parallel_envs_eval",
                env_cfg.get("num_parallel_envs", 32),
            )
        else:
            num_envs = getattr(env_cfg, "num_parallel_envs", 32)

    env = CognilandEnv(config, maps_path, int(num_envs))
    return MultiTaskEnvWrapper(env, config)
