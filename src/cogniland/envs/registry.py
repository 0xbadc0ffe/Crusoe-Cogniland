"""Environment registry — creates configured env instances."""

from __future__ import annotations

from typing import Any

from cogniland.envs.env import CognilandEnv
from cogniland.envs.multitask_wrapper import MultiTaskEnvWrapper


def make_env(env_id: str, config: Any, train: bool = True) -> MultiTaskEnvWrapper:
    """Create an environment instance from config.

    Args:
        env_id: environment identifier (currently only "cogniland-v0")
        config: full config object (Hydra DictConfig or plain dict)
        train: if True, use train maps; else use val maps

    Returns:
        MultiTaskEnvWrapper wrapping a CognilandEnv
    """
    env_cfg = config.env if hasattr(config, "env") else config.get("env", {})

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

    num_tasks = config.num_tasks if hasattr(config, "num_tasks") else config.get("num_tasks", 1)
    emb_dim = (
        config.task_embedding_dim
        if hasattr(config, "task_embedding_dim")
        else config.get("task_embedding_dim", 7)
    )

    env = CognilandEnv(config, maps_path, int(num_envs))
    return MultiTaskEnvWrapper(env, config, int(num_tasks), int(emb_dim))
