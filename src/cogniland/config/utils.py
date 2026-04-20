"""Configuration loading and sweep support."""

import os
from pathlib import Path

from omegaconf import OmegaConf


def load_config(agent_config_path: str, env_config_path: str) -> OmegaConf:
    """Merge env YAML + agent YAML.  Agent wins on conflicts."""
    env_cfg = OmegaConf.load(env_config_path)
    agent_cfg = OmegaConf.load(agent_config_path)
    cfg = OmegaConf.merge(env_cfg, agent_cfg)
    cfg.name = f"{Path(env_config_path).stem}_{Path(agent_config_path).stem}"
    cfg.pid = os.getpid()
    return cfg


def configure_sweep_config(
    base_config: OmegaConf, sweep_config_dict: dict
) -> OmegaConf:
    """Apply W&B sweep overrides to the base config.

    Uses ``OmegaConf.update`` with the raw Python values from ``run.config``
    so that ``None``/``bool``/``int``/``float`` are preserved. The prior
    dotlist approach stringified values (``f"{k}={v}"``), which turned
    ``None`` into the literal string ``"None"`` — surprising downstream code
    that expected a real ``None``.
    """
    cfg = OmegaConf.create(base_config)
    for k, v in sweep_config_dict.items():
        OmegaConf.update(cfg, k, v, merge=True)
    return cfg
