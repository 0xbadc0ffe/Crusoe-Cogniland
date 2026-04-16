import os
from pathlib import Path
from omegaconf import OmegaConf


def load_config(agent_config_path: str, env_config_path: str) -> OmegaConf:
    env_cfg = OmegaConf.load(env_config_path)
    agent_cfg = OmegaConf.load(agent_config_path)
    cfg = OmegaConf.merge(env_cfg, agent_cfg)

    cfg.name = f"{Path(env_config_path).stem}_{Path(agent_config_path).stem}"
    cfg.pid = os.getpid()
    return cfg


def configure_sweep_config(base_config: OmegaConf, sweep_config_dict: dict) -> OmegaConf:
    dotlist = [f"{k}={v}" for k, v in sweep_config_dict.items()]
    return OmegaConf.merge(base_config, OmegaConf.from_dotlist(dotlist))
