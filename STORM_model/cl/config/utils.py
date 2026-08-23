import os
from pathlib import Path

from omegaconf import OmegaConf

from cl.shared import setup_logger

logger = setup_logger(__name__)


def load_config(agent_config_path: str, env_config_path: str) -> OmegaConf:
    """
    Load a configuration from an agent-specific YAML file and optionally merge it
    with a base environment configuration.

    Args:
        agent_config_path (str): Path to the agent configuration YAML file.
        env_config_path (str): Path to a base environment configuration
            YAML file. Values from the agent config will override the env config.

    Returns:
        OmegaConf: The loaded and possibly merged configuration.
    """
    agent_config = OmegaConf.load(agent_config_path)
    env_config = OmegaConf.load(env_config_path)

    # Keys from the agent config will override the environment config
    config = OmegaConf.merge(env_config, agent_config)

    # Set config name and process ID
    agent_config_name = Path(agent_config_path).stem
    env_config_name = Path(env_config_path).stem
    config.name = f"{env_config_name}_{agent_config_name}"
    config.pid = os.getpid()

    return config


def configure_sweep_config(
    base_config: OmegaConf, sweep_config_dict: dict
) -> OmegaConf:
    """
    Combine a base OmegaConf (loaded from YAML) with a flat dict from
    `wandb.config, and return the merged OmegaConf. Log the parameters
    overriden by the sweep config.
    Args:
        base_config (OmegaConf):
            The configuration lodaed from the YAML file
        sweep_config (dict):
            The flat dictionary coming from `wandb.config`

    Returns:
        merged_config (OmegaConf):
            The configuration with parameters overriden by sweep_config
    """
    # 1) convert the flat dict into a dot-list
    dotlist = [f"{k}={v}" for k, v in sweep_config_dict.items()]
    sweep_overrides = OmegaConf.from_dotlist(dotlist)

    # 2) merge base and overrides (sweep takes priority on conflict)
    merged_config = OmegaConf.merge(base_config, sweep_overrides)

    # 3) logging to check if any value in the base config has been overridden
    changed = {
        k: v
        for k, v in sweep_config_dict.items()
        if OmegaConf.select(base_config, k) != v
    }

    if changed:
        logger.info("Sweep overrides applied:")
        for k, v in changed.items():
            logger.info(f"\t{k:20s} -> {v}")

    return merged_config
