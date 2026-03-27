from cogniland.env.types import EnvConfig
from cogniland.models.ppo import PPOAgent
from cogniland.models.drc import DRCAgent

def build_model(cfg, env_config: EnvConfig, device: str):
    """Factory method to build the agent based on config."""
    model_name = cfg.models.name
    
    if model_name == "ppo":
        return PPOAgent(cfg, env_config, device)
    elif model_name == "drc":
        return DRCAgent(cfg, env_config, device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
