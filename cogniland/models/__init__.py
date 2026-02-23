"""Model registry — build_model(cfg) returns a self-contained agent."""

from cogniland.env.types import EnvConfig


def build_model(cfg):
    """Build model from Hydra config. Returns a self-contained agent with .train(cfg)."""
    env_config = EnvConfig.from_hydra(cfg)
    device = env_config.resolved_device()
    name = cfg.models.name

    if name == "compass":
        from cogniland.models.compass import CompassModel
        return CompassModel(cfg, env_config, device)
    elif name == "ppo":
        from cogniland.models.ppo import PPOAgent
        return PPOAgent(cfg, env_config, device)
    else:
        raise ValueError(f"Unknown model: {name}")
