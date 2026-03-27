#!/usr/bin/env python3
"""Training entry point — Hydra loads config, model handles the rest.

Usage:
    python train.py                                                    # PPO default
    python train.py models.training.total_env_moves=2000 logging.wandb.mode=disabled
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    from cogniland.models import build_model
    from cogniland.env.types import EnvConfig
    env_config = EnvConfig.from_hydra(cfg)
    device = env_config.resolved_device()
    model = build_model(cfg, env_config, device)
    model.train(cfg)


if __name__ == "__main__":
    main()
