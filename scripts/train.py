#!/usr/bin/env python3
"""Training entry point — Hydra loads config, model handles the rest.

Usage:
    python scripts/train.py                                         # PPO, easy mode
    python scripts/train.py models=compass                          # compass baseline
    python scripts/train.py env=hard training.learning_rate=1e-4    # hard mode
    python scripts/train.py training.total_timesteps=5000 logging.wandb.mode=disabled
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cogniland.models import build_model
    model = build_model(cfg)
    model.train(cfg)


if __name__ == "__main__":
    main()
