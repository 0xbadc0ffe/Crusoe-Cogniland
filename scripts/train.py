#!/usr/bin/env python3
"""Hydra entry point for training.

Usage:
    python scripts/train.py                                     # defaults
    python scripts/train.py model=random env=hard               # random agent, hard mode
    python scripts/train.py training.learning_rate=1e-4         # override LR
    python scripts/train.py training.total_timesteps=5000 logging.wandb.mode=disabled  # smoke test
    python scripts/train.py device=cuda                         # force GPU
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cogniland.training.trainer import train
    train(cfg)


if __name__ == "__main__":
    main()
