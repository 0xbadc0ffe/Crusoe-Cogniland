"""Reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_reproducibility(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds and (optionally) enable deterministic mode."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
