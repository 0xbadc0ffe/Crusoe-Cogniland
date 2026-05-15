"""Optional wrappers for the Cogniland nav env."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class TorchTensorWrapper(gym.ObservationWrapper):
    """Convert ``obs["image"]`` to a ``torch.Tensor`` of float32 in [0, 1].

    Torch is imported lazily so that callers using only the pygame demo or
    headless mapgen don't pay the import cost (which on macOS can trigger
    OpenMP/libomp conflicts with pygame).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise RuntimeError("TorchTensorWrapper requires PyTorch") from exc

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        import torch
        out = dict(obs)
        img = obs["image"]  # uint8 [C, H, W]
        tensor = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
        out["image"] = tensor
        return out
