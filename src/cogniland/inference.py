"""Inference adapter for PPO-GRU checkpoints.

This module exposes a single class :class:`PPOAgent` that wraps a
``.pt`` checkpoint produced by ``scripts/crafter/train_ppo_gru.py`` and turns it
into a small, environment-friendly act-step interface:

    agent = PPOAgent.load(ckpt_path, device="cuda")  # or "cpu"
    obs, info = env.reset()
    hidden = agent.initial_hidden(batch=1)
    done = False
    while not done:
        action_dict, hidden = agent.act(obs, hidden, done=done, greedy=True)
        obs, r, term, trunc, info = env.step(action_dict)
        done = term or trunc

The :class:`PPOGRUPolicy` class itself is **not** re-defined here — we
load it from the training script with ``importlib`` exactly the same way
``scripts/crafter/play_ppo_gru.py`` does, so the two stay perfectly in sync and
any future change to the policy automatically flows through to inference.

Notes / assumptions
-------------------
* The build is a discrete move now (``build_raft`` / ``build_harness`` are
  two of the 6 actions of the categorical move head). The ``belief_head``
  (``tanh(linear)``) is a *deterministic* auxiliary map-recognition probe,
  not an action; ``act`` returns it alongside the move for inspection but
  never feeds it to the env. ``greedy=`` toggles the move head between
  argmax and Categorical sampling.
* ``act`` returns ``({"move": int, "belief": float}, hidden)``. The env
  only reads ``"move"`` (it also accepts a raw int move).
* This adapter is intentionally CPU/GPU agnostic — pass ``device=`` at
  load time and obs tensors are moved there inside :meth:`act`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ----------------------------------------------------------- policy import

def _load_ppo_gru_module():
    """Dynamically import ``scripts/crafter/train_ppo_gru.py`` as a module.

    We mirror the importlib trick from ``scripts/crafter/play_ppo_gru.py`` so the
    canonical :class:`PPOGRUPolicy` definition stays in one place. The
    trainer file is at ``<repo>/scripts/crafter/train_ppo_gru.py``; this file
    lives at ``<repo>/src/cogniland/inference.py``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    train_path = repo_root / "scripts" / "train_ppo_gru.py"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Could not locate train_ppo_gru.py at {train_path}. "
            "PPOAgent.load needs that file to recover the policy class."
        )
    spec = importlib.util.spec_from_file_location("train_ppo_gru", str(train_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------- agent wrapper

class PPOAgent:
    """Thin wrapper around a trained :class:`PPOGRUPolicy`."""

    def __init__(self, policy: torch.nn.Module, device: torch.device,
                 ckpt_args: dict[str, Any]):
        self.policy = policy
        self.device = device
        self.ckpt_args = ckpt_args
        # gru_hidden lives on the policy after __init__; keep a local copy
        # so callers can size the initial hidden state without touching the
        # underlying module.
        self.gru_hidden = int(getattr(policy, "gru_hidden"))

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, ckpt_path: str | Path, device: str = "cpu") -> "PPOAgent":
        """Load a PPO-GRU checkpoint and return a ready-to-act agent.

        ``device`` accepts the usual ``"cpu"``/``"cuda"``/``"cuda:0"``
        strings. The policy is moved to the requested device and put in
        ``eval()`` mode.
        """
        ckpt_path = Path(ckpt_path)
        dev = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        ckpt_args = dict(ckpt.get("args", {}))

        # We need an obs_space to instantiate the policy. Re-create one
        # using the same env settings the checkpoint was trained with —
        # this is exactly what scripts/crafter/play_ppo_gru.py does.
        from cogniland.nav import CognilandNavEnv  # local — avoid hard dep at import-time

        env = CognilandNavEnv(
            size=ckpt_args.get("env_size", 64),
            map_type=ckpt_args.get("map_type", "random"),
            view_size=ckpt_args.get("view_size", 21),
            tile_px=ckpt_args.get("tile_px", 8),
            obs_mode=ckpt_args.get("obs_mode", "symbolic"),
            max_steps=ckpt_args.get("max_steps", 1000),
            seed=int(ckpt_args.get("seed", 0)),
        )

        # action space is now Discrete(6) (build folded into the move axis);
        # fall back to the legacy Dict("move") layout for old checkpoints.
        aspace = env.action_space
        n_actions = aspace.n if hasattr(aspace, "n") else aspace["move"].n

        tp = _load_ppo_gru_module()
        policy = tp.PPOGRUPolicy(
            env.observation_space,
            num_move_actions=n_actions,
            gru_hidden=ckpt_args.get("gru_hidden", 128),
            embed_dim=ckpt_args.get("embed_dim", 256),
        ).to(dev)
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        env.close()

        return cls(policy=policy, device=dev, ckpt_args=ckpt_args)

    # -- runtime helpers -------------------------------------------------

    def initial_hidden(self, batch: int = 1) -> torch.Tensor:
        """Return a zeroed GRU hidden state of shape ``(1, batch, gru_hidden)``."""
        return torch.zeros(1, int(batch), self.gru_hidden, device=self.device)

    def _to_tensor_obs(self, obs: dict) -> dict[str, torch.Tensor]:
        """Convert a numpy obs dict (single env) into ``(1, ...)`` tensors."""
        out: dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            t = torch.as_tensor(np.asarray(v), device=self.device)
            out[k] = t.unsqueeze(0)  # add batch dim
        return out

    @torch.no_grad()
    def act(
        self,
        obs: dict,
        hidden: torch.Tensor,
        done: bool | torch.Tensor = False,
        greedy: bool = True,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Step the policy once and return ``(action_dict, new_hidden)``.

        ``obs`` is a single-env numpy dict (the same shape ``env.reset()``
        returns). With ``greedy=True`` the move is ``argmax`` of the logits;
        with ``greedy=False`` it is sampled from the Categorical. The build
        is just one of the 6 moves (``build_raft`` / ``build_harness``). The
        returned ``belief`` is the deterministic ``belief_head`` output (a
        map-recognition probe) — informational only, not sent to the env.
        """
        obs_t = self._to_tensor_obs(obs)
        if isinstance(done, bool):
            done_t = torch.tensor([1.0 if done else 0.0], device=self.device)
        else:
            done_t = done.to(self.device).float().reshape(-1)

        # `_gru_forward` expects obs with a leading time dim of 1 and a
        # done sequence of shape (T, B). We feed T=1.
        obs_seq = {k: v.unsqueeze(0) for k, v in obs_t.items()}  # (1, 1, ...)
        gru_out, h_new = self.policy._gru_forward(
            obs_seq, done_t.unsqueeze(0), hidden
        )
        x = gru_out.squeeze(0)  # (1, gru_hidden)
        logits, belief, _value = self.policy._heads(x)

        if greedy:
            move = int(logits.argmax(-1).item())
        else:
            from torch.distributions import Categorical
            move = int(Categorical(logits=logits).sample().item())

        belief_val = float(belief.squeeze().item())
        action_dict: dict[str, Any] = {"move": move, "belief": belief_val}
        return action_dict, h_new


__all__ = ["PPOAgent"]
