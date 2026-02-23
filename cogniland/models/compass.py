"""Compass agent — greedy baseline that follows the direction to the target.

This module is self-contained: CompassModel(cfg).train(cfg) runs evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cogniland.env.constants import ACTIONS
from cogniland.env.types import EnvConfig
from cogniland.env.wrappers import BatchedIslandEnv
from cogniland.logging import WandBLogger, compute_behavioral_metrics
from cogniland.utils import set_reproducibility


class CompassAgent(nn.Module):
    """Greedy compass-following baseline.

    Picks the action that minimises Manhattan distance to the target.
    No learnable parameters — deterministic policy.
    """

    def __init__(self, action_dim: int = 5):
        super().__init__()
        self.action_dim = action_dim
        # Movement deltas: up, down, left, right, stay  (row, col)
        self.register_buffer(
            "deltas",
            torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=torch.float32),
        )

    def get_action_and_value(self, obs, action=None):
        scalars = obs["scalars"]
        compass = scalars[:, 4:6]  # [B, 2] — direction to target (row, col)

        # Candidate positions after each action
        candidates = compass.unsqueeze(1) - self.deltas.unsqueeze(0)  # [B, 5, 2]
        distances = candidates.abs().sum(dim=-1)  # [B, 5] Manhattan distance

        best_action = distances.argmin(dim=-1)  # [B]

        # Dummy outputs matching PPO interface
        log_prob = torch.zeros_like(best_action, dtype=torch.float32)
        entropy = torch.zeros_like(log_prob)
        value = torch.zeros_like(log_prob)

        if action is not None:
            best_action = action

        return best_action, log_prob, entropy, value

    def get_value(self, obs):
        B = obs["scalars"].shape[0]
        return torch.zeros(B, device=obs["scalars"].device)


class CompassModel:
    """Compass model wrapper — .train() runs a single evaluation pass."""

    def __init__(self, cfg, env_config: EnvConfig, device: str):
        self.env_config = env_config
        self.device = device
        self.model = CompassAgent(action_dim=cfg.models.action_dim).to(device)

    def get_action_and_value(self, obs, action=None):
        return self.model.get_action_and_value(obs, action)

    def train(self, cfg):
        """Run evaluation (compass is deterministic — no training needed)."""
        set_reproducibility(cfg.env.seed)
        logger = WandBLogger(cfg)
        print(f"Device: {self.device}")
        print(f"Model: compass (baseline)")

        n_eps = cfg.training.eval_episodes
        eval_env = BatchedIslandEnv(self.env_config, num_envs=n_eps)
        obs = eval_env.reset(seed=cfg.env.seed + 1000)

        total_rewards = torch.zeros(n_eps, device=self.device)
        total_moves = torch.zeros(n_eps, device=self.device)
        reached = torch.zeros(n_eps, dtype=torch.bool, device=self.device)
        alive = torch.ones(n_eps, dtype=torch.bool, device=self.device)
        terrain_visits = torch.zeros(n_eps, 9, device=self.device)

        for move in range(self.env_config.max_steps):
            still_running = alive & ~reached

            with torch.no_grad():
                action, _, _, _ = self.model.get_action_and_value(obs)
            obs, reward, done, info = eval_env.step(action)

            total_rewards[still_running] += reward[still_running]
            total_moves[still_running] += 1

            newly_reached = info.get("reached", torch.zeros_like(done, dtype=torch.bool))
            reached = reached | (newly_reached & still_running)

            newly_dead = ~info.get("alive", torch.ones_like(done, dtype=torch.bool))
            alive = alive & ~newly_dead

            for i in torch.where(still_running)[0].tolist():
                t = int(eval_env.state.terrain_lev[i].item())
                if 0 <= t <= 8:
                    terrain_visits[i, t] += 1

            if (~alive | reached).all():
                break

        eval_metrics = {
            "eval/success_rate": reached.float().mean().item(),
            "eval/mean_reward": total_rewards.mean().item(),
            "eval/mean_moves": total_moves.mean().item(),
            "eval/mean_final_hp": eval_env.state.hp.mean().item(),
        }

        behavioral = compute_behavioral_metrics(terrain_visits, total_moves, eval_env.state)
        eval_metrics.update(behavioral)

        logger.log(eval_metrics, step=0)
        print(f"\nCompass Baseline Results ({n_eps} episodes):")
        print(f"  Success rate: {eval_metrics['eval/success_rate']:.1%}")
        print(f"  Mean reward:  {eval_metrics['eval/mean_reward']:.2f}")
        print(f"  Mean moves:   {eval_metrics['eval/mean_moves']:.0f}")

        logger.finish()
