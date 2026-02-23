"""Compass agent — greedy baseline that follows the direction to the target.

This module is self-contained: CompassModel(cfg).train(cfg) runs evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cogniland.env.constants import ACTION_DELTAS, ACTIONS, NUM_ACTIONS
from cogniland.env.types import EnvConfig
from cogniland.env.wrappers import BatchedIslandEnv
from cogniland.logging import WandBLogger, compute_behavioral_metrics
from cogniland.utils import render_trajectory, set_reproducibility


class CompassAgent(nn.Module):
    """Greedy compass-following baseline.

    Picks the action that minimises Manhattan distance to the target.
    No learnable parameters — deterministic policy.

    The compass observation is ``position - target`` (pointing *away* from the
    target).  We therefore pick the action whose movement delta best cancels
    out that vector, i.e. ``argmin |compass - delta|``.
    """

    def __init__(self, action_dim: int = 5):
        super().__init__()
        self.action_dim = action_dim
        # Use the canonical ACTION_DELTAS from constants (up/down/right/left/stay)
        self.register_buffer(
            "deltas",
            ACTION_DELTAS.float(),
        )

    def get_action_and_value(self, obs, action=None):
        scalars = obs["scalars"]
        compass = scalars[:, 0:2]  # [B, 2] — (position - target) in (row, col)

        # new_pos = pos + delta  =>  new_compass = (pos + delta) - target = compass + delta
        # We want to pick the action that minimises |new_compass|.
        candidates = compass.unsqueeze(1) + self.deltas.unsqueeze(0)  # [B, 5, 2]
        distances = candidates.abs().sum(dim=-1)  # [B, 5]

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
        self.model = CompassAgent(
            action_dim=cfg.models.get("action_dim", NUM_ACTIONS)
        ).to(device)

    def get_action_and_value(self, obs, action=None):
        return self.model.get_action_and_value(obs, action)

    def train(self, cfg):
        """Run evaluation (compass is deterministic — no training needed)."""
        set_reproducibility(cfg.env.seed)
        logger = WandBLogger(cfg)
        print(f"Device: {self.device}")
        print(f"Model: compass (baseline)")

        n_eps = cfg.models.eval_episodes
        eval_env = BatchedIslandEnv(self.env_config, num_envs=n_eps)
        obs = eval_env.reset(seed=cfg.env.seed + 1000)

        total_rewards = torch.zeros(n_eps, device=self.device)
        total_moves = torch.zeros(n_eps, device=self.device)
        reached = torch.zeros(n_eps, dtype=torch.bool, device=self.device)
        alive = torch.ones(n_eps, dtype=torch.bool, device=self.device)
        terrain_visits = torch.zeros(n_eps, 9, device=self.device)

        # Track positions for trajectory rendering
        trajectories: list[list[tuple[int, int]]] = [[] for _ in range(n_eps)]
        initial_targets = eval_env.target_pos.clone()
        for i in range(n_eps):
            p = eval_env.state.position[i].cpu().tolist()
            trajectories[i].append(tuple(p))

        for move in range(self.env_config.max_steps):
            still_running = alive & ~reached
            pre_move_hp = eval_env.state.hp.clone()

            with torch.no_grad():
                action, _, _, _ = self.model.get_action_and_value(obs)
            obs, reward, done, info = eval_env.step(action)

            total_rewards[still_running] += reward[still_running]
            total_moves[still_running] += 1

            newly_reached = info.get("reached", torch.zeros_like(done, dtype=torch.bool))
            reached = reached | (newly_reached & still_running)

            newly_dead = ~info.get("alive", torch.ones_like(done, dtype=torch.bool))
            alive = alive & ~newly_dead

            truncated = done & ~newly_reached & ~newly_dead

            # Record positions
            for i in torch.where(still_running)[0].tolist():
                if newly_reached[i]:
                    tgt = initial_targets[i].cpu().tolist()
                    trajectories[i].append(tuple(tgt))
                elif not (newly_dead[i] or truncated[i]):
                    p = eval_env.state.position[i].cpu().tolist()
                    trajectories[i].append(tuple(p))

            alive = alive & ~truncated

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
        logger.log_behavioral_profile(behavioral, step=0)
        logger.log_eval_table(
            reached, total_rewards, total_moves,
            eval_env.state.hp, trajectories, step=0,
        )

        # Log trajectory images
        if logger.enabled:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            from cogniland.env.constants import TERRAIN_LEVELS, palette

            max_images = cfg.logging.get("trajectory", {}).get("max_saved_per_eval", 4)
            figures, captions, env_indices = [], [], []
            for i in range(n_eps):
                if len(figures) >= max_images:
                    break
                if len(trajectories[i]) < 2:
                    continue
                fig = render_trajectory(
                    eval_env.env.world_map, trajectories[i],
                    initial_targets[i], reached[i].item(), i,
                    TERRAIN_LEVELS, palette,
                )
                outcome = "success" if reached[i].item() else "fail"
                n_moves = int(total_moves[i].item())
                figures.append(fig)
                captions.append(f"env{i} {outcome} {n_moves} moves")
                env_indices.append(i)

            if figures:
                logger.log_trajectory_images(figures, captions, env_indices, step=0)
                for fig in figures:
                    plt.close(fig)

        print(f"\nCompass Baseline Results ({n_eps} episodes):")
        print(f"  Success rate: {eval_metrics['eval/success_rate']:.1%}")
        print(f"  Mean reward:  {eval_metrics['eval/mean_reward']:.2f}")
        print(f"  Mean moves:   {eval_metrics['eval/mean_moves']:.0f}")

        logger.finish()
