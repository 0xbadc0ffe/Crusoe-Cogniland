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

    Picks the action most aligned with moving toward the target.
    No learnable parameters — deterministic policy.

    The compass observation is a unit vector ``(position − target) / dist``
    pointing *away* from the target.  We pick the action whose delta has the
    maximum dot product with the direction toward the target (i.e. ``-compass``),
    which is equivalent to ``argmin dot(compass, delta)``.
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
        compass = scalars[:, 0:2]  # [B, 2] — unit vector (position - target)

        # dot(compass, delta): negative when delta opposes compass (i.e. moves toward target)
        # argmin selects the action most aligned with moving toward target
        scores = (compass.unsqueeze(1) * self.deltas.unsqueeze(0)).sum(dim=-1)  # [B, 5]
        best_action = scores.argmin(dim=-1)  # [B]

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
            "eval-deterministic/success_rate_mean": reached.float().mean().item(),
        }

        beh_scalars, beh_dists = compute_behavioral_metrics(
            terrain_visits, total_moves, eval_env.state,
        )

        logger.log(eval_metrics, step=0)

        # Trajectory images first
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

        from cogniland.logging import TERRAIN_NAMES
        terrain_pcts = {
            name: beh_scalars[f"behavioral/terrain_{name}_pct"]
            for name in TERRAIN_NAMES
        }
        logger.log_terrain_distribution(terrain_pcts, step=0, mode_prefix="deterministic")

        # Behavioral heatmaps — HP group
        logger.log_density_heatmap(
            "behavioral-deterministic/mean_hp",
            beh_dists["hp_score"].numpy(), step=0,
            y_label="HP",
        )
        logger.log_density_heatmap(
            "behavioral-deterministic/final_hp",
            eval_env.state.hp.cpu().numpy(), step=0,
            y_label="HP",
        )
        logger.log_density_heatmap(
            "behavioral-deterministic/danger_fraction",
            torch.zeros(n_eps).numpy(), step=0,
            y_label="Fraction",
        )

        # Behavioral heatmaps — resource group
        logger.log_density_heatmap(
            "behavioral-deterministic/mean_resource",
            beh_dists["resource_held"].numpy(), step=0,
            y_label="Resources",
        )
        logger.log_density_heatmap(
            "behavioral-deterministic/final_resources",
            eval_env.state.resources.cpu().numpy(), step=0,
            y_label="Resources",
        )

        # Eval heatmaps — ordered: return, length, HP group, resource group
        logger.log_density_heatmap(
            "eval-deterministic/return",
            total_rewards.cpu().numpy(), step=0,
            y_label="Return",
        )
        logger.log_density_heatmap(
            "eval-deterministic/episode_length",
            total_moves.cpu().numpy(), step=0,
            y_label="Moves",
        )
        logger.log_density_heatmap(
            "eval-deterministic/final_hp",
            eval_env.state.hp.cpu().numpy(), step=0,
            y_label="HP",
        )
        logger.log_density_heatmap(
            "eval-deterministic/final_resources",
            eval_env.state.resources.cpu().numpy(), step=0,
            y_label="Resources",
        )

        logger.log_eval_table(
            reached, total_rewards, total_moves,
            eval_env.state.hp, trajectories, step=0,
        )

        print(f"\nCompass Baseline Results ({n_eps} episodes):")
        print(f"  Success rate: {eval_metrics['eval-deterministic/success_rate_mean']:.1%}")
        print(f"  Mean reward:  {total_rewards.mean().item():.2f}")
        print(f"  Mean moves:   {total_moves.mean().item():.0f}")

        logger.finish()
