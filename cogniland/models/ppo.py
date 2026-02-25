"""PPO agent — architecture, rollout, GAE, training loop, and evaluation.

This module is fully self-contained: to train PPO, just call PPOAgent(cfg).train(cfg).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim

from cogniland.env.constants import NUM_ACTIONS
from cogniland.env.types import EnvConfig
from cogniland.env.wrappers import BatchedIslandEnv
from cogniland.logging import WandBLogger, compute_behavioral_metrics
from cogniland.utils import render_trajectory, save_checkpoint, set_reproducibility


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

def _layer_init(layer: nn.Module, std: float = 1.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """CNN (minimap) + MLP (scalars) → shared trunk → actor / critic heads."""

    def __init__(
        self,
        scalar_dim: int = 6,
        minimap_channels: int = 1,
        minimap_size: int = 51,
        hidden_dim: int = 128,
        action_dim: int = 5,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            _layer_init(nn.Conv2d(minimap_channels, 16, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            _layer_init(nn.Conv2d(16, 32, 3, padding=1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        cnn_out = 32 * 4 * 4

        self.scalar_net = nn.Sequential(
            _layer_init(nn.Linear(scalar_dim, 64)),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(cnn_out + 64, hidden_dim)),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )

        self.actor = _layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.critic = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        cnn_feat = self.cnn(obs["minimap"])
        scalar_feat = self.scalar_net(obs["scalars"])
        return self.trunk(torch.cat([cnn_feat, scalar_feat], dim=-1))

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(self._features(obs)).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        feat = self._features(obs)
        logits = self.actor(feat)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(feat).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer + GAE
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Stores a single rollout of experience for PPO training."""

    obs_scalars: list[torch.Tensor] = field(default_factory=list)
    obs_minimaps: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    dones: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs_scalars.append(obs["scalars"])
        self.obs_minimaps.append(obs["minimap"])
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def flatten(self) -> dict[str, torch.Tensor]:
        T = len(self.obs_scalars)
        B = self.obs_scalars[0].shape[0]
        return {
            "obs_scalars": torch.stack(self.obs_scalars).reshape(T * B, -1),
            "obs_minimaps": torch.stack(self.obs_minimaps).reshape(T * B, *self.obs_minimaps[0].shape[1:]),
            "actions": torch.stack(self.actions).reshape(T * B),
            "log_probs": torch.stack(self.log_probs).reshape(T * B),
            "rewards": torch.stack(self.rewards),
            "dones": torch.stack(self.dones),
            "values": torch.stack(self.values),
        }


@torch.no_grad()
def _collect_rollout(env, model, obs, rollout_steps):
    buffer = RolloutBuffer()
    all_final_rewards, all_final_lengths, all_final_reached = [], [], []

    for _ in range(rollout_steps):
        value = model.get_value(obs)
        action, log_prob, _, _ = model.get_action_and_value(obs)
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, log_prob, reward, done, value)

        if "final_rewards" in info:
            all_final_rewards.append(info["final_rewards"])
            all_final_lengths.append(info["final_lengths"])
            all_final_reached.append(info["final_reached"])
        obs = next_obs

    episode_stats = {}
    if all_final_rewards:
        episode_stats["episode_rewards"] = torch.cat(all_final_rewards)
        episode_stats["episode_lengths"] = torch.cat(all_final_lengths)
        episode_stats["episode_reached"] = torch.cat(all_final_reached)
    return buffer, obs, episode_stats


def _compute_gae(buffer, next_value, gamma=0.99, gae_lambda=0.95):
    rewards = torch.stack(buffer.rewards)
    dones = torch.stack(buffer.dones)
    values = torch.stack(buffer.values)
    T, B = rewards.shape

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages.reshape(T * B), returns.reshape(T * B)


# ---------------------------------------------------------------------------
# PPO agent (self-contained: architecture + training)
# ---------------------------------------------------------------------------

class PPOAgent:
    """Full PPO agent — build with build_model(), then call .train(cfg)."""

    def __init__(self, cfg, env_config: EnvConfig, device: str):
        self.env_config = env_config
        self.device = device

        minimap_size = 2 * cfg.env.minimap_ray + 1
        scalar_dim = cfg.models.get("scalar_dim", 6)
        action_dim = cfg.models.get("action_dim", NUM_ACTIONS)
        self.model = ActorCritic(
            scalar_dim=scalar_dim,
            minimap_channels=cfg.models.minimap_channels,
            minimap_size=minimap_size,
            hidden_dim=cfg.models.hidden_dim,
            action_dim=action_dim,
        ).to(device)

    def get_action_and_value(self, obs, action=None):
        return self.model.get_action_and_value(obs, action)

    def get_value(self, obs):
        return self.model.get_value(obs)

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval()

    def train_mode(self):
        self.model.train()

    def train(self, cfg):
        """Full PPO training loop: rollout → GAE → update → eval → checkpoint."""
        set_reproducibility(cfg.env.seed)
        device = self.device
        model = self.model

        logger = WandBLogger(cfg)
        print(f"Device: {device}")
        print(f"Model: ppo")

        env = BatchedIslandEnv(self.env_config, num_envs=cfg.models.training.parallel_envs)
        optimizer = optim.Adam(model.parameters(), lr=cfg.models.training.learning_rate, eps=1e-5)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {param_count:,}")

        num_envs = cfg.models.training.parallel_envs
        rollout_steps = cfg.models.training.moves_per_rollout
        total_timesteps = cfg.models.training.total_env_moves
        num_updates = total_timesteps // (num_envs * rollout_steps)
        print(f"Total updates: {num_updates}, Moves per update: {num_envs * rollout_steps}")

        obs = env.reset(seed=cfg.env.seed)
        global_step = 0
        start_time = time.time()

        for update in range(1, num_updates + 1):
            # LR annealing
            if cfg.models.training.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                lr = frac * cfg.models.training.learning_rate
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            # Collect rollout
            buffer, obs, episode_stats = _collect_rollout(env, model, obs, rollout_steps)
            global_step += num_envs * rollout_steps

            # Log episode stats
            if episode_stats:
                ep_rewards = episode_stats["episode_rewards"]
                ep_lengths = episode_stats["episode_lengths"]
                logger.log({
                    "train/mean_reward": ep_rewards.mean().item(),
                    "train/mean_episode_length": ep_lengths.mean().item(),
                    "train/global_step": global_step,
                }, step=update)

            # Compute GAE
            with torch.no_grad():
                next_value = model.get_value(obs)
            advantages, returns = _compute_gae(
                buffer, next_value,
                gamma=cfg.models.training.discount_factor,
                gae_lambda=cfg.models.training.gae_lambda,
            )

            # PPO update
            flat_data = buffer.flatten()
            train_metrics = self._ppo_update(optimizer, flat_data, advantages, returns, cfg)

            current_lr = optimizer.param_groups[0]["lr"]
            train_metrics["train/learning_rate"] = current_lr
            sps = int(global_step / (time.time() - start_time))
            train_metrics["train/sps"] = sps
            train_metrics["train/global_step"] = global_step
            logger.log(train_metrics, step=update)

            # Periodic eval
            if update % cfg.models.training.eval_every_n_updates == 0:
                print(f"[Update {update}/{num_updates}] Running evaluation...")
                model.eval()
                eval_metrics = self._run_eval(cfg, logger=logger, global_step=update)
                model.train()
                logger.log(eval_metrics, step=update)
                print(f"  eval/success_rate: {eval_metrics['eval/success_rate']:.3f}, "
                      f"eval/mean_reward: {eval_metrics['eval/mean_reward']:.2f}")

            # Periodic checkpoint
            if update % cfg.models.training.checkpoint_every_n_updates == 0:
                save_checkpoint(model, optimizer, global_step, path=f"checkpoints/ckpt_{update}.pt")
                print(f"  Checkpoint saved at step {global_step}")

        logger.finish()
        print(f"Training complete. Total timesteps: {global_step}")

    def _ppo_update(self, optimizer, flat_data, advantages, returns, cfg):
        model = self.model
        N = flat_data["actions"].shape[0]
        minibatch_size = cfg.models.training.minibatch_size
        clip_coef = cfg.models.training.policy_clip_range
        vf_coef = cfg.models.training.value_loss_weight
        ent_coef = cfg.models.training.entropy_bonus_weight
        max_grad_norm = cfg.models.training.max_grad_norm

        adv = advantages
        if adv.std() > 0:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_pg_loss = total_vf_loss = total_entropy = 0.0
        total_clipfrac = total_approx_kl = 0.0
        n_updates = 0

        for _epoch in range(cfg.models.training.epochs_per_update):
            indices = torch.randperm(N, device=flat_data["actions"].device)
            for start in range(0, N, minibatch_size):
                end = start + minibatch_size
                if end > N:
                    break
                mb_idx = indices[start:end]

                mb_obs = {
                    "scalars": flat_data["obs_scalars"][mb_idx],
                    "minimap": flat_data["obs_minimaps"][mb_idx],
                }
                mb_actions = flat_data["actions"][mb_idx]
                mb_old_logprobs = flat_data["log_probs"][mb_idx]
                mb_advantages = adv[mb_idx]
                mb_returns = returns[mb_idx]

                _, new_logprob, entropy, new_value = model.get_action_and_value(mb_obs, mb_actions)

                log_ratio = new_logprob - mb_old_logprobs
                ratio = log_ratio.exp()
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy_loss.item()
                total_clipfrac += clipfrac
                total_approx_kl += approx_kl
                n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            "train/policy_loss": total_pg_loss / n_updates,
            "train/value_loss": total_vf_loss / n_updates,
            "train/entropy": total_entropy / n_updates,
            "train/clipfrac": total_clipfrac / n_updates,
            "train/approx_kl": total_approx_kl / n_updates,
        }

    def _run_eval(self, cfg, logger=None, global_step=0):
        """Run evaluation episodes and return metrics + log trajectory images."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from cogniland.env.constants import TERRAIN_LEVELS, palette

        model = self.model
        device = self.device
        env_config = self.env_config

        n_eps = cfg.models.training.eval_episodes
        eval_env = BatchedIslandEnv(env_config, num_envs=n_eps)
        obs = eval_env.reset(seed=cfg.env.seed + 1000)

        total_rewards = torch.zeros(n_eps, device=device)
        total_moves = torch.zeros(n_eps, device=device)
        reached = torch.zeros(n_eps, dtype=torch.bool, device=device)
        alive = torch.ones(n_eps, dtype=torch.bool, device=device)
        final_hp = torch.zeros(n_eps, device=device)
        terrain_visits = torch.zeros(n_eps, 9, device=device)

        trajectories: list[list[tuple[int, int]]] = [[] for _ in range(n_eps)]
        initial_targets = eval_env.target_pos.clone()

        for i in range(n_eps):
            p = eval_env.state.position[i].cpu().tolist()
            trajectories[i].append(tuple(p))

        for move in range(env_config.max_steps):
            still_running = alive & ~reached
            pre_move_terrain = eval_env.state.terrain_lev.clone()
            pre_move_hp = eval_env.state.hp.clone()

            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs)
            obs, reward, done, info = eval_env.step(action)

            total_rewards[still_running] += reward[still_running]
            total_moves[still_running] += 1

            newly_reached = info.get("reached", torch.zeros_like(done, dtype=torch.bool))
            reached = reached | (newly_reached & still_running)

            newly_dead = ~info.get("alive", torch.ones_like(done, dtype=torch.bool))
            alive = alive & ~newly_dead

            truncated = done & ~newly_reached & ~newly_dead
            just_finished = (newly_reached | newly_dead | truncated) & still_running
            final_hp[just_finished] = pre_move_hp[just_finished]

            for i in torch.where(still_running)[0].tolist():
                if newly_reached[i]:
                    tgt = initial_targets[i].cpu().tolist()
                    trajectories[i].append(tuple(tgt))
                elif not (newly_dead[i] or truncated[i]):
                    p = eval_env.state.position[i].cpu().tolist()
                    trajectories[i].append(tuple(p))

            alive = alive & ~truncated

            for i in torch.where(still_running)[0].tolist():
                t = int(pre_move_terrain[i].item())
                if 0 <= t <= 8:
                    terrain_visits[i, t] += 1

            if (~alive | reached).all():
                break

        still_running = alive & ~reached
        final_hp[still_running] = eval_env.state.hp[still_running]
        total_moves[still_running] = env_config.max_steps

        initial_dist = torch.norm(
            (eval_env.state.position - eval_env.target_pos).float(), dim=1
        )
        path_efficiency = torch.where(
            total_moves > 0,
            initial_dist / total_moves.clamp(min=1),
            torch.zeros_like(total_moves),
        )

        eval_metrics = {
            "eval/success_rate": reached.float().mean().item(),
            "eval/mean_reward": total_rewards.mean().item(),
            "eval/mean_moves": total_moves.mean().item(),
            "eval/mean_final_hp": final_hp.mean().item(),
            "eval/path_efficiency": path_efficiency.mean().item(),
        }

        behavioral = compute_behavioral_metrics(terrain_visits, total_moves, eval_env.state)
        eval_metrics.update(behavioral)

        # Log trajectory images + behavioral profile
        if logger is not None:
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
                eval_episode = global_step // cfg.models.training.eval_every_n_updates
                logger.log_trajectory_images(figures, captions, env_indices, step=eval_episode)
                for fig in figures:
                    plt.close(fig)

            logger.log_behavioral_profile(behavioral, step=global_step)

            # Structured per-episode eval table for final report
            logger.log_eval_table(
                reached, total_rewards, total_moves, final_hp,
                trajectories, step=global_step,
            )

        return eval_metrics


