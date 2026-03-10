"""PPO agent — architecture, rollout, GAE, training loop, and evaluation.

This module is fully self-contained: to train PPO, just call PPOAgent(cfg).train(cfg).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim

from cogniland.env.constants import NUM_ACTIONS, TERRAIN_COSTS
from cogniland.env.pathfinding import batch_astar
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
        scalar_dim: int = 7,
        minimap_channels: int = 2,
        hidden_dim: int = 128,
        action_dim: int = 5,
        cnn_channels: int = 32,
        cnn_out_spatial: int = 4,
        scalar_hidden: int = 64,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            _layer_init(nn.Conv2d(minimap_channels, cnn_channels // 2, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            _layer_init(nn.Conv2d(cnn_channels // 2, cnn_channels, 3, padding=1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(cnn_out_spatial),
            nn.Flatten(),
        )
        cnn_out = cnn_channels * cnn_out_spatial * cnn_out_spatial

        self.scalar_net = nn.Sequential(
            _layer_init(nn.Linear(scalar_dim, scalar_hidden)),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(cnn_out + scalar_hidden, hidden_dim)),
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

    @torch.no_grad()
    def get_deterministic_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.actor(self._features(obs))
        return logits.argmax(dim=-1)


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

        scalar_dim = cfg.models.get("scalar_dim", 7)
        action_dim = cfg.models.get("action_dim", NUM_ACTIONS)
        self.model = ActorCritic(
            scalar_dim=scalar_dim,
            minimap_channels=cfg.models.minimap_channels,
            hidden_dim=cfg.models.hidden_dim,
            action_dim=action_dim,
            cnn_channels=cfg.models.get("cnn_channels", 32),
            cnn_out_spatial=cfg.models.get("cnn_out_spatial", 4),
            scalar_hidden=cfg.models.get("scalar_hidden", 64),
        ).to(device)

    def get_action_and_value(self, obs, action=None):
        return self.model.get_action_and_value(obs, action)

    def get_deterministic_action(self, obs):
        return self.model.get_deterministic_action(obs)

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
            train_metrics["train/steps_per_second"] = sps
            train_metrics["train/global_step"] = global_step
            logger.log(train_metrics, step=update)

            # Periodic eval
            if update % cfg.models.training.eval_every_n_updates == 0:
                print(f"[Update {update}/{num_updates}] Running evaluation...")
                model.eval()
                eval_metrics = self._run_eval(cfg, logger=logger, global_step=update)
                model.train()
                logger.log(eval_metrics, step=update)
                det_sr = eval_metrics.get("eval-deterministic/success_rate_mean", 0.0)
                sto_sr = eval_metrics.get("eval-stochastic/success_rate_mean", 0.0)
                print(f"  deterministic success: {det_sr:.3f}, "
                      f"stochastic success: {sto_sr:.3f}")

            # Periodic checkpoint
            if update % cfg.models.training.checkpoint_every_n_updates == 0:
                import os
                run_id = logger._run.id if logger.enabled and logger._run else "local"
                ckpt_dir = f"artifacts/{run_id}"
                os.makedirs(ckpt_dir, exist_ok=True)
                
                ckpt_path = f"{ckpt_dir}/ckpt_{update}.pt"
                save_checkpoint(model, optimizer, global_step, path=ckpt_path)
                
                store_wandb = cfg.logging.wandb.get("store_last_ckpt", False)
                if store_wandb and update == num_updates:
                    logger.log_model_artifact(
                        name=f"{cfg.models.name}_agent",
                        path=ckpt_path,
                        aliases=["latest", f"update_{update}"]
                    )
                    print(f"  Checkpoint saved locally & as WandB artifact at step {global_step}")
                elif store_wandb:
                    print(f"  Checkpoint saved locally at {ckpt_path} (WandB upload deferred to last step)")
                else:
                    print(f"  Checkpoint saved locally at {ckpt_path}")

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

    def _run_eval_mode(self, cfg, deterministic: bool, mode_prefix: str):
        """Run K evaluation episodes with one policy mode.

        Returns:
            (metrics_dict, raw_data_dict) where raw_data_dict contains per-episode
            tensors and trajectory lists for downstream logging.
        """
        import matplotlib
        matplotlib.use("Agg")
        from cogniland.env.constants import TERRAIN_LEVELS, palette

        model = self.model
        device = self.device
        env_config = self.env_config

        eval_cfg = cfg.logging.get("eval", {})
        if deterministic:
            n_eps = eval_cfg.get("deterministic_episodes", cfg.models.training.eval_episodes)
        else:
            n_eps = eval_cfg.get("stochastic_episodes", cfg.models.training.eval_episodes)
        hp_danger_threshold = eval_cfg.get("hp_danger_threshold", 30.0)

        eval_env = BatchedIslandEnv(env_config, num_envs=n_eps)
        obs = eval_env.reset(seed=cfg.env.seed + 1000)

        # Record initial positions before the loop
        initial_spawns = eval_env.state.position.clone()
        initial_targets = eval_env.target_pos.clone()

        # Compute A* optimal path costs
        astar_costs = batch_astar(
            eval_env.env.world_map, TERRAIN_COSTS,
            initial_spawns, initial_targets,
        ).to(device)

        total_rewards = torch.zeros(n_eps, device=device)
        total_moves = torch.zeros(n_eps, device=device)
        reached = torch.zeros(n_eps, dtype=torch.bool, device=device)
        alive = torch.ones(n_eps, dtype=torch.bool, device=device)
        final_hp = torch.zeros(n_eps, device=device)
        terrain_visits = torch.zeros(n_eps, 9, device=device)

        # Distribution-aware tracking
        min_hp = torch.full((n_eps,), float(env_config.init_hp), dtype=torch.float32, device=device)
        danger_steps = torch.zeros(n_eps, device=device)
        # Welford online accumulators for resource mean
        resource_mean = torch.zeros(n_eps, device=device)
        resource_count = torch.zeros(n_eps, device=device)
        # Welford online accumulators for HP mean
        hp_mean = torch.zeros(n_eps, device=device)
        hp_m2 = torch.zeros(n_eps, device=device)
        hp_count = torch.zeros(n_eps, device=device)
        # Per-episode max resources
        max_resources = torch.zeros(n_eps, device=device)

        trajectories: list[list[tuple[int, int]]] = [[] for _ in range(n_eps)]

        for i in range(n_eps):
            p = eval_env.state.position[i].cpu().tolist()
            trajectories[i].append(tuple(p))

        for move in range(env_config.max_steps):
            still_running = alive & ~reached
            pre_move_terrain = eval_env.state.terrain_lev.clone()
            pre_move_hp = eval_env.state.hp.clone().float()

            with torch.no_grad():
                if deterministic:
                    action = model.get_deterministic_action(obs)
                else:
                    action, _, _, _ = model.get_action_and_value(obs)
            obs, reward, done, info = eval_env.step(action)

            total_rewards[still_running] += reward[still_running]
            total_moves[still_running] += 1

            # Track min HP
            current_hp = eval_env.state.hp
            min_hp[still_running] = torch.minimum(
                min_hp[still_running], current_hp[still_running]
            )

            # Track danger steps
            danger_mask = still_running & (current_hp < hp_danger_threshold)
            danger_steps[danger_mask] += 1

            # Welford online update for resource mean
            current_resources = eval_env.state.resources
            resource_count[still_running] += 1
            delta = current_resources[still_running] - resource_mean[still_running]
            resource_mean[still_running] += delta / resource_count[still_running]

            # Welford online update for HP mean
            hp_count[still_running] += 1
            hp_delta = current_hp[still_running] - hp_mean[still_running]
            hp_mean[still_running] += hp_delta / hp_count[still_running]
            hp_delta2 = current_hp[still_running] - hp_mean[still_running]
            hp_m2[still_running] += hp_delta * hp_delta2

            # Track max resources
            max_resources[still_running] = torch.maximum(
                max_resources[still_running], current_resources[still_running]
            )

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

            # Count terrain visit for every still-running episode (including stay actions).
            # terrain_lev is always in [0, 8] (clamped by compute_terrain_levels).
            running_idx = torch.where(still_running)[0]
            terrain_visits[running_idx, pre_move_terrain[running_idx].long()] += 1

            if (~alive | reached).all():
                break

        # Handle still-running episodes
        still_running = alive & ~reached
        final_hp[still_running] = eval_env.state.hp[still_running]
        total_moves[still_running] = env_config.max_steps

        # Per-episode metrics
        danger_fraction = danger_steps / total_moves.clamp(min=1)
        final_resources = eval_env.state.resources

        # Path efficiency: bounded [0, 1] where 1 = optimal A* path
        agent_cost = eval_env.state.cost
        astar_valid = astar_costs > 0
        path_efficiency = torch.where(
            astar_valid & (agent_cost > 0),
            (astar_costs / agent_cost.clamp(min=1e-6)).clamp(0.0, 1.0),
            torch.zeros_like(agent_cost),
        )

        # Aggregate distribution stats for each metric
        per_episode = {
            "return": total_rewards,
            "episode_length": total_moves,
            "min_hp": min_hp,
            "final_hp": final_hp,
            "danger_fraction": danger_fraction,
            "final_resources": final_resources,
            "max_resources": max_resources,
            "path_efficiency": path_efficiency,
            "mean_resource": resource_mean,
            "mean_hp": hp_mean,
        }

        metrics = {}
        prefix = f"eval-{mode_prefix}"
        metrics[f"{prefix}/success_rate_mean"] = reached.float().mean().item()
        
        # Add scalar means for all tracked metrics
        for name, values in per_episode.items():
            metrics[f"{prefix}/{name}_mean"] = values.float().mean().item()

        # Build histogram data
        hist_data = {}
        for name, values in per_episode.items():
            hist_data[name] = values.cpu()

        raw_data = {
            "reached": reached,
            "total_rewards": total_rewards,
            "total_moves": total_moves,
            "final_hp": final_hp,
            "trajectories": trajectories,
            "terrain_visits": terrain_visits,
            "eval_env": eval_env,
            "initial_targets": initial_targets,
            "n_eps": n_eps,
            "hist_data": hist_data,
        }

        return metrics, raw_data

    def _run_eval(self, cfg, logger=None, global_step=0):
        """Orchestrator: run deterministic + stochastic eval, merge metrics, log."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from cogniland.env.constants import TERRAIN_LEVELS, palette

        # Run both modes
        det_metrics, det_raw = self._run_eval_mode(cfg, deterministic=True, mode_prefix="deterministic")
        sto_metrics, sto_raw = self._run_eval_mode(cfg, deterministic=False, mode_prefix="stochastic")

        # Merge all metrics
        eval_metrics = {}
        eval_metrics.update(det_metrics)
        eval_metrics.update(sto_metrics)

        # Behavioral metrics per mode
        from cogniland.logging import TERRAIN_NAMES
        for mode_prefix, raw_data in [("deterministic", det_raw), ("stochastic", sto_raw)]:
            beh_scalars, beh_dists = compute_behavioral_metrics(
                raw_data["terrain_visits"], raw_data["total_moves"],
                raw_data["eval_env"].state,
            )
            raw_data["beh_scalars"] = beh_scalars
            raw_data["beh_dists"] = beh_dists

        if logger is not None:
            # Trajectory images first (deterministic mode only)
            max_images = cfg.logging.get("trajectory", {}).get("max_saved_per_eval", 4)
            figures, captions, env_indices = [], [], []
            det_env = det_raw["eval_env"]
            det_targets = det_raw["initial_targets"]
            det_trajs = det_raw["trajectories"]
            det_reached = det_raw["reached"]
            det_moves = det_raw["total_moves"]

            for i in range(det_raw["n_eps"]):
                if len(figures) >= max_images:
                    break
                if len(det_trajs[i]) < 2:
                    continue
                fig = render_trajectory(
                    det_env.env.world_map, det_trajs[i],
                    det_targets[i], det_reached[i].item(), i,
                    TERRAIN_LEVELS, palette,
                )
                outcome = "success" if det_reached[i].item() else "fail"
                n_moves = int(det_moves[i].item())
                figures.append(fig)
                captions.append(f"env{i} {outcome} {n_moves} moves")
                env_indices.append(i)

            if figures:
                logger.log_trajectory_images(figures, captions, env_indices, step=global_step)
                for fig in figures:
                    plt.close(fig)

            # Per-mode density heatmaps and terrain distribution
            for mode_prefix, raw_data in [("deterministic", det_raw), ("stochastic", sto_raw)]:
                hist = raw_data["hist_data"]
                beh = raw_data["beh_scalars"]

                # behavioral-{mode}/ panels: terrain, HP group, resource group
                terrain_pcts = {
                    name: beh[f"behavioral/terrain_{name}_pct"]
                    for name in TERRAIN_NAMES
                }
                logger.log_terrain_distribution(
                    terrain_pcts, step=global_step, mode_prefix=mode_prefix,
                )

                # HP group
                logger.log_density_heatmap(
                    f"behavioral-{mode_prefix}/mean_hp",
                    hist["mean_hp"].numpy(), global_step,
                    y_label="HP",
                )
                logger.log_density_heatmap(
                    f"behavioral-{mode_prefix}/final_hp",
                    hist["final_hp"].numpy(), global_step,
                    y_label="HP",
                )
                logger.log_density_heatmap(
                    f"behavioral-{mode_prefix}/danger_fraction",
                    hist["danger_fraction"].numpy(), global_step,
                    y_label="Fraction",
                )

                # Resource group
                logger.log_density_heatmap(
                    f"behavioral-{mode_prefix}/mean_resource",
                    hist["mean_resource"].numpy(), global_step,
                    y_label="Resources",
                )
                logger.log_density_heatmap(
                    f"behavioral-{mode_prefix}/final_resources",
                    hist["final_resources"].numpy(), global_step,
                    y_label="Resources",
                )
                logger.log_density_heatmap(
                    f"behavioral-{mode_prefix}/max_resources",
                    hist["max_resources"].numpy(), global_step,
                    y_label="Resources",
                )

                # eval-{mode}/ panels — explicit ordered list
                eval_metric_order = [
                    ("return", "Return"),
                    ("episode_length", "Moves"),
                    ("min_hp", "HP"),
                    ("mean_hp", "HP"),
                    ("final_hp", "HP"),
                    ("danger_fraction", "Fraction"),
                    ("mean_resource", "Resources"),
                    ("final_resources", "Resources"),
                    ("max_resources", "Resources"),
                    ("path_efficiency", "Efficiency"),
                ]
                for metric_name, y_label in eval_metric_order:
                    if metric_name in hist:
                        logger.log_density_heatmap(
                            f"eval-{mode_prefix}/{metric_name}",
                            hist[metric_name].numpy(), global_step,
                            y_label=y_label,
                        )

            # Eval tables per mode
            for mode_prefix, raw_data in [("deterministic", det_raw), ("stochastic", sto_raw)]:
                logger.log_eval_table(
                    raw_data["reached"], raw_data["total_rewards"],
                    raw_data["total_moves"], raw_data["final_hp"],
                    raw_data["trajectories"], step=global_step,
                    prefix=mode_prefix,
                )

        return eval_metrics


