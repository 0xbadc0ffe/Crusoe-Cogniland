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
from cogniland.env.dataset import MapDataset
from cogniland.env.types import CurriculumStage, EnvConfig
from cogniland.env.wrappers import BatchedIslandEnv
from cogniland.eval import CognilandSummarizer, EvalRunner
from cogniland.logging import WandBLogger, log_rollout_stats
from cogniland.utils import load_checkpoint, render_trajectory, save_checkpoint, set_reproducibility



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
        scalar_dim: int = 5,
        minimap_channels: int = 3,
        hidden_dim: int = 256,
        action_dim: int = 5,
        cnn_channels: int = 32,
        cnn_out_spatial: int = 4,
        scalar_hidden: int = 64,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            _layer_init(nn.Conv2d(minimap_channels, cnn_channels // 2, 3, padding=1)),  # 3→16 ch, 45x45
            nn.ReLU(),
            nn.MaxPool2d(2), # 45x45 → 22x22
            _layer_init(nn.Conv2d(cnn_channels // 2, cnn_channels, 3, padding=1)), # 16→32 ch, 22x22
            nn.ReLU(),
            _layer_init(nn.Conv2d(cnn_channels, cnn_channels, 3, padding=1)), # 32→32 ch, 22x22
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(cnn_out_spatial),  # 22x22 → 4x4
            nn.Flatten(), # 32*4*4 = 512
        )
        cnn_out = cnn_channels * cnn_out_spatial * cnn_out_spatial  # 32*4*4 = 512

        self.scalar_net = nn.Sequential(
            _layer_init(nn.Linear(scalar_dim, scalar_hidden)),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(cnn_out + scalar_hidden, hidden_dim)),  # 512+64 = 576 -> 256
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),  # 256→256
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
        set_reproducibility(cfg.env.map_generation.seed)
        device = self.device
        model = self.model

        logger = WandBLogger(cfg)
        print(f"Device: {device}")
        print(f"Model: ppo")

        training_cfg = cfg.models.training
        dataset_cfg = training_cfg.get("dataset", {})
        curriculum_switch_steps = dataset_cfg.get("curriculum_switch_steps", 0)
        curriculum_switch_steps_2 = dataset_cfg.get("curriculum_switch_steps_2", 0)
        curriculum_extra_easy_radius = dataset_cfg.get("curriculum_extra_easy_radius", 25)
        curriculum_easy_radius = dataset_cfg.get("curriculum_easy_radius", 50)

        # Load map dataset if configured (train/val/test splits)
        dataset: MapDataset | None = None
        if dataset_cfg:
            dataset = MapDataset.load_from_config(dataset_cfg)
        if dataset is not None:
            train_path = dataset_cfg.get("train_path", "")
            val_path = dataset_cfg.get("val_path", "")
            test_path = dataset_cfg.get("test_path", "")
            print("Loading MapDataset from split files:")
            print(f"  train={train_path}")
            print(f"  val={val_path}")
            print(f"  test={test_path}")
            print(f"  train={dataset.n_train} val={dataset.n_val} test={dataset.n_test} maps")

        # Train environment — uses train split maps if dataset loaded, else procedural
        env = BatchedIslandEnv(
            self.env_config,
            num_envs=cfg.models.training.parallel_envs,
            world_maps=dataset.train_maps if dataset else None,
            curriculum_extra_easy_radius=curriculum_extra_easy_radius,
            curriculum_easy_radius=curriculum_easy_radius,
        )
        optimizer = optim.Adam(model.parameters(), lr=cfg.models.training.learning_rate, eps=1e-5)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {param_count:,}")

        num_envs = cfg.models.training.parallel_envs
        rollout_steps = cfg.models.training.moves_per_rollout
        total_timesteps = cfg.models.training.total_env_moves
        num_updates = total_timesteps // (num_envs * rollout_steps)
        print(f"Total updates: {num_updates}, Moves per update: {num_envs * rollout_steps}")

        # Curriculum setup
        curriculum_active = curriculum_switch_steps > 0
        if curriculum_active:
            env.set_curriculum_stage(CurriculumStage.EXTRA_EASY)
            print(f"Curriculum: EXTRA_EASY stage until global_step={curriculum_switch_steps}, then EASY until {curriculum_switch_steps_2}, then NORMAL")

        obs = env.reset(seed=cfg.env.map_generation.seed)
        global_step = 0
        start_update = 1

        resume_path = cfg.get("resume", None)
        if resume_path:
            ckpt = load_checkpoint(resume_path, model, optimizer, device=device)
            global_step = ckpt["step"]
            start_update = global_step // (num_envs * rollout_steps) + 1
            print(f"Resumed from {resume_path} — global_step={global_step}, starting at update {start_update}")

        # Pre-generate and cache the evaluation environment (val split maps)
        eval_cfg = cfg.logging.get("eval", {})
        eval_seed = cfg.env.map_generation.seed + eval_cfg.get("eval_seed_offset", 1000)
        n_eps_det = eval_cfg.get("deterministic_episodes", cfg.models.training.eval_episodes)
        n_eps_sto = eval_cfg.get("stochastic_episodes", cfg.models.training.eval_episodes)
        max_eval_eps = max(n_eps_det, n_eps_sto)
        print(f"Caching Eval Env (seed offset {eval_cfg.get('eval_seed_offset', 1000)})...")
        self.eval_env = BatchedIslandEnv(
            self.env_config,
            num_envs=max_eval_eps,
            world_maps=dataset.val_maps if dataset else None,
            curriculum_extra_easy_radius=curriculum_extra_easy_radius,
            curriculum_easy_radius=curriculum_easy_radius,
        )
        self.eval_env.reset(seed=eval_seed)

        # Store test maps for lazy test env construction at end of training
        self._test_maps = dataset.test_maps if dataset else None
        self._max_eval_eps = max_eval_eps
        self._eval_seed = eval_seed

        # Cache EvalRunner to avoid rebuilding on every periodic eval
        self._eval_runner = EvalRunner(self.eval_env, self.env_config, device)

        import os
        run_id = logger._run.id if logger.enabled and logger._run else "local"
        ckpt_dir = f"artifacts/{run_id}"
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt_path = f"{ckpt_dir}/ckpt_best.pt"
        best_val_sr = -1.0

        start_time = time.time()

        for update in range(start_update, num_updates + 1):
            # LR annealing (uses absolute update position so resumed runs decay correctly)
            if cfg.models.training.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                lr = frac * cfg.models.training.learning_rate
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            # Collect rollout
            buffer, obs, episode_stats = _collect_rollout(env, model, obs, rollout_steps)
            global_step += num_envs * rollout_steps

            # Curriculum: step through EXTRA_EASY → EASY → NORMAL
            if curriculum_active:
                stage = env._curriculum_stage
                if stage == CurriculumStage.EXTRA_EASY and global_step >= curriculum_switch_steps:
                    env.set_curriculum_stage(CurriculumStage.EASY)
                    print(f"[Update {update}] Curriculum switched to EASY (global_step={global_step})")
                elif stage == CurriculumStage.EASY and curriculum_switch_steps_2 > 0 and global_step >= curriculum_switch_steps_2:
                    env.set_curriculum_stage(CurriculumStage.NORMAL)
                    curriculum_active = False
                    print(f"[Update {update}] Curriculum switched to NORMAL (global_step={global_step})")

            # Log episode stats from rollout
            log_rollout_stats(logger, episode_stats, step=update)

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
            train_metrics["train/model/ppo/learning_rate"] = current_lr
            sps = int(global_step / (time.time() - start_time))
            train_metrics["train/sps"] = sps
            logger.log(train_metrics, step=update)

            # Periodic eval
            if update % cfg.models.training.eval_every_n_updates == 0:
                print(f"[Update {update}/{num_updates}] Running evaluation...")
                model.eval()
                eval_metrics = self._run_eval(cfg, logger=logger, global_step=update, split="val")
                model.train()
                logger.log(eval_metrics, step=update)
                det_sr = eval_metrics.get("val_det/env/success_rate", 0.0)
                sto_sr = eval_metrics.get("val_stoch/env/success_rate", 0.0)
                print(f"  deterministic success: {det_sr:.3f}, "
                      f"stochastic success: {sto_sr:.3f}")

                if det_sr > best_val_sr:
                    best_val_sr = det_sr
                    save_checkpoint(model, optimizer, global_step, path=best_ckpt_path)
                    print(f"  New best val success rate {det_sr:.3f} — saved {best_ckpt_path}")

            # Periodic checkpoint (0 = disabled, save best-only)
            ckpt_interval = cfg.models.training.checkpoint_every_n_updates
            if ckpt_interval > 0 and update % ckpt_interval == 0:
                import os
                run_id = logger._run.id if logger.enabled and logger._run else "local"
                ckpt_dir = f"artifacts/{run_id}"
                os.makedirs(ckpt_dir, exist_ok=True)

                ckpt_path = f"{ckpt_dir}/ckpt_{update}.pt"
                save_checkpoint(model, optimizer, global_step, path=ckpt_path)
                print(f"  Checkpoint saved locally at {ckpt_path}")

        # ── Upload best checkpoint to WandB ──
        if best_val_sr < 0:  # no eval ran — save current weights as fallback
            save_checkpoint(model, optimizer, global_step, path=best_ckpt_path)
        print(f"  Best checkpoint: val_det success {best_val_sr:.3f} → {best_ckpt_path}")
        store_wandb = cfg.logging.wandb.get("store_last_ckpt", False)
        if store_wandb:
            logger.log_model_artifact(
                name=f"{cfg.models.name}_agent",
                path=best_ckpt_path,
                aliases=["best", f"sr{best_val_sr:.3f}"]
            )
            print(f"  Best checkpoint uploaded to WandB as artifact")

        # ── Final test evaluation ──
        print("Running final test evaluation...")
        model.eval()
        test_metrics = self._run_eval(cfg, logger=logger, global_step=global_step, split="test")
        model.train()
        test_metrics["test_det/env/best_ckpt_path"] = best_ckpt_path
        logger.log(test_metrics, step=num_updates + 1)
        logger.log_final_test_summary(test_metrics)
        test_sr = test_metrics.get("test_det/env/success_rate", 0.0)
        print(f"  test deterministic success: {test_sr:.3f}")

        # ── Behavioral test evaluation ──
        print("Running behavioral map evaluation...")
        model.eval()
        beh_metrics = self._run_behavioral_eval(logger=logger, global_step=num_updates + 2)
        model.train()
        if beh_metrics:
            logger.log(beh_metrics, step=num_updates + 2)
            beh_sr = beh_metrics.get("test/behavioral/success_rate", 0.0)
            print(f"  behavioral success rate: {beh_sr:.3f}")

        logger.finish()
        print(f"Training complete. Total timesteps: {global_step}")

    def _run_behavioral_eval(self, logger=None, global_step: int = 0) -> dict[str, float]:
        """Deterministic eval on each hand-crafted behavioral map (one episode per map).

        Loads data/test_behavior.pt, runs the deterministic policy on each map with its
        fixed spawn/target, and returns per-map metrics under ``test/behavioral/{name}/...``.
        Trajectory images are uploaded to WandB if a logger is provided.
        """
        import dataclasses
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
        from cogniland.env.types import CustomMapConfig

        behavioral_path = Path("data/test_behavior.pt")
        if not behavioral_path.exists():
            print("  Skipping behavioral eval: data/test_behavior.pt not found")
            return {}

        data = torch.load(str(behavioral_path), map_location="cpu", weights_only=False)
        names: list[str] = data["names"]

        model = self.model
        metrics: dict[str, float] = {}
        figs, captions, indices = [], [], []

        for i, name in enumerate(names):
            # Build a single-map env with the fixed spawn/target from custom_maps
            map_cfg = dataclasses.replace(
                self.env_config,
                custom_map=CustomMapConfig(map_name=name),
            )
            env = BatchedIslandEnv(map_cfg, num_envs=1)
            env.reset(seed=self._eval_seed)
            target_pos = env.target_pos[0].clone()

            trajectory = [env.state.position[0].tolist()]
            total_reward = 0.0
            reached = False
            alive = True

            for _ in range(self.env_config.max_steps):
                obs = env.get_obs()
                with torch.no_grad():
                    action = model.get_deterministic_action(obs)
                _obs, reward, done, info = env.step(action)
                total_reward += reward[0].item()
                trajectory.append(env.state.position[0].tolist())
                if done[0]:
                    reached = bool(info["reached"][0].item())
                    alive   = bool(info["alive"][0].item())
                    break

            outcome = "success" if reached else ("death" if not alive else "timeout")
            length  = len(trajectory) - 1

            metrics[f"test/behavioral/{name}/success"]        = float(reached)
            metrics[f"test/behavioral/{name}/return"]         = total_reward
            metrics[f"test/behavioral/{name}/episode_length"] = float(length)

            if logger is not None:
                world_map = env.env.world_maps[0]
                fig = render_trajectory(
                    world_map, trajectory, target_pos,
                    reached, i, env.compiled,
                )
                figs.append(fig)
                captions.append(
                    f"[{name}] {outcome.upper()} — {length} steps  R={total_reward:.1f}"
                )
                indices.append(i)

        metrics["test/behavioral/success_rate"] = (
            sum(v for k, v in metrics.items() if k.endswith("/success")) / len(names)
            if names else 0.0
        )

        if logger is not None and figs:
            logger.log_trajectory_images(figs, captions, indices, step=global_step)
            for fig in figs:
                plt.close(fig)

        return metrics

    def _ppo_update(self, optimizer, flat_data, advantages, returns, cfg):
        model = self.model
        N = flat_data["actions"].shape[0]
        minibatch_size = cfg.models.training.minibatch_size
        clip_coef = cfg.models.training.policy_clip_range
        vf_coef = cfg.models.training.value_loss_weight
        ent_coef = cfg.models.training.entropy_bonus_weight
        max_grad_norm = cfg.models.training.max_grad_norm

        # Compute explained variance and return estimation variance from rollout values
        rollout_values = flat_data["values"].reshape(-1)
        y_var = returns.var()
        explained_variance = (1.0 - (returns - rollout_values).var() / (y_var + 1e-8)).item()
        return_estimation_variance = rollout_values.var().item()

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
            "train/model/ppo/policy_loss": total_pg_loss / n_updates,
            "train/model/ppo/value_loss": total_vf_loss / n_updates,
            "train/model/ppo/entropy": total_entropy / n_updates,
            "train/model/ppo/clipfrac": total_clipfrac / n_updates,
            "train/model/ppo/approx_kl": total_approx_kl / n_updates,
            "train/model/ppo/explained_variance": explained_variance,
            "train/model/ppo/return_estimation_variance": return_estimation_variance,
        }

    def _run_eval(self, cfg, logger=None, global_step: int = 0, split: str = "val"):
        """Orchestrator: run deterministic + stochastic eval, merge metrics, log."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eval_cfg = cfg.logging.get("eval", {})
        n_eps_det = eval_cfg.get("deterministic_episodes", cfg.models.training.eval_episodes)
        n_eps_sto = eval_cfg.get("stochastic_episodes", cfg.models.training.eval_episodes)
        hp_danger_threshold = eval_cfg.get("hp_danger_threshold", 30.0)
        max_images = cfg.logging.get("trajectory", {}).get("max_saved_per_eval", 4)

        model = self.model

        # For test split, build a fresh env with the held-out test maps
        if split == "test":
            test_env = BatchedIslandEnv(
                self.env_config,
                num_envs=self._max_eval_eps,
                world_maps=self._test_maps,
            )
            test_env.reset(seed=self._eval_seed + 1000)
            runner = EvalRunner(test_env, self.env_config, self.device)
        else:
            runner = self._eval_runner  # reuse cached runner for val

        summarizer = CognilandSummarizer()

        det_result = runner.run(
            policy_fn=lambda obs: model.get_deterministic_action(obs),
            n_episodes=n_eps_det,
            mode="det",
            split=split,
            global_step=global_step,
            hp_danger_threshold=hp_danger_threshold,
            max_trajectory_eps=max_images,
        )
        sto_result = runner.run(
            policy_fn=lambda obs: model.get_action_and_value(obs)[0],
            n_episodes=n_eps_sto,
            mode="stoch",
            split=split,
            global_step=global_step,
            hp_danger_threshold=hp_danger_threshold,
            max_trajectory_eps=0,  # no trajectory storage for stochastic mode
        )

        # Aggregate scalar metrics
        eval_metrics: dict[str, float] = {}
        eval_metrics.update(summarizer.scalar_metrics(det_result))
        eval_metrics.update(summarizer.scalar_metrics(sto_result))

        if logger is not None:
            # Trajectory images (deterministic mode only)
            figures, captions, env_indices = [], [], []
            eval_env = runner.eval_env
            targets = det_result.initial_targets

            for i, ep in enumerate(det_result.episodes):
                if len(figures) >= max_images:
                    break
                if ep.trajectory is None or len(ep.trajectory) < 2:
                    continue
                world_map_i = eval_env.env.world_maps[ep.map_id]
                fig = render_trajectory(
                    world_map_i, ep.trajectory,
                    targets[i], ep.outcome == "success", i,
                    eval_env.compiled,
                    observed_mask=ep.observed_mask,
                )
                figures.append(fig)
                captions.append(
                    f"{ep.outcome.upper()} ({ep.episode_length} moves) "
                    f"- Time: {ep.metrics['terrain_cost']:.1f}  "
                    f"Return: {ep.total_return:.1f}"
                )
                env_indices.append(i)

            if figures:
                logger.log_trajectory_images(figures, captions, env_indices, step=global_step)
                for fig in figures:
                    plt.close(fig)

            # Terrain distribution (val only) and per-episode tables
            for result in [det_result, sto_result]:
                ns = f"{result.split}_{result.mode}"
                if split == "val":
                    logger.log_terrain_scalars(summarizer.terrain_pcts(result), step=global_step, namespace=ns)
                columns, rows = summarizer.eval_table_rows(result)
                logger.log_eval_table(columns, rows, step=global_step, namespace=ns)

        return eval_metrics
