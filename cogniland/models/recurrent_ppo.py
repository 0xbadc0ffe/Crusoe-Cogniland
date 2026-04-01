"""Recurrent PPO agent — same base architecture as PPO with a vanilla RNN for state memory.

Architecture:
    CNN (minimap) + MLP (scalars) → shared trunk → RNN → actor / critic heads

The RNN is an Elman network (single tanh hidden state), chosen for interpretability:
the hidden state h_t ∈ R^rnn_hidden_dim is a compact, analysable summary of the
agent's history.

Training uses sequence chunks from the rollout buffer to preserve temporal structure
through the RNN. Hidden states are carried across timesteps during collection and
truncated-BPTT is used during PPO updates.
"""

from __future__ import annotations

import os
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


class RecurrentActorCritic(nn.Module):
    """CNN (minimap) + MLP (scalars) → trunk → vanilla RNN → actor / critic heads.

    The RNN hidden state h_t is a single vector — no cell state, no gating —
    making it straightforward to visualise and interpret.
    """

    def __init__(
        self,
        scalar_dim: int = 5,
        minimap_channels: int = 3,
        hidden_dim: int = 256,
        rnn_hidden_dim: int = 64,
        action_dim: int = 5,
        cnn_channels: int = 32,
        cnn_out_spatial: int = 4,
        scalar_hidden: int = 64,
    ):
        super().__init__()
        self.rnn_hidden_dim = rnn_hidden_dim

        # ── Feature extraction (identical to PPO ActorCritic) ────────────
        self.cnn = nn.Sequential(
            _layer_init(nn.Conv2d(minimap_channels, cnn_channels // 2, 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            _layer_init(nn.Conv2d(cnn_channels // 2, cnn_channels, 3, padding=1)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(cnn_channels, cnn_channels, 3, padding=1)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(cnn_out_spatial),
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

        # ── Vanilla RNN (Elman network) ─────────────────────────────────
        self.rnn = nn.RNNCell(hidden_dim, rnn_hidden_dim, nonlinearity="tanh")
        # Orthogonal init for RNN weights
        nn.init.orthogonal_(self.rnn.weight_ih)
        nn.init.orthogonal_(self.rnn.weight_hh)
        nn.init.zeros_(self.rnn.bias_ih)
        nn.init.zeros_(self.rnn.bias_hh)

        # ── Actor / critic heads ─────────────────────────────────────────
        self.actor = _layer_init(nn.Linear(rnn_hidden_dim, action_dim), std=0.01)
        self.critic = _layer_init(nn.Linear(rnn_hidden_dim, 1), std=1.0)

    def _features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observation (no RNN)."""
        cnn_feat = self.cnn(obs["minimap"])
        scalar_feat = self.scalar_net(obs["scalars"])
        return self.trunk(torch.cat([cnn_feat, scalar_feat], dim=-1))

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        h: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: obs + hidden → action, log_prob, entropy, value, new_hidden.

        Args:
            obs: {"minimap": [B, C, H, W], "scalars": [B, scalar_dim]}
            h: [B, rnn_hidden_dim] — RNN hidden state from previous step
            action: optional [B] — if provided, evaluate this action instead of sampling
        Returns:
            action, log_prob, entropy, value, h_new
        """
        feat = self._features(obs)          # [B, hidden_dim]
        h_new = self.rnn(feat, h)           # [B, rnn_hidden_dim]

        logits = self.actor(h_new)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(h_new).squeeze(-1),
            h_new,
        )

    def get_value(self, obs: dict[str, torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        feat = self._features(obs)
        h_new = self.rnn(feat, h)
        return self.critic(h_new).squeeze(-1)

    @torch.no_grad()
    def get_deterministic_action(
        self, obs: dict[str, torch.Tensor], h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, h_new) for deterministic evaluation."""
        feat = self._features(obs)
        h_new = self.rnn(feat, h)
        action = self.actor(h_new).argmax(dim=-1)
        return action, h_new

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialised hidden state."""
        return torch.zeros(batch_size, self.rnn_hidden_dim, device=device)


# ---------------------------------------------------------------------------
# Rollout buffer (stores hidden states for sequence-based training)
# ---------------------------------------------------------------------------

@dataclass
class RecurrentRolloutBuffer:
    """Stores a rollout with per-step RNN hidden states for truncated BPTT."""

    obs_scalars: list[torch.Tensor] = field(default_factory=list)
    obs_minimaps: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    dones: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    hiddens: list[torch.Tensor] = field(default_factory=list)  # h_t at start of step

    def add(self, obs, action, log_prob, reward, done, value, h):
        self.obs_scalars.append(obs["scalars"])
        self.obs_minimaps.append(obs["minimap"])
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.hiddens.append(h)


# ---------------------------------------------------------------------------
# Rollout collection (carries hidden state across steps)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_rollout(env, model, obs, h, rollout_steps):
    """Collect experience, carrying RNN hidden state across steps.

    Hidden states are reset to zero for environments that just finished an episode.
    Returns buffer, next_obs, updated hidden state, and episode stats.
    """
    buffer = RecurrentRolloutBuffer()
    all_final_rewards, all_final_lengths, all_final_reached = [], [], []

    for _ in range(rollout_steps):
        action, log_prob, _, value, h_new = model(obs, h)
        next_obs, reward, done, info = env.step(action)

        buffer.add(obs, action, log_prob, reward, done, value, h)

        # Reset hidden state for finished episodes (auto-reset envs)
        if done.any():
            h_new = h_new.clone()
            h_new[done] = 0.0

        h = h_new

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
    return buffer, obs, h, episode_stats


# ---------------------------------------------------------------------------
# GAE computation (same as PPO)
# ---------------------------------------------------------------------------

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
    return advantages, returns


# ---------------------------------------------------------------------------
# Recurrent PPO Agent
# ---------------------------------------------------------------------------

class RecurrentPPOAgent:
    """Recurrent PPO agent with vanilla RNN state memory.

    Same interface as PPOAgent for build_model() compatibility.
    """

    def __init__(self, cfg, env_config: EnvConfig, device: str):
        self.env_config = env_config
        self.device = device

        m = cfg.models
        self.model = RecurrentActorCritic(
            scalar_dim=m.get("scalar_dim", 5),
            minimap_channels=m.minimap_channels,
            hidden_dim=m.hidden_dim,
            rnn_hidden_dim=m.get("rnn_hidden_dim", 128),
            action_dim=m.get("action_dim", NUM_ACTIONS),
            cnn_channels=m.get("cnn_channels", 32),
            cnn_out_spatial=m.get("cnn_out_spatial", 4),
            scalar_hidden=m.get("scalar_hidden", 64),
        ).to(device)

    # ── Inference API (compatible with PPOAgent) ────────────────────────
    # For eval runner compatibility, these stateless wrappers use zero hidden state.
    # For proper recurrent evaluation, use the model directly with hidden state.

    def get_action_and_value(self, obs, action=None):
        h = self.model.init_hidden(obs["minimap"].shape[0], obs["minimap"].device)
        act, lp, ent, val, _ = self.model(obs, h, action)
        return act, lp, ent, val

    def get_deterministic_action(self, obs):
        h = self.model.init_hidden(obs["minimap"].shape[0], obs["minimap"].device)
        act, _ = self.model.get_deterministic_action(obs, h)
        return act

    def get_value(self, obs):
        h = self.model.init_hidden(obs["minimap"].shape[0], obs["minimap"].device)
        return self.model.get_value(obs, h)

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval()

    def train_mode(self):
        self.model.train()

    # ── Training loop ───────────────────────────────────────────────────

    def train(self, cfg):
        """Recurrent PPO training: rollout with hidden state → sequence-chunk PPO updates."""
        set_reproducibility(cfg.env.map_generation.seed)
        device = self.device
        model = self.model

        logger = WandBLogger(cfg)
        print(f"Device: {device}")
        print(f"Model: recurrent_ppo (RNN hidden_dim={model.rnn_hidden_dim})")

        training_cfg = cfg.models.training
        dataset_cfg = training_cfg.get("dataset", {})
        curriculum_switch_steps = dataset_cfg.get("curriculum_switch_steps", 0)
        curriculum_switch_steps_2 = dataset_cfg.get("curriculum_switch_steps_2", 0)
        curriculum_extra_easy_radius = dataset_cfg.get("curriculum_extra_easy_radius", 25)
        curriculum_easy_radius = dataset_cfg.get("curriculum_easy_radius", 50)
        seq_len = training_cfg.get("seq_len", 16)

        dataset: MapDataset | None = None
        if dataset_cfg:
            dataset = MapDataset.load_from_config(dataset_cfg)
        if dataset is not None:
            print(f"MapDataset: train={dataset.n_train} val={dataset.n_val} test={dataset.n_test}")

        env = BatchedIslandEnv(
            self.env_config,
            num_envs=training_cfg.parallel_envs,
            world_maps=dataset.train_maps if dataset else None,
            curriculum_extra_easy_radius=curriculum_extra_easy_radius,
            curriculum_easy_radius=curriculum_easy_radius,
        )
        optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate, eps=1e-5)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {param_count:,}")

        num_envs = training_cfg.parallel_envs
        rollout_steps = training_cfg.moves_per_rollout
        total_timesteps = training_cfg.total_env_moves
        num_updates = total_timesteps // (num_envs * rollout_steps)
        print(f"Total updates: {num_updates}, Moves per update: {num_envs * rollout_steps}")
        print(f"Sequence length for BPTT: {seq_len}")

        curriculum_active = curriculum_switch_steps > 0
        if curriculum_active:
            env.set_curriculum_stage(CurriculumStage.EXTRA_EASY)
            print(f"Curriculum: EXTRA_EASY → EASY at {curriculum_switch_steps}, → NORMAL at {curriculum_switch_steps_2}")

        obs = env.reset(seed=cfg.env.map_generation.seed)
        h = model.init_hidden(num_envs, device)
        global_step = 0
        start_update = 1

        resume_path = cfg.get("resume", None)
        if resume_path:
            ckpt = load_checkpoint(resume_path, model, optimizer, device=device)
            global_step = ckpt["step"]
            start_update = global_step // (num_envs * rollout_steps) + 1
            print(f"Resumed from {resume_path} — global_step={global_step}")

        # Eval env
        eval_cfg = cfg.logging.get("eval", {})
        eval_seed = cfg.env.map_generation.seed + eval_cfg.get("eval_seed_offset", 1000)
        n_eps_det = eval_cfg.get("deterministic_episodes", training_cfg.eval_episodes)
        n_eps_sto = eval_cfg.get("stochastic_episodes", training_cfg.eval_episodes)
        max_eval_eps = max(n_eps_det, n_eps_sto)
        self.eval_env = BatchedIslandEnv(
            self.env_config,
            num_envs=max_eval_eps,
            world_maps=dataset.val_maps if dataset else None,
            curriculum_extra_easy_radius=curriculum_extra_easy_radius,
            curriculum_easy_radius=curriculum_easy_radius,
        )
        self.eval_env.reset(seed=eval_seed)
        self._test_maps = dataset.test_maps if dataset else None
        self._max_eval_eps = max_eval_eps
        self._eval_seed = eval_seed
        self._eval_runner = EvalRunner(self.eval_env, self.env_config, device)

        run_id = logger._run.id if logger.enabled and logger._run else "local"
        ckpt_dir = f"artifacts/{run_id}"
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt_path = f"{ckpt_dir}/ckpt_best.pt"
        best_val_sr = -1.0

        start_time = time.time()

        for update in range(start_update, num_updates + 1):
            if training_cfg.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                lr = frac * training_cfg.learning_rate
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            buffer, obs, h, episode_stats = _collect_rollout(env, model, obs, h, rollout_steps)
            global_step += num_envs * rollout_steps

            if curriculum_active:
                stage = env._curriculum_stage
                if stage == CurriculumStage.EXTRA_EASY and global_step >= curriculum_switch_steps:
                    env.set_curriculum_stage(CurriculumStage.EASY)
                    print(f"[Update {update}] Curriculum → EASY (step={global_step})")
                elif stage == CurriculumStage.EASY and curriculum_switch_steps_2 > 0 and global_step >= curriculum_switch_steps_2:
                    env.set_curriculum_stage(CurriculumStage.NORMAL)
                    curriculum_active = False
                    print(f"[Update {update}] Curriculum → NORMAL (step={global_step})")

            log_rollout_stats(logger, episode_stats, step=update)

            with torch.no_grad():
                next_value = model.get_value(obs, h)
            advantages, returns = _compute_gae(
                buffer, next_value,
                gamma=training_cfg.discount_factor,
                gae_lambda=training_cfg.gae_lambda,
            )

            train_metrics = self._recurrent_ppo_update(
                optimizer, buffer, advantages, returns, cfg, seq_len
            )

            current_lr = optimizer.param_groups[0]["lr"]
            train_metrics["train/model/learning_rate"] = current_lr
            sps = int(global_step / (time.time() - start_time))
            train_metrics["train/sps"] = sps
            logger.log(train_metrics, step=update)

            if update % training_cfg.eval_every_n_updates == 0:
                print(f"[Update {update}/{num_updates}] Evaluating...")
                model.eval()
                eval_metrics = self._run_eval(cfg, logger=logger, global_step=update, split="val")
                model.train()
                logger.log(eval_metrics, step=update)

                det_sr = eval_metrics.get("val_det/env/success_rate", 0.0)
                sto_sr = eval_metrics.get("val_stoch/env/success_rate", 0.0)
                print(f"  det={det_sr:.3f}  stoch={sto_sr:.3f}")

                last_ckpt_path = f"{ckpt_dir}/ckpt_last.pt"
                save_checkpoint(model, optimizer, global_step, path=last_ckpt_path)

                if det_sr > best_val_sr:
                    best_val_sr = det_sr
                    save_checkpoint(model, optimizer, global_step, path=best_ckpt_path)
                    print(f"  New best {det_sr:.3f} → {best_ckpt_path}")

            ckpt_interval = training_cfg.checkpoint_every_n_updates
            if ckpt_interval > 0 and update % ckpt_interval == 0:
                ckpt_path = f"{ckpt_dir}/ckpt_{update}.pt"
                save_checkpoint(model, optimizer, global_step, path=ckpt_path)

        # Finalise
        if best_val_sr < 0:
            save_checkpoint(model, optimizer, global_step, path=best_ckpt_path)
        print(f"Best val det SR: {best_val_sr:.3f} → {best_ckpt_path}")

        if cfg.logging.wandb.get("store_last_ckpt", False):
            logger.log_model_artifact(
                name=f"{cfg.models.name}_agent",
                path=best_ckpt_path,
                aliases=["best", f"sr{best_val_sr:.3f}"],
            )

        print("Running final test evaluation...")
        model.eval()
        test_metrics = self._run_eval(cfg, logger=logger, global_step=global_step, split="test")
        model.train()
        test_metrics["test_det/env/best_ckpt_path"] = best_ckpt_path
        logger.log(test_metrics, step=num_updates + 1)
        logger.log_final_test_summary(test_metrics)
        print(f"  test det SR: {test_metrics.get('test_det/env/success_rate', 0.0):.3f}")

        print("Running behavioral eval...")
        model.eval()
        beh_metrics = self._run_behavioral_eval(logger=logger, global_step=num_updates + 2)
        model.train()
        if beh_metrics:
            logger.log(beh_metrics, step=num_updates + 2)
            print(f"  behavioral SR: {beh_metrics.get('test/behavioral/success_rate', 0.0):.3f}")

        logger.finish()
        print(f"Training complete. Total timesteps: {global_step}")

    # ── Sequence-chunk PPO update ───────────────────────────────────────

    def _recurrent_ppo_update(self, optimizer, buffer, advantages, returns, cfg, seq_len):
        """PPO update using sequence chunks to preserve RNN temporal structure.

        Instead of shuffling individual transitions, we:
        1. Reshape the rollout [T, B] into chunks of length seq_len
        2. For each epoch, shuffle chunk indices (not individual transitions)
        3. Re-run the RNN forward pass through each chunk from its stored initial hidden state
        """
        model = self.model
        T = len(buffer.actions)
        B = buffer.actions[0].shape[0]

        clip_coef = cfg.models.training.policy_clip_range
        vf_coef = cfg.models.training.value_loss_weight
        ent_coef = cfg.models.training.entropy_bonus_weight
        max_grad_norm = cfg.models.training.max_grad_norm
        minibatch_envs = cfg.models.training.get("minibatch_envs", B)

        # Stack rollout data: [T, B, ...]
        all_minimaps = torch.stack(buffer.obs_minimaps)   # [T, B, C, H, W]
        all_scalars = torch.stack(buffer.obs_scalars)     # [T, B, scalar_dim]
        all_actions = torch.stack(buffer.actions)          # [T, B]
        all_old_logprobs = torch.stack(buffer.log_probs)  # [T, B]
        all_hiddens = torch.stack(buffer.hiddens)          # [T, B, rnn_hidden_dim]
        all_dones = torch.stack(buffer.dones)              # [T, B]

        # advantages and returns are [T, B]
        adv = advantages
        if adv.std() > 0:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Explained variance from rollout
        rollout_values = torch.stack(buffer.values).reshape(-1)
        flat_returns = returns.reshape(-1)
        y_var = flat_returns.var()
        explained_variance = (1.0 - (flat_returns - rollout_values).var() / (y_var + 1e-8)).item()

        # Split T into chunks of seq_len
        n_chunks = T // seq_len
        if n_chunks == 0:
            n_chunks = 1
            seq_len = T

        total_pg_loss = total_vf_loss = total_entropy = 0.0
        total_clipfrac = total_approx_kl = 0.0
        n_updates = 0

        for _epoch in range(cfg.models.training.epochs_per_update):
            # Shuffle env indices for this epoch
            env_perm = torch.randperm(B, device=all_actions.device)

            for mb_start in range(0, B, minibatch_envs):
                mb_end = min(mb_start + minibatch_envs, B)
                mb_envs = env_perm[mb_start:mb_end]
                G = mb_envs.shape[0]

                for chunk_idx in range(n_chunks):
                    t_start = chunk_idx * seq_len
                    t_end = t_start + seq_len

                    # Slice: [seq_len, G, ...]
                    chunk_minimaps = all_minimaps[t_start:t_end, mb_envs]
                    chunk_scalars = all_scalars[t_start:t_end, mb_envs]
                    chunk_actions = all_actions[t_start:t_end, mb_envs]
                    chunk_old_lp = all_old_logprobs[t_start:t_end, mb_envs]
                    chunk_adv = adv[t_start:t_end, mb_envs]
                    chunk_ret = returns[t_start:t_end, mb_envs]
                    chunk_dones = all_dones[t_start:t_end, mb_envs]

                    # Initial hidden state for this chunk
                    h = all_hiddens[t_start, mb_envs].detach()  # [G, rnn_hidden_dim]

                    # Forward pass through the chunk sequentially
                    new_logprobs = []
                    new_entropies = []
                    new_values = []

                    for t in range(seq_len):
                        obs_t = {
                            "minimap": chunk_minimaps[t],
                            "scalars": chunk_scalars[t],
                        }
                        _, lp, ent, val, h = model(obs_t, h, chunk_actions[t])
                        new_logprobs.append(lp)
                        new_entropies.append(ent)
                        new_values.append(val)

                        # Reset hidden for done envs within chunk
                        if chunk_dones[t].any():
                            h = h.clone()
                            h[chunk_dones[t]] = 0.0

                    new_logprobs = torch.stack(new_logprobs)    # [seq_len, G]
                    new_entropies = torch.stack(new_entropies)  # [seq_len, G]
                    new_values = torch.stack(new_values)        # [seq_len, G]

                    # PPO losses
                    log_ratio = new_logprobs - chunk_old_lp
                    ratio = log_ratio.exp()

                    pg_loss1 = -chunk_adv * ratio
                    pg_loss2 = -chunk_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    vf_loss = 0.5 * ((new_values - chunk_ret) ** 2).mean()
                    entropy_loss = new_entropies.mean()
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
            "train/model/policy_loss": total_pg_loss / n_updates,
            "train/model/value_loss": total_vf_loss / n_updates,
            "train/model/entropy": total_entropy / n_updates,
            "train/model/clipfrac": total_clipfrac / n_updates,
            "train/model/approx_kl": total_approx_kl / n_updates,
            "train/model/explained_variance": explained_variance,
        }

    # ── Evaluation (recurrent-aware) ────────────────────────────────────

    def _run_eval(self, cfg, logger=None, global_step: int = 0, split: str = "val"):
        """Run eval with recurrent hidden state carried across episode steps."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eval_cfg = cfg.logging.get("eval", {})
        n_eps_det = eval_cfg.get("deterministic_episodes", cfg.models.training.eval_episodes)
        n_eps_sto = eval_cfg.get("stochastic_episodes", cfg.models.training.eval_episodes)
        hp_danger_threshold = eval_cfg.get("hp_danger_threshold", 30.0)
        max_images = cfg.logging.get("trajectory", {}).get("max_saved_per_eval", 4)

        model = self.model

        if split == "test":
            test_env = BatchedIslandEnv(
                self.env_config,
                num_envs=self._max_eval_eps,
                world_maps=self._test_maps,
            )
            test_env.reset(seed=self._eval_seed + 1000)
            runner = EvalRunner(test_env, self.env_config, self.device)
        else:
            runner = self._eval_runner

        summarizer = CognilandSummarizer()

        # Recurrent policy functions that carry hidden state via closure
        h_det = [model.init_hidden(n_eps_det, self.device)]

        def det_policy(obs):
            act, h_new = model.get_deterministic_action(obs, h_det[0])
            h_det[0] = h_new
            return act

        h_sto = [model.init_hidden(n_eps_sto, self.device)]

        def stoch_policy(obs):
            act, _, _, _, h_new = model(obs, h_sto[0])
            h_sto[0] = h_new
            return act

        det_result = runner.run(
            policy_fn=det_policy,
            n_episodes=n_eps_det,
            mode="det",
            split=split,
            global_step=global_step,
            hp_danger_threshold=hp_danger_threshold,
            max_trajectory_eps=max_images,
        )

        h_sto[0] = model.init_hidden(n_eps_sto, self.device)
        sto_result = runner.run(
            policy_fn=stoch_policy,
            n_episodes=n_eps_sto,
            mode="stoch",
            split=split,
            global_step=global_step,
            hp_danger_threshold=hp_danger_threshold,
            max_trajectory_eps=0,
        )

        eval_metrics: dict[str, float] = {}
        eval_metrics.update(summarizer.scalar_metrics(det_result))
        eval_metrics.update(summarizer.scalar_metrics(sto_result))

        if logger is not None:
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

            for result in [det_result, sto_result]:
                ns = f"{result.split}_{result.mode}"
                if split == "val":
                    logger.log_terrain_scalars(summarizer.terrain_pcts(result),
                                               step=global_step, namespace=ns)
                columns, rows = summarizer.eval_table_rows(result)
                logger.log_eval_table(columns, rows, step=global_step, namespace=ns)

        return eval_metrics

    def _run_behavioral_eval(self, logger=None, global_step: int = 0) -> dict[str, float]:
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
            map_cfg = dataclasses.replace(
                self.env_config,
                custom_map=CustomMapConfig(map_name=name),
            )
            env = BatchedIslandEnv(map_cfg, num_envs=1)
            env.reset(seed=self._eval_seed)
            target_pos = env.target_pos[0].clone()

            trajectory = [env.state.position[0].tolist()]
            total_reward = 0.0
            reached = alive = False
            h = model.init_hidden(1, self.device)

            for _ in range(self.env_config.max_steps):
                obs = env.get_obs()
                with torch.no_grad():
                    action, h = model.get_deterministic_action(obs, h)
                _obs, reward, done, info = env.step(action)
                total_reward += reward[0].item()
                trajectory.append(env.state.position[0].tolist())
                if done[0]:
                    reached = bool(info["reached"][0].item())
                    alive = bool(info["alive"][0].item())
                    break

            outcome = "success" if reached else ("death" if not alive else "timeout")
            length = len(trajectory) - 1
            metrics[f"test/behavioral/{name}/success"] = float(reached)
            metrics[f"test/behavioral/{name}/return"] = total_reward
            metrics[f"test/behavioral/{name}/episode_length"] = float(length)

            if logger is not None:
                world_map = env.env.world_maps[0]
                fig = render_trajectory(world_map, trajectory, target_pos,
                                        reached, i, env.compiled)
                figs.append(fig)
                captions.append(f"[{name}] {outcome.upper()} — {length} steps  R={total_reward:.1f}")
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
