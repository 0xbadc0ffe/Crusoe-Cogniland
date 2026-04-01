"""DRC (Deep Repeating ConvLSTM) agent — IMPALA V-trace, self-contained.

Architecture: Guez et al. 2019 / Chung et al. 2024  DRC(D, N)
  - Two 4×4 conv layers (no nonlinearity) encode the minimap observation.
  - D separate ConvLSTM cells stacked in depth, each applied N times per step.
  - Within each think step n:
      * Cell 0 receives the encoded observation as its spatial input.
      * Cell d (d>0) receives h[d-1] (output of the previous layer) as its spatial input.
      * Additionally, pool-and-inject from each cell's own previous hidden state is
        added to its spatial input (mean+max pool → linear → channel-wise broadcast).
  - Over N think steps, information propagates through the recurrent state and the
    D-layer depth, giving the network D×N layers of sequential computation.
  - After all D×N passes, h[D-1] is spatially max-pooled, flattened, concatenated
    with the scalar embedding, and fed to a 2-layer MLP → actor/critic.

Training: IMPALA V-trace actor-critic (Espeholt et al. 2018).
  - Rollout of T steps across B parallel actors → V-trace off-policy correction.
  - Loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus (mean losses).
  - Advantages are NOT normalised (Chung et al. 2024 explicitly requires this).

Observation space (from BatchedIslandEnv.get_obs):
  "minimap": [B, 3, 2*max_ray+1, 2*max_ray+1]  (channel 0=height, 1=target, 2=vis)
  "scalars":  [B, 5]  (compass_x, compass_y, terrain_idx, resources, hp)

This module is fully self-contained and does NOT depend on ppo.py in any way.
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
# Neural network — DRC(D, N)
# ---------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    """Spatial ConvLSTM cell with non-standard tanh output gate (Jozefowicz et al. 2015).

    The forget-gate bias is initialised to 1 (standard LSTM stabilisation trick).
    Input projection uses bias; hidden-to-hidden does not (saves parameters, matches paper).
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # Input → gates (4 gates: i, f, g, o)
        self.conv_ih = nn.Conv2d(in_channels, 4 * hidden_channels, kernel_size, padding=padding)
        # Hidden → gates (no bias to avoid double-bias)
        self.conv_hh = nn.Conv2d(hidden_channels, 4 * hidden_channels, kernel_size,
                                 padding=padding, bias=False)
        # Forget gate bias = 1 for stable gradients at the start of training
        nn.init.ones_(self.conv_ih.bias[hidden_channels:2 * hidden_channels])  # forget gate

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.conv_ih(x) + self.conv_hh(h)
        ci, cf, cg, co = gates.chunk(4, dim=1)

        i = torch.sigmoid(ci)
        f = torch.sigmoid(cf)
        g = torch.tanh(cg)
        o = torch.sigmoid(co)

        c_next = f * c + i * g
        # Non-standard: tanh on c_next before output gate (paper follows Jozefowicz 2015)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class DRCActorCritic(nn.Module):
    """DRC(D, N) actor-critic.

    Args:
        minimap_channels: C_in for the encoder (must match env observation channels).
        scalar_dim:       Dimension of the scalar observation vector.
        drc_channels:     C — number of channels in all ConvLSTM hidden states.
        drc_depth:        D — number of stacked ConvLSTM layers.
        drc_thinking_steps: N — number of recurrent think-steps per environment step.
        cnn_out_spatial:  Spatial size after AdaptiveMaxPool2d at readout (s×s).
        scalar_hidden:    Width of the scalar MLP.
        hidden_dim:       Width of the shared MLP trunk.
        action_dim:       Number of discrete actions.
    """

    def __init__(
        self,
        minimap_channels: int = 3,
        scalar_dim: int = 5,
        drc_channels: int = 56,
        drc_depth: int = 3,
        drc_thinking_steps: int = 3,
        cnn_out_spatial: int = 6,
        scalar_hidden: int = 64,
        hidden_dim: int = 512,
        action_dim: int = 5,
    ):
        super().__init__()
        self.drc_channels = drc_channels
        self.drc_depth = drc_depth
        self.drc_thinking_steps = drc_thinking_steps
        self.cnn_out_spatial = cnn_out_spatial

        # ── Encoder: 2× Conv2d(4×4) with NO nonlinearity (paper §A) ──────────────
        # Padding=1 → each 4×4 conv reduces spatial dim by 2: e.g. 45→43→41
        self.encoder = nn.Sequential(
            nn.Conv2d(minimap_channels, drc_channels, kernel_size=4, padding=1, bias=True),
            nn.Conv2d(drc_channels, drc_channels, kernel_size=4, padding=1, bias=True),
        )

        # ── D separate stacked ConvLSTM cells ────────────────────────────────
        # Cell 0 input  = encoder output (C channels).
        # Cell d input  = h[d-1]         (C channels) — standard stacked LSTM.
        # All cells additionally receive pool-and-inject from their own hidden state.
        # Therefore every cell has in_channels = drc_channels.
        self.drc_cells = nn.ModuleList([
            ConvLSTMCell(drc_channels, drc_channels, kernel_size=3)
            for _ in range(drc_depth)
        ])

        # ── Pool-and-inject: one linear per layer ─────────────────────────────
        # Mean+max pool of h[d] over spatial dims → concat [B, 2C] → Linear → [B, C]
        # then broadcast spatially and added to the cell's spatial input.
        self.pool_proj = nn.ModuleList([
            nn.Linear(2 * drc_channels, drc_channels, bias=True)
            for _ in range(drc_depth)
        ])

        # ── Readout: spatial pool → flatten ───────────────────────────────────
        self.readout = nn.Sequential(
            nn.AdaptiveMaxPool2d(cnn_out_spatial),
            nn.Flatten(),
        )
        cnn_flat = drc_channels * cnn_out_spatial * cnn_out_spatial

        # ── Scalar branch ─────────────────────────────────────────────────────
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, scalar_hidden),
            nn.ReLU(),
        )

        # ── Shared MLP trunk ──────────────────────────────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(cnn_flat + scalar_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ── Heads ─────────────────────────────────────────────────────────────
        self.actor  = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation (orthogonal for linear, normal for convs)
    # ------------------------------------------------------------------

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # Normal truncated at 2σ, σ = sqrt(1/fan_in) — matches Flax default
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                std = (1.0 / fan_in) ** 0.5
                nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Policy / value heads with orthogonal (smaller std)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    # ------------------------------------------------------------------
    # Pool-and-inject helper
    # ------------------------------------------------------------------

    def _pool_inject(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Mean+max pool hidden state, project to C channels, broadcast spatially.

        Args:
            h: [B, C, H, W] hidden state.
        Returns:
            inject: [B, C, H, W] to be added to the next cell input.
        """
        B, C, H, W = h.shape
        mean_pool = h.mean(dim=(2, 3))           # [B, C]
        max_pool  = h.amax(dim=(2, 3))           # [B, C]
        pooled    = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2C]
        injected  = self.pool_proj[layer_idx](pooled)         # [B, C]
        return injected.view(B, C, 1, 1).expand_as(h)        # [B, C, H, W]

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        minimap = obs["minimap"]    # [B, 3, H, W]
        B = minimap.shape[0]
        device = minimap.device

        # Encode observation once — reused at every think step for cell 0
        enc = self.encoder(minimap)              # [B, C, H', W']
        _, C, Henc, Wenc = enc.shape

        # Initialise D hidden states to zero at the start of each environment step
        h_states = [torch.zeros(B, C, Henc, Wenc, device=device) for _ in range(self.drc_depth)]
        c_states = [torch.zeros(B, C, Henc, Wenc, device=device) for _ in range(self.drc_depth)]

        # N think steps, each sweeping through D stacked cells
        for _n in range(self.drc_thinking_steps):
            for d in range(self.drc_depth):
                # Spatial input: enc for the first layer, h[d-1] for deeper layers
                spatial_in = enc if d == 0 else h_states[d - 1]   # [B, C, H', W']
                # Pool-and-inject from this cell's own previous hidden state
                inject = self._pool_inject(h_states[d], d)         # [B, C, H', W']
                x_in   = spatial_in + inject
                h_states[d], c_states[d] = self.drc_cells[d](x_in, (h_states[d], c_states[d]))

        # Readout: AdaptiveMaxPool on the final (deepest) layer's hidden state
        cnn_feat    = self.readout(h_states[-1])           # [B, C*s*s]
        scalar_feat = self.scalar_net(obs["scalars"])       # [B, scalar_hidden]
        combined    = torch.cat([cnn_feat, scalar_feat], dim=-1)
        return self.trunk(combined)                          # [B, hidden_dim]

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(self._features(obs)).squeeze(-1)  # [B]

    def get_action_and_value(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat   = self._features(obs)
        logits = self.actor(feat)
        dist   = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(feat).squeeze(-1)

    @torch.no_grad()
    def get_deterministic_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor(self._features(obs)).argmax(dim=-1)


# ---------------------------------------------------------------------------
# Rollout storage (on-policy, for V-trace)
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Stores T steps × B envs of experience."""

    obs_minimaps: list[torch.Tensor] = field(default_factory=list)
    obs_scalars:  list[torch.Tensor] = field(default_factory=list)
    actions:      list[torch.Tensor] = field(default_factory=list)
    log_probs:    list[torch.Tensor] = field(default_factory=list)  # μ(a|x) — behaviour policy
    rewards:      list[torch.Tensor] = field(default_factory=list)
    dones:        list[torch.Tensor] = field(default_factory=list)
    values:       list[torch.Tensor] = field(default_factory=list)  # V(x_t) from behaviour

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs_minimaps.append(obs["minimap"])
        self.obs_scalars.append(obs["scalars"])
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)


# ---------------------------------------------------------------------------
# V-trace (Espeholt et al. 2018)
# ---------------------------------------------------------------------------

@torch.no_grad()
def vtrace_from_logratios(
    log_rhos: torch.Tensor,     # [T, B] — log π/μ importance weights
    rewards:  torch.Tensor,     # [T, B]
    values:   torch.Tensor,     # [T, B]   — behaviour value estimates
    bootstrap: torch.Tensor,    # [B]      — V(x_{T+1})
    dones:    torch.Tensor,     # [T, B] bool
    gamma:    float = 0.97,
    rho_bar:  float = 1.0,      # clipping threshold for IS weights in targets
    c_bar:    float = 1.0,      # clipping threshold for trace coefficients
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute V-trace targets and advantages.

    Returns:
        vs:         [T, B]  V-trace targets  (used for value loss)
        advantages: [T, B]  V-trace policy advantages  (used for policy gradient)
    """
    T, B = rewards.shape
    device = rewards.device

    rhos = torch.exp(log_rhos).clamp(max=rho_bar)   # [T, B]
    cs   = torch.exp(log_rhos).clamp(max=c_bar)      # [T, B]

    non_terminal = (~dones).float()   # [T, B]

    # Extend values with bootstrap so values_ext[t+1] is always a clean tensor.
    # values_ext: [T+1, B] where values_ext[T] = bootstrap (V(x_{T+1}))
    values_ext = torch.cat([values, bootstrap.unsqueeze(0)], dim=0)   # [T+1, B]

    # V-trace targets computed backwards (Espeholt et al. 2018, eq. 1)
    vs = torch.zeros(T + 1, B, device=device)
    vs[T] = bootstrap

    for t in reversed(range(T)):
        # δ_t = ρ_t * (r_t + γ * V(x_{t+1}) * (1-done_t) - V(x_t))
        delta_t = rhos[t] * (rewards[t] + gamma * values_ext[t + 1] * non_terminal[t] - values[t])
        # vs[t] = V(x_t) + δ_t + γ * c_t * (1-done_t) * (vs[t+1] - V(x_{t+1}))
        vs[t] = values[t] + delta_t + gamma * cs[t] * non_terminal[t] * (vs[t + 1] - values_ext[t + 1])

    vs = vs[:T]  # [T, B] — V-trace targets

    # Policy gradient advantages: ρ_t * (r_t + γ * vs[t+1] * (1-done) - V(x_t))
    vs_next = torch.cat([vs[1:], bootstrap.unsqueeze(0)], dim=0)  # [T, B]
    advantages = rhos * (rewards + gamma * vs_next * non_terminal - values)

    return vs, advantages


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_rollout(
    env: BatchedIslandEnv,
    model: DRCActorCritic,
    obs: dict[str, torch.Tensor],
    rollout_steps: int,
) -> tuple[RolloutBuffer, dict[str, torch.Tensor], dict]:
    buffer = RolloutBuffer()
    episode_info: dict[str, list] = {}

    for _ in range(rollout_steps):
        action, log_prob, _, value = model.get_action_and_value(obs)
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, log_prob, reward, done, value)

        if "final_rewards" in info:
            episode_info.setdefault("episode_rewards", []).append(info["final_rewards"])
            episode_info.setdefault("episode_lengths", []).append(info["final_lengths"])
            episode_info.setdefault("episode_reached", []).append(info["final_reached"])

        obs = next_obs

    stats = {k: torch.cat(v) for k, v in episode_info.items()} if episode_info else {}
    return buffer, obs, stats


# ---------------------------------------------------------------------------
# IMPALA V-trace update
# ---------------------------------------------------------------------------

def _vtrace_update(
    model: DRCActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    bootstrap_value: torch.Tensor,
    cfg,
) -> dict[str, float]:
    """One V-trace gradient update over the entire rollout (no minibatches, like IMPALA)."""
    training = cfg.models.training

    T = len(buffer.actions)
    B = buffer.actions[0].shape[0]
    device = buffer.actions[0].device

    # Stack rollout tensors → [T, B, ...]
    obs_minimaps = torch.stack(buffer.obs_minimaps)   # [T, B, C, H, W]
    obs_scalars  = torch.stack(buffer.obs_scalars)    # [T, B, scalar_dim]
    actions      = torch.stack(buffer.actions)        # [T, B]
    mu_log_probs = torch.stack(buffer.log_probs)      # [T, B] — behaviour log π
    rewards      = torch.stack(buffer.rewards)        # [T, B]
    dones        = torch.stack(buffer.dones)          # [T, B]
    values_mu    = torch.stack(buffer.values)         # [T, B]

    # Re-evaluate all [T, B] observations under current policy in one vectorised pass
    flat_obs = {
        "minimap":  obs_minimaps.reshape(T * B, *obs_minimaps.shape[2:]),
        "scalars":  obs_scalars.reshape(T * B, *obs_scalars.shape[2:]),
    }
    flat_actions = actions.reshape(T * B)

    _, pi_log_probs_flat, entropy_flat, values_pi_flat = model.get_action_and_value(flat_obs, flat_actions)

    pi_log_probs = pi_log_probs_flat.reshape(T, B)   # [T, B]
    entropy      = entropy_flat.reshape(T, B)         # [T, B]
    values_pi    = values_pi_flat.reshape(T, B)       # [T, B]

    # Log importance weights: log π(a|x) - log μ(a|x)
    log_rhos = pi_log_probs - mu_log_probs.detach()   # [T, B]

    # V-trace targets  (no gradient through this)
    vs, advantages = vtrace_from_logratios(
        log_rhos=log_rhos.detach(),
        rewards=rewards,
        values=values_mu.detach(),        # use behaviour-policy values as baseline
        bootstrap=bootstrap_value,
        dones=dones,
        gamma=training.discount_factor,
        rho_bar=training.get("rho_bar", 1.0),
        c_bar=training.get("c_bar", 1.0),
    )

    # ── Losses (mean per-step, as in Chung et al. 2024) ─────────────────────
    # Policy gradient — do NOT normalise advantages (paper explicitly requires this)
    pg_loss    = -(advantages.detach() * pi_log_probs).mean()

    # Value loss on current-policy values vs V-trace targets
    vf_loss    = 0.5 * ((values_pi - vs.detach()) ** 2).mean()

    entropy_loss = entropy.mean()

    loss = (pg_loss
            + training.value_loss_weight * vf_loss
            - training.entropy_bonus_weight * entropy_loss)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), training.max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        rhos = log_rhos.exp()
        mean_rho = rhos.mean().item()
        explained_var_num = (vs - values_mu).var()
        explained_var_den = vs.var()
        ev = (1.0 - explained_var_num / (explained_var_den + 1e-8)).item()

    return {
        "train/model/policy_loss":          pg_loss.item(),
        "train/model/value_loss":           vf_loss.item(),
        "train/model/entropy":              entropy_loss.item(),
        "train/model/mean_importance_ratio": mean_rho,
        "train/model/explained_variance":   ev,
    }


# ---------------------------------------------------------------------------
# DRCAgent — public API (mirrors PPOAgent interface for build_model / eval)
# ---------------------------------------------------------------------------

class DRCAgent:
    """Full DRC(D,N) agent with IMPALA V-trace training.

    Build via build_model() in cogniland/models/__init__.py, then call .train(cfg).
    Public interface (get_action_and_value, get_deterministic_action, get_value,
    parameters, eval, train_mode) is identical to PPOAgent.
    """

    def __init__(self, cfg, env_config: EnvConfig, device: str):
        self.env_config = env_config
        self.device     = device

        m = cfg.models  # shorthand
        self.model = DRCActorCritic(
            minimap_channels    = m.minimap_channels,
            scalar_dim          = m.scalar_dim,
            drc_channels        = m.get("drc_channels",       56),
            drc_depth           = m.get("drc_depth",           3),
            drc_thinking_steps  = m.get("drc_thinking_steps",  3),
            cnn_out_spatial     = m.get("cnn_out_spatial",     6),
            scalar_hidden       = m.get("scalar_hidden",       64),
            hidden_dim          = m.get("hidden_dim",         512),
            action_dim          = m.get("action_dim",  NUM_ACTIONS),
        ).to(device)

    # ── Inference API (shared with PPOAgent) ──────────────────────────────────

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

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, cfg):
        """IMPALA V-trace training loop: rollout → V-trace update → eval → checkpoint."""
        set_reproducibility(cfg.env.map_generation.seed)
        device = self.device
        model  = self.model

        logger = WandBLogger(cfg)
        print(f"Device:     {device}")
        print(f"Model:      DRC(D={cfg.models.get('drc_depth', 3)}, N={cfg.models.get('drc_thinking_steps', 3)})")

        training_cfg  = cfg.models.training
        dataset_cfg   = training_cfg.get("dataset", {})
        curriculum_switch_steps = dataset_cfg.get("curriculum_switch_steps", 0)
        curriculum_easy_radius  = dataset_cfg.get("curriculum_easy_radius",  40)

        # ── Dataset ──────────────────────────────────────────────────────────
        dataset: MapDataset | None = None
        if dataset_cfg:
            dataset = MapDataset.load_from_config(dataset_cfg)
        if dataset is not None:
            print("MapDataset loaded:")
            print(f"  train={dataset.n_train}  val={dataset.n_val}  test={dataset.n_test}")

        # ── Training environment ──────────────────────────────────────────────
        env = BatchedIslandEnv(
            self.env_config,
            num_envs=training_cfg.parallel_envs,
            world_maps=dataset.train_maps if dataset else None,
            curriculum_easy_radius=curriculum_easy_radius,
        )

        # ── Optimiser (Adam, β₁=0.9, β₂=0.99, ε=1.5625e-7 — paper values) ──
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_cfg.learning_rate,
            betas=(training_cfg.get("adam_beta1", 0.9),
                   training_cfg.get("adam_beta2", 0.99)),
            eps=training_cfg.get("adam_eps", 1.5625e-7),
        )

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {param_count:,}")

        num_envs      = training_cfg.parallel_envs
        rollout_steps = training_cfg.moves_per_rollout
        total_steps   = training_cfg.total_env_moves
        num_updates   = total_steps // (num_envs * rollout_steps)
        print(f"Total updates:  {num_updates}")
        print(f"Moves/update:   {num_envs * rollout_steps:,}  "
              f"({num_envs} envs × {rollout_steps} steps)")

        # ── Curriculum ───────────────────────────────────────────────────────
        curriculum_active = curriculum_switch_steps > 0
        if curriculum_active:
            env.set_curriculum_stage(CurriculumStage.EASY)
            print(f"Curriculum:     EASY until global_step={curriculum_switch_steps:,}")

        obs          = env.reset(seed=cfg.env.map_generation.seed)
        global_step  = 0
        start_update = 1

        # ── Resume ───────────────────────────────────────────────────────────
        resume_path = cfg.get("resume", None)
        if resume_path:
            ckpt = load_checkpoint(resume_path, model, optimizer, device=device)
            global_step  = ckpt["step"]
            start_update = global_step // (num_envs * rollout_steps) + 1
            print(f"Resumed:        {resume_path}  step={global_step:,}")

        # ── Eval env (val split, pre-cached) ─────────────────────────────────
        eval_cfg      = cfg.logging.get("eval", {})
        eval_seed     = cfg.env.map_generation.seed + eval_cfg.get("eval_seed_offset", 1000)
        n_eps_det     = eval_cfg.get("deterministic_episodes", training_cfg.eval_episodes)
        n_eps_sto     = eval_cfg.get("stochastic_episodes",    training_cfg.eval_episodes)
        max_eval_eps  = max(n_eps_det, n_eps_sto)
        self.eval_env = BatchedIslandEnv(
            self.env_config,
            num_envs=max_eval_eps,
            world_maps=dataset.val_maps if dataset else None,
            curriculum_easy_radius=curriculum_easy_radius,
        )
        self.eval_env.reset(seed=eval_seed)

        self._test_maps     = dataset.test_maps if dataset else None
        self._max_eval_eps  = max_eval_eps
        self._eval_seed     = eval_seed
        self._eval_runner   = EvalRunner(self.eval_env, self.env_config, device)

        # ── Checkpoint dir ───────────────────────────────────────────────────
        run_id        = logger._run.id if logger.enabled and logger._run else "local"
        ckpt_dir      = f"artifacts/{run_id}"
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt_path = f"{ckpt_dir}/ckpt_best.pt"
        best_val_sr    = -1.0

        start_time = time.time()

        # ── Main loop ─────────────────────────────────────────────────────────
        for update in range(start_update, num_updates + 1):

            # LR linear anneal: 4e-4 → 4e-6  (paper schedule)
            if training_cfg.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                lr_min = training_cfg.get("learning_rate_min", 4e-6)
                lr     = lr_min + frac * (training_cfg.learning_rate - lr_min)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            # ── Rollout ──────────────────────────────────────────────────────
            buffer, obs, episode_stats = _collect_rollout(env, model, obs, rollout_steps)
            global_step += num_envs * rollout_steps

            if curriculum_active and global_step >= curriculum_switch_steps:
                env.set_curriculum_stage(CurriculumStage.NORMAL)
                curriculum_active = False
                print(f"[Update {update}] Curriculum → NORMAL (step={global_step:,})")

            log_rollout_stats(logger, episode_stats, step=update)

            # ── Bootstrap value ──────────────────────────────────────────────
            with torch.no_grad():
                bootstrap_value = model.get_value(obs)   # [B]

            # ── V-trace update ───────────────────────────────────────────────
            model.train()
            train_metrics = _vtrace_update(model, optimizer, buffer, bootstrap_value, cfg)

            current_lr = optimizer.param_groups[0]["lr"]
            train_metrics["train/model/learning_rate"] = current_lr
            train_metrics["train/sps"] = int(global_step / (time.time() - start_time))
            logger.log(train_metrics, step=update)

            # ── Periodic eval ────────────────────────────────────────────────
            if update % training_cfg.eval_every_n_updates == 0:
                print(f"[Update {update}/{num_updates}] Evaluating...")
                model.eval()
                eval_metrics = self._run_eval(cfg, logger=logger, global_step=update, split="val")
                model.train()
                logger.log(eval_metrics, step=update)

                det_sr = eval_metrics.get("val_det/env/success_rate",   0.0)
                sto_sr = eval_metrics.get("val_stoch/env/success_rate", 0.0)
                print(f"  det={det_sr:.3f}  stoch={sto_sr:.3f}")

                if det_sr > best_val_sr:
                    best_val_sr = det_sr
                    save_checkpoint(model, optimizer, global_step, path=best_ckpt_path)
                    print(f"  ✓ new best: {det_sr:.3f}  → {best_ckpt_path}")

            # ── Periodic checkpoint ──────────────────────────────────────────
            ckpt_interval = training_cfg.checkpoint_every_n_updates
            if ckpt_interval > 0 and update % ckpt_interval == 0:
                ckpt_path = f"{ckpt_dir}/ckpt_{update}.pt"
                save_checkpoint(model, optimizer, global_step, path=ckpt_path)
                print(f"  Checkpoint → {ckpt_path}")

        # ── Finalise ──────────────────────────────────────────────────────────
        if best_val_sr < 0:
            save_checkpoint(model, optimizer, global_step, path=best_ckpt_path)
        print(f"Best val det SR: {best_val_sr:.3f}  → {best_ckpt_path}")

        if cfg.logging.wandb.get("store_last_ckpt", False):
            logger.log_model_artifact(
                name=f"{cfg.models.name}_agent",
                path=best_ckpt_path,
                aliases=["best", f"sr{best_val_sr:.3f}"],
            )

        # Final test eval
        print("Running final test evaluation...")
        model.eval()
        test_metrics = self._run_eval(cfg, logger=logger, global_step=global_step, split="test")
        model.train()
        test_metrics["test_det/env/best_ckpt_path"] = best_ckpt_path
        logger.log(test_metrics, step=num_updates + 1)
        logger.log_final_test_summary(test_metrics)
        test_sr = test_metrics.get("test_det/env/success_rate", 0.0)
        print(f"  test det SR: {test_sr:.3f}")

        # Behavioral eval
        print("Running behavioral eval...")
        model.eval()
        beh_metrics = self._run_behavioral_eval(logger=logger, global_step=num_updates + 2)
        model.train()
        if beh_metrics:
            logger.log(beh_metrics, step=num_updates + 2)
            print(f"  behavioral SR: {beh_metrics.get('test/behavioral/success_rate', 0.0):.3f}")

        logger.finish()
        print(f"Training complete.  total_steps={global_step:,}")

    # ── Evaluation helpers (identical to PPOAgent._run_eval / _run_behavioral_eval) ──

    def _run_eval(self, cfg, logger=None, global_step: int = 0, split: str = "val",
                  c_rad: int = 40) -> dict[str, float]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eval_cfg            = cfg.logging.get("eval", {})
        n_eps_det           = eval_cfg.get("deterministic_episodes", cfg.models.training.eval_episodes)
        n_eps_sto           = eval_cfg.get("stochastic_episodes",    cfg.models.training.eval_episodes)
        hp_danger_threshold = eval_cfg.get("hp_danger_threshold", 30.0)
        max_images          = cfg.logging.get("trajectory", {}).get("max_saved_per_eval", 4)

        model = self.model

        if split == "test":
            test_env = BatchedIslandEnv(
                self.env_config,
                num_envs=self._max_eval_eps,
                world_maps=self._test_maps,
                curriculum_easy_radius=c_rad,
            )
            test_env.reset(seed=self._eval_seed + 1000)
            runner = EvalRunner(test_env, self.env_config, self.device)
        else:
            runner = self._eval_runner

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
            max_trajectory_eps=0,
        )

        eval_metrics: dict[str, float] = {}
        eval_metrics.update(summarizer.scalar_metrics(det_result))
        eval_metrics.update(summarizer.scalar_metrics(sto_result))

        if logger is not None:
            figures, captions, env_indices = [], [], []
            eval_env = runner.eval_env
            targets  = det_result.initial_targets

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

        data  = torch.load(str(behavioral_path), map_location="cpu", weights_only=False)
        names: list[str] = data["names"]

        model   = self.model
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

            trajectory   = [env.state.position[0].tolist()]
            total_reward = 0.0
            reached = alive = False

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