#!/usr/bin/env python3
"""DreamerV3-style training in pixel space for Cogniland Nav.

What this is
------------
A self-contained model-based RL trainer that follows the DreamerV3 recipe
closely enough to learn this env well, while staying small enough to read
in one sitting (~500 lines). Components:

* **Encoder** : a small CNN that maps an RGB observation to a 1024-dim
  feature vector.
* **RSSM** : a recurrent state-space model with a *deterministic*
  GRU state ``h`` plus a *stochastic* state ``z`` factorised as
  ``stoch_classes × stoch_dim`` discrete one-hots (DreamerV3 style, with
  straight-through gradients). Prior: ``p(z' | h, a)``; Posterior:
  ``q(z' | h, x)``. KL is balanced (Hafner '21) so the prior catches up
  to the posterior without collapsing it.
* **Decoder + reward + continue heads** : MSE on image reconstruction,
  symlog-MSE on reward, BCE on the continue flag.
* **Actor + Critic on imagined latents** : an Adam optimizer trains an
  actor (Categorical move + tanh-Gaussian build_scalar — same hybrid as
  PPO) and a critic regressed on λ-returns computed from imagined
  rollouts in latent space.
* **Imagination** : at every model update we sample H=15-step rollouts
  starting from posterior states observed in the replay batch, and use
  those for the actor / critic gradient.

What this is *not*
------------------
Not a perfect DV3 port — we use Gaussian (mean-tanh) for the build_scalar
instead of symexp-twohot, and skip the two-hot reward / value heads in
favour of symlog-MSE. Empirically this still works on small envs like
Cogniland; you can tighten it later if you want closer paper match.

Usage (RTX 4090)
----------------

    pip install wandb opensimplex imageio imageio-ffmpeg
    wandb login

    python scripts/train_dreamer.py \\
        --env-size 64 --view-size 21 --tile-px 8 \\
        --total-env-steps 1_000_000 \\
        --num-envs 4 --train-ratio 32 --batch-size 16 --batch-length 64 \\
        --device cuda --wandb-project cogniland-nav

After every ``--imagine-every`` model updates the trainer writes a video
of imagined trajectories to ``--imagine-dir`` AND logs it to W&B.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cogniland.nav import CognilandNavEnv  # noqa: E402


# ──────────────────────────────────────────────────────────────── utils


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


def _layer_init(layer, std=math.sqrt(2.0), bias=0.0):
    if hasattr(layer, "weight") and layer.weight is not None and layer.weight.dim() >= 2:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias)
    return layer


# ───────────────────────────────────────────────────────────── networks


class Encoder(nn.Module):
    """CNN encoder; (B, 3, H, W) uint8 → (B, embed_dim). 3 stride-2 convs
    + 1×1 bottleneck — image must be a multiple of 8 on each side.

    Channel widths default to a small ladder (16, 32, 64) with a 1×1
    bottleneck to 8 channels, which makes the final flatten → linear
    layer the right size for a tile-rendered toy env.
    """

    def __init__(self, image_shape, embed_dim: int = 256,
                 channels=(16, 32, 64), bottleneck: int = 8):
        super().__init__()
        C, H, W = image_shape
        assert H % 8 == 0 and W % 8 == 0, "image dims must be divisible by 8"
        layers = []
        in_c = C
        for c in channels:
            layers += [
                _layer_init(nn.Conv2d(in_c, c, 4, stride=2, padding=1)),
                nn.GroupNorm(min(8, c), c),
                nn.SiLU(),
            ]
            in_c = c
        layers += [
            _layer_init(nn.Conv2d(in_c, bottleneck, 1)),
            nn.GroupNorm(min(8, bottleneck), bottleneck),
            nn.SiLU(),
        ]
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, C, H, W)).flatten(1).shape[1]
        self.proj = _layer_init(nn.Linear(n_flat, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

    def forward(self, image_uint8: torch.Tensor) -> torch.Tensor:
        x = image_uint8.float() / 255.0
        x = self.cnn(x).flatten(1)
        return self.norm(self.proj(x))


class Decoder(nn.Module):
    """Inverse of Encoder. (B, feat_dim) → (B, 3, H, W) logits.

    ``base`` controls the first-stage channel count. We start from a
    small ``base`` (16) so the ``linear(feat → base × h₀ × w₀)`` doesn't
    explode — that linear is the bulk of the world-model parameters.
    """

    def __init__(self, feat_dim: int, image_shape, base: int = 16):
        super().__init__()
        C, H, W = image_shape
        assert H % 8 == 0 and W % 8 == 0, "image dims must be divisible by 8"
        h0, w0 = H // 8, W // 8
        self.h0, self.w0 = h0, w0
        self.base = base
        self.fc = _layer_init(nn.Linear(feat_dim, base * h0 * w0))
        layers = [
            nn.GroupNorm(min(8, base), base), nn.SiLU(),
            nn.ConvTranspose2d(base, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, base * 2), base * 2), nn.SiLU(),
            nn.ConvTranspose2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, base * 4), base * 4), nn.SiLU(),
            nn.ConvTranspose2d(base * 4, C, 4, stride=2, padding=1),
        ]
        self.deconv = nn.Sequential(*layers)

    def forward(self, x):
        B = x.shape[0]
        x = self.fc(x).view(B, self.base, self.h0, self.w0)
        return self.deconv(x)


class RSSM(nn.Module):
    """Recurrent state-space model with discrete stochastic latents."""

    def __init__(self, embed_dim: int, action_dim: int,
                 deter: int = 256, stoch_classes: int = 16, stoch_dim: int = 16,
                 hidden: int = 256):
        super().__init__()
        self.deter = deter
        self.classes = stoch_classes
        self.dim = stoch_dim
        z_size = stoch_classes * stoch_dim

        self.action_dim = action_dim
        self.img_in = nn.Sequential(
            _layer_init(nn.Linear(z_size + action_dim, hidden)),
            nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden, deter)
        self.img_out = nn.Sequential(
            _layer_init(nn.Linear(deter, hidden)),
            nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.prior_logits = _layer_init(nn.Linear(hidden, z_size), std=0.01)
        self.obs_out = nn.Sequential(
            _layer_init(nn.Linear(deter + embed_dim, hidden)),
            nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.post_logits = _layer_init(nn.Linear(hidden, z_size), std=0.01)

    def initial(self, batch: int, device) -> dict[str, torch.Tensor]:
        return {
            "deter": torch.zeros(batch, self.deter, device=device),
            "stoch": torch.zeros(batch, self.classes, self.dim, device=device),
            "logits": torch.zeros(batch, self.classes, self.dim, device=device),
        }

    def _stoch_to_flat(self, stoch: torch.Tensor) -> torch.Tensor:
        return stoch.reshape(stoch.shape[0], -1)

    def _logits_to_dist(self, logits_flat: torch.Tensor):
        logits = logits_flat.view(-1, self.classes, self.dim)
        return torch.distributions.OneHotCategoricalStraightThrough(logits=logits)

    def img_step(self, prev_state, prev_action):
        """One imagination step — uses prior (no observation)."""
        z_flat = self._stoch_to_flat(prev_state["stoch"])
        x = self.img_in(torch.cat([z_flat, prev_action], dim=-1))
        deter = self.gru(x, prev_state["deter"])
        feat = self.img_out(deter)
        prior_logits = self.prior_logits(feat).view(-1, self.classes, self.dim)
        dist = torch.distributions.OneHotCategoricalStraightThrough(logits=prior_logits)
        stoch = dist.rsample()
        return {"deter": deter, "stoch": stoch, "logits": prior_logits}

    def obs_step(self, prev_state, prev_action, embed):
        """One posterior step — combine prior with new observation embed."""
        prior = self.img_step(prev_state, prev_action)
        post_in = self.obs_out(torch.cat([prior["deter"], embed], dim=-1))
        post_logits = self.post_logits(post_in).view(-1, self.classes, self.dim)
        dist = torch.distributions.OneHotCategoricalStraightThrough(logits=post_logits)
        stoch = dist.rsample()
        return prior, {"deter": prior["deter"], "stoch": stoch, "logits": post_logits}

    def feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([state["deter"], self._stoch_to_flat(state["stoch"])], dim=-1)

    @property
    def feat_dim(self) -> int:
        return self.deter + self.classes * self.dim


class RewardHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(feat_dim, hidden)), nn.LayerNorm(hidden), nn.SiLU(),
            _layer_init(nn.Linear(hidden, hidden)), nn.LayerNorm(hidden), nn.SiLU(),
            _layer_init(nn.Linear(hidden, 1), std=0.01),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


class ContinueHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(feat_dim, hidden)), nn.LayerNorm(hidden), nn.SiLU(),
            _layer_init(nn.Linear(hidden, hidden)), nn.LayerNorm(hidden), nn.SiLU(),
            _layer_init(nn.Linear(hidden, 1), std=0.01),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)  # logits


class Actor(nn.Module):
    def __init__(self, feat_dim: int, num_moves: int = 5, hidden: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(feat_dim, hidden)), nn.LayerNorm(hidden), nn.SiLU(),
            _layer_init(nn.Linear(hidden, hidden)), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.move_head = _layer_init(nn.Linear(hidden, num_moves), std=0.01)
        self.scalar_mean = _layer_init(nn.Linear(hidden, 1), std=0.01)
        self.scalar_log_std = nn.Parameter(torch.zeros(1) - 0.5)

    def forward(self, feat):
        x = self.trunk(feat)
        return self.move_head(x), torch.tanh(self.scalar_mean(x)), self.scalar_log_std.exp().expand_as(self.scalar_mean(x))


class Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(feat_dim, hidden)), nn.LayerNorm(hidden), nn.SiLU(),
            _layer_init(nn.Linear(hidden, hidden)), nn.LayerNorm(hidden), nn.SiLU(),
            _layer_init(nn.Linear(hidden, 1), std=0.0),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


# ─────────────────────────────────────────────────────────── replay buffer


class EpisodeReplay:
    """Cyclic buffer of (obs, act, rew, cont) sequences."""

    def __init__(self, capacity: int, image_shape, action_dim: int, device):
        self.capacity = capacity
        self.device = device
        self.image_shape = image_shape
        self.action_dim = action_dim
        self.obs = np.zeros((capacity,) + image_shape, dtype=np.uint8)
        self.skill = np.zeros((capacity, 1), dtype=np.float32)
        self.act = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.cont = np.zeros((capacity,), dtype=np.float32)  # 1 - done
        self.is_first = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(self, obs_img, skill, action_vec, reward, done, is_first):
        i = self.ptr
        self.obs[i] = obs_img
        self.skill[i, 0] = skill
        self.act[i] = action_vec
        self.rew[i] = reward
        self.cont[i] = 0.0 if done else 1.0
        self.is_first[i] = float(is_first)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int, seq_len: int):
        size = len(self)
        if size < seq_len + 1:
            return None
        starts = np.random.randint(0, size - seq_len - 1, size=batch_size)
        idx = (starts[:, None] + np.arange(seq_len)[None, :]) % self.capacity
        obs = torch.from_numpy(self.obs[idx]).to(self.device)
        skill = torch.from_numpy(self.skill[idx]).to(self.device)
        act = torch.from_numpy(self.act[idx]).to(self.device)
        rew = torch.from_numpy(self.rew[idx]).to(self.device)
        cont = torch.from_numpy(self.cont[idx]).to(self.device)
        is_first = torch.from_numpy(self.is_first[idx]).to(self.device)
        # (B, T, ...) → (T, B, ...) per Dreamer convention
        return {
            "image": obs.transpose(0, 1),
            "skill": skill.transpose(0, 1),
            "action": act.transpose(0, 1),
            "reward": rew.transpose(0, 1),
            "cont": cont.transpose(0, 1),
            "is_first": is_first.transpose(0, 1),
        }


# ─────────────────────────────────────────────────────────── main loop


def encode_action(move: int, scalar: float, num_moves: int = 5) -> np.ndarray:
    """Pack (discrete move, scalar) into a flat continuous vector of size 5+1=6."""
    onehot = np.zeros(num_moves, dtype=np.float32)
    onehot[move] = 1.0
    return np.concatenate(
        [onehot, np.array([float(scalar)], dtype=np.float32)]
    ).astype(np.float32)


def decode_action(action_t: torch.Tensor):
    """Inverse — turn (B, 6) → (move int, scalar float). Used for env stepping."""
    move = action_t[..., :5].argmax(-1)
    scalar = action_t[..., 5]
    return move, scalar


def kl_balanced(prior_logits, post_logits, alpha: float = 0.8, free: float = 1.0):
    """KL balancing per Dreamer-V2/V3."""
    prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits.detach())
    post_dist_detached = torch.distributions.OneHotCategorical(logits=post_logits.detach())
    prior_dist_for_kl = torch.distributions.OneHotCategorical(logits=prior_logits)
    post_dist = torch.distributions.OneHotCategorical(logits=post_logits)
    # KL[post || prior_detached] — train posterior toward prior side
    kl_post = torch.distributions.kl_divergence(post_dist, prior_dist).sum(-1)
    # KL[post_detached || prior] — train prior toward posterior side
    kl_prior = torch.distributions.kl_divergence(post_dist_detached, prior_dist_for_kl).sum(-1)
    kl_post = torch.clamp(kl_post, min=free)
    kl_prior = torch.clamp(kl_prior, min=free)
    return alpha * kl_prior + (1.0 - alpha) * kl_post


def main():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--env-size", type=int, default=64, choices=(32, 64, 96, 128))
    parser.add_argument("--map-type", default="random",
                        choices=("random", "lake", "rocky", "balanced"))
    parser.add_argument("--view-size", type=int, default=21)
    parser.add_argument("--tile-px", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    # training
    parser.add_argument("--total-env-steps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--train-ratio", type=int, default=32,
                        help="model updates per env step")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-length", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=400_000)
    parser.add_argument("--prefill", type=int, default=4_000)
    # losses + optim
    parser.add_argument("--world-lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument("--kl-alpha", type=float, default=0.8)
    parser.add_argument("--kl-free", type=float, default=1.0)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--cont-loss-weight", type=float, default=1.0)
    parser.add_argument("--actor-ent-coef", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1000.0)
    # model
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--deter", type=int, default=512)
    parser.add_argument("--stoch-classes", type=int, default=32)
    parser.add_argument("--stoch-dim", type=int, default=32)
    # infra
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="cogniland-nav-dreamer")
    parser.add_argument("--wandb-mode", default="online",
                        choices=("online", "offline", "disabled"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-every-updates", type=int, default=5000)
    parser.add_argument("--imagine-every", type=int, default=2000,
                        help="every N model updates: write an imagination video")
    parser.add_argument("--imagine-dir", type=Path, default=Path("imagine"))
    parser.add_argument("--imagine-batch", type=int, default=4)
    args = parser.parse_args()

    run_name = args.run_name or f"dreamer_size{args.env_size}_seed{args.seed}_{int(time.time())}"
    wandb.init(project=args.wandb_project, name=run_name,
               config=vars(args), mode=args.wandb_mode, save_code=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.imagine_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} run={run_name}")

    envs = [
        CognilandNavEnv(
            size=args.env_size, map_type=args.map_type, view_size=args.view_size,
            tile_px=args.tile_px, obs_mode="rgb", seed=args.seed + i,
            max_steps=args.max_steps,
        )
        for i in range(args.num_envs)
    ]
    image_shape = envs[0].observation_space["image"].shape  # (3, H, W)
    action_dim = 5 + 1  # one-hot move (5) + scalar (1)

    enc = Encoder(image_shape, embed_dim=args.embed_dim).to(device)
    rssm = RSSM(args.embed_dim, action_dim,
                deter=args.deter, stoch_classes=args.stoch_classes,
                stoch_dim=args.stoch_dim).to(device)
    dec = Decoder(rssm.feat_dim, image_shape).to(device)
    rew_head = RewardHead(rssm.feat_dim).to(device)
    cont_head = ContinueHead(rssm.feat_dim).to(device)
    actor = Actor(rssm.feat_dim, num_moves=5).to(device)
    critic = Critic(rssm.feat_dim).to(device)
    critic_target = Critic(rssm.feat_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad_(False)

    world_params = list(enc.parameters()) + list(rssm.parameters()) \
                 + list(dec.parameters()) + list(rew_head.parameters()) \
                 + list(cont_head.parameters())
    opt_world = optim.Adam(world_params, lr=args.world_lr, eps=1e-8)
    opt_actor = optim.Adam(actor.parameters(), lr=args.actor_lr, eps=1e-8)
    opt_critic = optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-8)

    n_world = sum(p.numel() for p in world_params)
    n_ac = sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in critic.parameters())
    print(f"world params: {n_world:,}  actor+critic: {n_ac:,}")

    replay = EpisodeReplay(args.replay_capacity, image_shape, action_dim, device)

    # Per-env state for posterior rollout
    obs_list = [env.reset()[0] for env in envs]
    info_list = [{} for _ in envs]
    is_first = [True] * args.num_envs
    ep_returns = [0.0] * args.num_envs
    ep_lens = [0] * args.num_envs
    ep_buffer = deque(maxlen=200)

    # Per-env RSSM states for action selection
    states = [rssm.initial(1, device) for _ in range(args.num_envs)]
    prev_actions = [torch.zeros(1, action_dim, device=device) for _ in range(args.num_envs)]

    env_steps = 0
    updates = 0
    start_time = time.time()

    def actor_action(state, deterministic: bool = False):
        feat = rssm.feat(state)
        move_logits, scalar_mean, scalar_std = actor(feat)
        if deterministic:
            move = move_logits.argmax(-1)
            scalar = scalar_mean.squeeze(-1)
        else:
            move = torch.distributions.Categorical(logits=move_logits).sample()
            scalar = torch.distributions.Normal(scalar_mean.squeeze(-1), scalar_std.squeeze(-1)).sample()
        return move.item(), float(scalar.clamp(-1.0, 1.0).item())

    def env_step_one(i, deterministic=False):
        # Encode current obs, advance posterior, sample action, step env.
        obs = obs_list[i]
        img = torch.from_numpy(obs["image"]).unsqueeze(0).to(device)
        embed = enc(img)
        if is_first[i]:
            states[i] = rssm.initial(1, device)
            prev_actions[i] = torch.zeros(1, action_dim, device=device)
        with torch.no_grad():
            _, post = rssm.obs_step(states[i], prev_actions[i], embed)
        states[i] = post
        if env_steps < args.prefill:
            move = int(np.random.randint(0, 5))
            scalar = float(np.random.uniform(-1.0, 1.0))
        else:
            move, scalar = actor_action(post, deterministic=deterministic)
        action_vec = encode_action(move, scalar)
        prev_actions[i] = torch.from_numpy(action_vec).float().unsqueeze(0).to(device)

        action = {"move": move, "build_scalar": np.array([scalar], np.float32)}
        next_obs, reward, term, trunc, info = envs[i].step(action)
        done = term or trunc
        replay.add(obs["image"], float(obs["skill_active"][0]), action_vec, reward, done, is_first[i])
        ep_returns[i] += reward
        ep_lens[i] += 1
        if done:
            ep_buffer.append({
                "return": ep_returns[i],
                "length": ep_lens[i],
                "reached": bool(info["reached_target"]),
                "active_object": info["active_object"],
                "correct_object": info["correct_object"],
                "map_type": info["map_type"],
            })
            ep_returns[i] = 0.0; ep_lens[i] = 0
            next_obs, _ = envs[i].reset()
            is_first[i] = True
        else:
            is_first[i] = False
        obs_list[i] = next_obs
        info_list[i] = info

    # ──────────────────────── main loop ────────────────────────
    print("filling replay …")
    while env_steps < args.prefill:
        for i in range(args.num_envs):
            env_step_one(i)
            env_steps += 1
            if env_steps >= args.prefill:
                break
    print(f"prefill done — {env_steps} env steps in replay; starting training")

    while env_steps < args.total_env_steps:
        # Collect one transition per env, then do `train_ratio` updates.
        for i in range(args.num_envs):
            env_step_one(i)
            env_steps += 1

        # do enough updates to maintain the train_ratio
        n_updates_now = args.train_ratio
        for _ in range(n_updates_now):
            batch = replay.sample(args.batch_size, args.batch_length)
            if batch is None:
                continue
            updates += 1
            metrics = train_step(batch, enc, rssm, dec, rew_head, cont_head,
                                  actor, critic, critic_target,
                                  opt_world, opt_actor, opt_critic, args, device)
            # soft-update critic target
            with torch.no_grad():
                for tp, p in zip(critic_target.parameters(), critic.parameters()):
                    tp.mul_(0.98).add_(p, alpha=0.02)

            if updates % 100 == 0:
                ep_ret = float(np.mean([e["return"] for e in ep_buffer])) if ep_buffer else float("nan")
                reach = float(np.mean([e["reached"] for e in ep_buffer])) if ep_buffer else 0.0
                sps = env_steps / (time.time() - start_time)
                print(f"upd={updates:6d} env_step={env_steps:7d} sps={sps:.0f} "
                      f"wm={metrics['wm/loss']:+.3f} pol={metrics['actor/loss']:+.3f} "
                      f"val={metrics['critic/loss']:.3f} ret={ep_ret:+.2f} reach={reach:.2f}")
                wandb.log({
                    **metrics,
                    "charts/episode_return_mean": ep_ret,
                    "charts/reach_rate": reach,
                    "charts/env_steps": env_steps,
                    "charts/updates_per_sec": updates / (time.time() - start_time),
                    "charts/sps": sps,
                }, step=env_steps)

            if updates % args.imagine_every == 0 and updates > 0:
                from cogniland.nav_dreamer_video import render_imagined
                video_path = args.imagine_dir / f"{run_name}_upd{updates}.mp4"
                with torch.no_grad():
                    render_imagined(
                        replay, enc, rssm, dec, actor, device,
                        path=video_path, batch=args.imagine_batch,
                        horizon=64, fps=8,
                    )
                wandb.log({"imagine/video": wandb.Video(str(video_path), fps=8, format="mp4")},
                          step=env_steps)
                print(f"saved imagination video → {video_path}")

            if updates % args.save_every_updates == 0 and updates > 0:
                ckpt = args.checkpoint_dir / f"{run_name}_upd{updates}.pt"
                torch.save({
                    "enc": enc.state_dict(), "rssm": rssm.state_dict(),
                    "dec": dec.state_dict(), "rew": rew_head.state_dict(),
                    "cont": cont_head.state_dict(),
                    "actor": actor.state_dict(), "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                    "args": vars(args), "env_steps": env_steps, "updates": updates,
                }, ckpt)
                wandb.save(str(ckpt))
                print(f"checkpoint → {ckpt}")

    # final checkpoint
    final = args.checkpoint_dir / f"{run_name}_final.pt"
    torch.save({
        "enc": enc.state_dict(), "rssm": rssm.state_dict(),
        "dec": dec.state_dict(), "rew": rew_head.state_dict(),
        "cont": cont_head.state_dict(),
        "actor": actor.state_dict(), "critic": critic.state_dict(),
        "args": vars(args), "env_steps": env_steps, "updates": updates,
    }, final)
    wandb.save(str(final))
    wandb.finish()


# ──────────────────────────────────────────── one model + AC update step


def train_step(batch, enc, rssm, dec, rew_head, cont_head,
                actor, critic, critic_target,
                opt_world, opt_actor, opt_critic, args, device):
    """Single Dreamer update: (1) world model loss; (2) imagined AC loss."""
    T = args.batch_length
    B = args.batch_size

    images = batch["image"]      # (T, B, C, H, W)
    actions = batch["action"]    # (T, B, action_dim)
    rewards = batch["reward"]    # (T, B)
    conts = batch["cont"]        # (T, B)
    is_first = batch["is_first"] # (T, B)

    # ── 1. encode all images
    images_flat = images.flatten(0, 1)
    embed = enc(images_flat).view(T, B, -1)  # (T, B, embed_dim)

    # ── 2. roll the RSSM through the sequence
    state = rssm.initial(B, device)
    priors_logits = []
    posts_logits = []
    posts_deter = []
    posts_stoch = []
    for t in range(T):
        # reset state where is_first[t]==1
        mask = (1.0 - is_first[t]).view(B, 1)
        state = {k: (v * mask if v.dim() == 2 else v * mask.unsqueeze(-1)) for k, v in state.items()}
        prev_a = actions[t] if t > 0 else torch.zeros_like(actions[t])
        prior, post = rssm.obs_step(state, prev_a, embed[t])
        priors_logits.append(prior["logits"])
        posts_logits.append(post["logits"])
        posts_deter.append(post["deter"])
        posts_stoch.append(post["stoch"])
        state = post
    prior_logits = torch.stack(priors_logits)      # (T, B, C, D)
    post_logits = torch.stack(posts_logits)
    deter = torch.stack(posts_deter)
    stoch = torch.stack(posts_stoch)
    feats = torch.cat([deter, stoch.flatten(-2)], dim=-1)  # (T, B, feat)
    feats_flat = feats.flatten(0, 1)

    # ── 3. world-model losses
    recon = dec(feats_flat)                # (T*B, C, H, W) logits
    image_target = images.float().flatten(0, 1) / 255.0
    image_loss = F.mse_loss(torch.sigmoid(recon), image_target)

    pred_reward = rew_head(feats_flat).view(T, B)
    reward_loss = F.mse_loss(pred_reward, symlog(rewards))

    pred_cont_logits = cont_head(feats_flat).view(T, B)
    cont_loss = F.binary_cross_entropy_with_logits(pred_cont_logits, conts)

    kl = kl_balanced(prior_logits.flatten(0, 1), post_logits.flatten(0, 1),
                     alpha=args.kl_alpha, free=args.kl_free)
    kl_loss = kl.mean()

    wm_loss = (
        image_loss
        + args.reward_loss_weight * reward_loss
        + args.cont_loss_weight * cont_loss
        + kl_loss
    )

    opt_world.zero_grad(set_to_none=True)
    wm_loss.backward()
    nn.utils.clip_grad_norm_(
        list(enc.parameters()) + list(rssm.parameters()) + list(dec.parameters())
        + list(rew_head.parameters()) + list(cont_head.parameters()),
        args.max_grad_norm,
    )
    opt_world.step()

    # ── 4. imagined rollouts for AC training
    H = args.imagine_horizon
    # Detach posterior states so AC gradients don't flow into world model.
    starts = {
        "deter": deter.flatten(0, 1).detach(),
        "stoch": stoch.flatten(0, 1).detach(),
        "logits": post_logits.flatten(0, 1).detach(),
    }
    img_feats = []
    img_actions = []
    img_log_probs = []
    img_entropies = []
    cur = starts
    for t in range(H):
        feat = rssm.feat(cur)
        img_feats.append(feat)
        move_logits, scalar_mean, scalar_std = actor(feat)
        cat = torch.distributions.Categorical(logits=move_logits)
        norm = torch.distributions.Normal(scalar_mean.squeeze(-1), scalar_std.squeeze(-1))
        move = cat.sample()
        scalar = norm.rsample().clamp(-1, 1)
        log_prob = cat.log_prob(move) + norm.log_prob(scalar)
        entropy = cat.entropy() + norm.entropy()
        img_log_probs.append(log_prob)
        img_entropies.append(entropy)
        onehot = F.one_hot(move, 5).float()
        a_vec = torch.cat([onehot, scalar.unsqueeze(-1)], dim=-1)
        img_actions.append(a_vec)
        cur = rssm.img_step(cur, a_vec)
    img_feats.append(rssm.feat(cur))  # one extra for bootstrap value
    img_feats = torch.stack(img_feats)               # (H+1, T*B, feat)
    img_log_probs = torch.stack(img_log_probs)       # (H, T*B)
    img_entropies = torch.stack(img_entropies)
    img_feats_flat = img_feats.flatten(0, 1)
    pred_r = symexp(rew_head(img_feats_flat)).view(H + 1, -1)  # in original reward units
    pred_c = torch.sigmoid(cont_head(img_feats_flat)).view(H + 1, -1)

    with torch.no_grad():
        target_values = critic_target(img_feats_flat).view(H + 1, -1)
    # GAE-λ in latent space
    gamma = args.gamma
    lam = args.gae_lambda
    advs = torch.zeros(H, img_feats.shape[1], device=device)
    last = target_values[H]
    for t in reversed(range(H)):
        delta = pred_r[t] + gamma * pred_c[t] * target_values[t + 1] - target_values[t]
        last = delta + gamma * lam * pred_c[t] * last
        advs[t] = last
    returns = advs + target_values[:H]

    # Actor loss — REINFORCE with baseline + entropy bonus
    pg = -(img_log_probs * advs.detach()).mean()
    ent = -args.actor_ent_coef * img_entropies.mean()
    actor_loss = pg + ent

    opt_actor.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
    opt_actor.step()

    # Critic loss — λ-return regression. Inputs are detached because the
    # imagined-dynamics graph was already consumed by actor_loss.backward();
    # the critic only needs gradients w.r.t. its own parameters.
    critic_in = img_feats_flat.detach()[: H * img_feats.shape[1]]
    critic_pred = critic(critic_in).view(H, -1)
    critic_loss = F.mse_loss(critic_pred, returns.detach())

    opt_critic.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
    opt_critic.step()

    return {
        "wm/loss": wm_loss.item(),
        "wm/image_loss": image_loss.item(),
        "wm/reward_loss": reward_loss.item(),
        "wm/cont_loss": cont_loss.item(),
        "wm/kl_loss": kl_loss.item(),
        "actor/loss": actor_loss.item(),
        "actor/pg": pg.item(),
        "actor/entropy": img_entropies.mean().item(),
        "critic/loss": critic_loss.item(),
        "critic/value_mean": target_values.mean().item(),
        "imag/return_mean": returns.mean().item(),
    }


if __name__ == "__main__":
    main()
