#!/usr/bin/env python3
"""PPO + GRU for the Cogniland navigation env, with W&B logging.

Why not vanilla stable-baselines3?
---------------------------------
The env has a *hybrid* action space:

    {"move": Discrete(5),  "build_scalar": Box(-1, 1, (1,))}

SB3 (and `sb3_contrib.RecurrentPPO`) only accept `Discrete`,
`MultiDiscrete`, `MultiBinary`, or `Box` action spaces — there is no clean
way to drive a categorical move *and* a continuous tanh scalar from the
same policy out of the box. This script implements the exact PPO algorithm
SB3 uses (clipped surrogate objective, GAE, multi-epoch minibatch updates,
value clipping skipped, advantage normalisation per minibatch) on top of a
small custom policy with **both heads**:

  CNN trunk → linear → GRU(128) → ┬─ Categorical over 5 moves
                                  ├─ tanh-squashed Gaussian scalar (μ from
                                  │   tanh(linear), shared learned log_std)
                                  └─ value head

The build_scalar is sampled every step and stored, but the env only
*consumes* it on build actions; the policy learns through policy gradient
to bias positive (raft) on lake maps and negative (harness) on rocky
maps, because the build action's downstream reward depends on the sign.

`obs["skill_active"]` is the only signal the agent gets that it has
already built — exactly what you asked for: the observable flips from 0
to 1 the moment the build action fires.

How to run on an RTX 4090
-------------------------

    pip install wandb                            # one-time
    wandb login                                  # one-time, paste your key

    python scripts/train_ppo_gru.py \\
        --total-timesteps 5_000_000 \\
        --num-envs 32 --num-steps 128 \\
        --env-size 64 --view-size 11 --tile-px 8 \\
        --device cuda --wandb-project cogniland-nav

A 4090 can comfortably push num_envs=32, num_steps=128 with tile_px=8
(observation tensor is ``[32, 3, 88, 88]``). Bigger view / tile_px works
too but at the cost of fps. For a quick smoke run use
``--total-timesteps 100000 --num-envs 4 --wandb-mode disabled``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# macOS conda env safety; harmless on Linux.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cogniland.nav import CognilandNavEnv  # noqa: E402
from cogniland.nav.tiles import NUM_TILES  # noqa: E402


# =============================================================== vec env

class VecCognilandEnv:
    """Synchronous vector env around N CognilandNavEnv instances.

    Tracks per-env episode return/length and reports them via the ``info``
    dict on the step where ``done=True`` (same pattern as
    ``RecordEpisodeStatistics``).
    """

    def __init__(self, num_envs: int, **env_kwargs):
        base_seed = env_kwargs.pop("seed", 0)
        self.envs = [
            CognilandNavEnv(seed=base_seed + i, **env_kwargs) for i in range(num_envs)
        ]
        self.num_envs = num_envs
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.ep_returns = np.zeros(num_envs, dtype=np.float32)
        self.ep_lengths = np.zeros(num_envs, dtype=np.int32)

    def reset(self):
        obses = [e.reset()[0] for e in self.envs]
        self.ep_returns[:] = 0.0
        self.ep_lengths[:] = 0
        return self._stack(obses)

    def step(self, moves: np.ndarray, scalars: np.ndarray):
        next_obs, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            action = {
                "move": int(moves[i]),
                "build_scalar": np.array([float(scalars[i])], np.float32),
            }
            o, r, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            self.ep_returns[i] += r
            self.ep_lengths[i] += 1
            if done:
                info["episode"] = {
                    "return": float(self.ep_returns[i]),
                    "length": int(self.ep_lengths[i]),
                    "map_type": info["map_type"],
                    "correct_object": info["correct_object"],
                    "active_object": info["active_object"],
                    "reached_target": bool(info["reached_target"]),
                }
                self.ep_returns[i] = 0.0
                self.ep_lengths[i] = 0
                o, _ = env.reset()
            next_obs.append(o)
            rewards.append(r)
            dones.append(done)
            infos.append(info)
        return (
            self._stack(next_obs),
            np.asarray(rewards, np.float32),
            np.asarray(dones, np.bool_),
            infos,
        )

    def _stack(self, obses):
        out: dict[str, np.ndarray] = {
            "skill_active": np.stack([o["skill_active"] for o in obses]),
        }
        if "semantic" in obses[0]:
            out["semantic"] = np.stack([o["semantic"] for o in obses])
        if "image" in obses[0]:
            out["image"] = np.stack([o["image"] for o in obses])
        return out


# =============================================================== policy

def _layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOGRUPolicy(nn.Module):
    """Tile-embed → CNN → MLP → GRU → (Categorical move, tanh-Gaussian scalar, value).

    Inputs come as a Dict with either ``semantic`` (int8 [view, view] tile
    ids, default) or ``image`` (uint8 [3, H, W] RGB). The symbolic path
    embeds each tile id to ``tile_embed_dim`` and runs a small 2D CNN over
    the embedded grid — far cheaper than a pixel CNN and learns relational
    features over the tile grid directly.
    """

    def __init__(self, obs_space, num_move_actions: int = 5,
                 gru_hidden: int = 128, embed_dim: int = 256,
                 tile_embed_dim: int = 16, num_tile_classes: int = NUM_TILES):
        super().__init__()
        self.has_semantic = "semantic" in obs_space.spaces
        self.has_image = "image" in obs_space.spaces

        if self.has_semantic:
            V, _ = obs_space["semantic"].shape
            self.view = V
            self.tile_embed = nn.Embedding(num_tile_classes, tile_embed_dim)
            nn.init.normal_(self.tile_embed.weight, std=0.5)
            # CoordConv-lite: 2 extra channels carrying normalised row/col in [-1, 1]
            in_c = tile_embed_dim + 2
            self.cnn = nn.Sequential(
                _layer_init(nn.Conv2d(in_c, 32, kernel_size=3, padding=0)), nn.ReLU(),
                _layer_init(nn.Conv2d(32, 32, kernel_size=3, padding=0)), nn.ReLU(),
                _layer_init(nn.Conv2d(32, 32, kernel_size=3, padding=0)), nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                n_flat = self.cnn(torch.zeros(1, in_c, V, V)).shape[1]
        else:
            C, H, W = obs_space["image"].shape
            self.cnn = nn.Sequential(
                _layer_init(nn.Conv2d(C, 32, kernel_size=8, stride=4)), nn.ReLU(),
                _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
                _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                n_flat = self.cnn(torch.zeros(1, C, H, W)).shape[1]

        self.embed = nn.Sequential(
            _layer_init(nn.Linear(n_flat + 1, embed_dim)),
            nn.ReLU(),
        )
        self.gru = nn.GRU(embed_dim, gru_hidden, batch_first=False)
        for name, p in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p, 1.0)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)

        self.actor = _layer_init(nn.Linear(gru_hidden, num_move_actions), std=0.01)
        self.scalar_mean = _layer_init(nn.Linear(gru_hidden, 1), std=0.01)
        self.scalar_log_std = nn.Parameter(torch.zeros(1) - 0.5)
        self.critic = _layer_init(nn.Linear(gru_hidden, 1), std=1.0)
        self.gru_hidden = gru_hidden

    def _encode(self, obs):
        if self.has_semantic:
            sem = obs["semantic"].long()  # (B, V, V)
            B, V, _ = sem.shape
            emb = self.tile_embed(sem)  # (B, V, V, E)
            # CoordConv channels
            rr = torch.linspace(-1, 1, V, device=sem.device).view(1, V, 1).expand(B, V, V)
            cc = torch.linspace(-1, 1, V, device=sem.device).view(1, 1, V).expand(B, V, V)
            coords = torch.stack([rr, cc], dim=-1)  # (B, V, V, 2)
            x = torch.cat([emb, coords], dim=-1).permute(0, 3, 1, 2)  # (B, E+2, V, V)
            feat = self.cnn(x)
        else:
            img = obs["image"].float() / 255.0
            feat = self.cnn(img)
        feat = torch.cat([feat, obs["skill_active"].float()], dim=-1)
        return self.embed(feat)

    def _gru_forward(self, obs_seq, done_seq, hidden):
        """Sequential GRU over T steps (resets hidden when done_seq[t]==1)."""
        # Pick any obs key to read T, B from.
        any_key = next(iter(obs_seq))
        T, B = obs_seq[any_key].shape[:2]
        flat = {k: v.flatten(0, 1) for k, v in obs_seq.items()}
        feat_flat = self._encode(flat)  # (T*B, embed)
        feat = feat_flat.reshape(T, B, -1)
        h = hidden
        outs = []
        for t in range(T):
            mask = (1.0 - done_seq[t].float()).view(1, B, 1)
            h = h * mask
            y, h = self.gru(feat[t:t + 1], h)
            outs.append(y)
        return torch.cat(outs, dim=0), h  # (T, B, hidden), (1, B, hidden)

    def _heads(self, x):
        logits = self.actor(x)
        scalar_mean = torch.tanh(self.scalar_mean(x))
        scalar_std = self.scalar_log_std.exp().expand_as(scalar_mean)
        value = self.critic(x).squeeze(-1)
        return logits, scalar_mean, scalar_std, value

    # ---- 1-step path (rollout collection) ------------------------------

    def get_action_and_value(self, obs, hidden, done, action=None, scalar=None):
        # add fake time dim of 1
        obs_seq = {k: v.unsqueeze(0) for k, v in obs.items()}
        gru_out, h_new = self._gru_forward(obs_seq, done.unsqueeze(0), hidden)
        x = gru_out.squeeze(0)
        logits, mean, std, value = self._heads(x)
        cat = Categorical(logits=logits)
        norm = Normal(mean, std)
        if action is None:
            action = cat.sample()
        if scalar is None:
            scalar = norm.sample()
        log_prob = cat.log_prob(action) + norm.log_prob(scalar).squeeze(-1)
        entropy = cat.entropy() + norm.entropy().squeeze(-1)
        return action, scalar, log_prob, entropy, value, h_new

    # ---- T-step path (PPO update) --------------------------------------

    def evaluate(self, obs_seq, done_seq, hidden, actions, scalars):
        gru_out, _ = self._gru_forward(obs_seq, done_seq, hidden)
        T, B = gru_out.shape[:2]
        x = gru_out.reshape(T * B, -1)
        logits, mean, std, value = self._heads(x)
        cat = Categorical(logits=logits)
        norm = Normal(mean, std)
        actions_flat = actions.reshape(T * B)
        scalars_flat = scalars.reshape(T * B, 1)
        lp_a = cat.log_prob(actions_flat).reshape(T, B)
        lp_s = norm.log_prob(scalars_flat).squeeze(-1).reshape(T, B)
        ent = (cat.entropy() + norm.entropy().squeeze(-1)).reshape(T, B)
        return lp_a + lp_s, ent, value.reshape(T, B)


# =============================================================== training

def _to_device(obs: dict, device):
    return {k: torch.from_numpy(v).to(device) for k, v in obs.items()}


def main():
    parser = argparse.ArgumentParser()
    # env / data
    parser.add_argument("--env-size", type=int, default=64, choices=(32, 64, 96, 128))
    parser.add_argument("--map-type", default="random",
                        choices=("random", "lake", "rocky", "balanced"))
    parser.add_argument("--view-size", type=int, default=21)
    parser.add_argument("--tile-px", type=int, default=8,
                        help="render resolution per tile (only used in rgb mode)")
    parser.add_argument("--obs-mode", default="symbolic",
                        choices=("symbolic", "rgb", "both"))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=128,
                        help="rollout length per env before each PPO update")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    # PPO
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="minibatches over the env dimension — must divide num_envs")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    # network
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    # infra
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="cogniland-nav")
    parser.add_argument("--wandb-mode", default="online",
                        choices=("online", "offline", "disabled"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-every-iters", type=int, default=50)
    args = parser.parse_args()

    assert args.num_envs % args.num_minibatches == 0, \
        "num_envs must be divisible by num_minibatches (minibatching is over envs)"

    run_name = args.run_name or (
        f"ppo_gru_size{args.env_size}_seed{args.seed}_{int(time.time())}"
    )
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        mode=args.wandb_mode,
        save_code=True,
    )
    device = torch.device(args.device)
    print(f"device={device}  run_name={run_name}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -------------------------------------- envs + policy + optimizer
    vec = VecCognilandEnv(
        args.num_envs,
        size=args.env_size,
        map_type=args.map_type,
        view_size=args.view_size,
        tile_px=args.tile_px,
        obs_mode=args.obs_mode,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    policy = PPOGRUPolicy(
        vec.single_observation_space,
        gru_hidden=args.gru_hidden,
        embed_dim=args.embed_dim,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy params: {n_params:,}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------- rollout buffers (on device)
    obs_buf: dict[str, torch.Tensor] = {
        "skill_active": torch.zeros(
            (args.num_steps, args.num_envs, 1), dtype=torch.float32, device=device
        ),
    }
    if "semantic" in vec.single_observation_space.spaces:
        sem_shape = vec.single_observation_space["semantic"].shape
        obs_buf["semantic"] = torch.zeros(
            (args.num_steps, args.num_envs) + sem_shape, dtype=torch.int8, device=device
        )
    if "image" in vec.single_observation_space.spaces:
        img_shape = vec.single_observation_space["image"].shape
        obs_buf["image"] = torch.zeros(
            (args.num_steps, args.num_envs) + img_shape, dtype=torch.uint8, device=device
        )
    actions_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    scalars_buf = torch.zeros((args.num_steps, args.num_envs, 1), dtype=torch.float32, device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    next_obs = vec.reset()
    next_obs_t = _to_device(next_obs, device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    next_hidden = torch.zeros(1, args.num_envs, args.gru_hidden, device=device)

    batch_size = args.num_envs * args.num_steps
    envs_per_minibatch = args.num_envs // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    global_step = 0
    start_time = time.time()

    print(
        f"num_iterations={num_iterations}  batch_size={batch_size}  "
        f"envs_per_minibatch={envs_per_minibatch}"
    )

    # ============================================================ TRAIN
    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            for pg in optimizer.param_groups:
                pg["lr"] = frac * args.learning_rate

        initial_hidden = next_hidden.clone()

        # -------- collect rollout ----------------------------------------
        ep_returns_recent: list[float] = []
        ep_lengths_recent: list[int] = []
        ep_reached: list[float] = []
        match_obj: dict[str, list[float]] = {"raft_built_on_lake": [], "harness_built_on_rocky": []}
        wrong_obj: list[float] = []
        none_obj: list[float] = []

        for step in range(args.num_steps):
            global_step += args.num_envs
            for k in obs_buf:
                obs_buf[k][step] = next_obs_t[k]
            dones_buf[step] = next_done

            with torch.no_grad():
                action, scalar, log_prob, _, value, next_hidden = policy.get_action_and_value(
                    next_obs_t, next_hidden, next_done
                )
            actions_buf[step] = action
            scalars_buf[step] = scalar  # store pre-clip sample for log_prob consistency
            logprobs_buf[step] = log_prob
            values_buf[step] = value

            # env consumes scalar in [-1, 1]
            np_moves = action.cpu().numpy()
            np_scalars = torch.clamp(scalar, -1.0, 1.0).squeeze(-1).cpu().numpy()
            next_obs, reward, done, infos = vec.step(np_moves, np_scalars)

            rewards_buf[step] = torch.from_numpy(reward).to(device)
            next_obs_t = _to_device(next_obs, device)
            next_done = torch.from_numpy(done.astype(np.float32)).to(device)

            for info in infos:
                if "episode" not in info:
                    continue
                ep = info["episode"]
                ep_returns_recent.append(ep["return"])
                ep_lengths_recent.append(ep["length"])
                ep_reached.append(float(ep["reached_target"]))
                correct = ep["correct_object"]
                active = ep["active_object"]
                if active == "none":
                    none_obj.append(1.0)
                elif active == correct:
                    if ep["map_type"] == "lake":
                        match_obj["raft_built_on_lake"].append(1.0)
                    else:
                        match_obj["harness_built_on_rocky"].append(1.0)
                else:
                    wrong_obj.append(1.0)

        # -------- bootstrap + GAE ---------------------------------------
        with torch.no_grad():
            _, _, _, _, next_value, _ = policy.get_action_and_value(
                next_obs_t, next_hidden, next_done
            )
            advantages = torch.zeros_like(rewards_buf)
            last_gae = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_v = next_value
                    next_nonterm = 1.0 - next_done
                else:
                    next_v = values_buf[t + 1]
                    next_nonterm = 1.0 - dones_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * next_v * next_nonterm - values_buf[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_nonterm * last_gae
                advantages[t] = last_gae
            returns = advantages + values_buf

        # -------- PPO update (env-minibatched) --------------------------
        env_idx = np.arange(args.num_envs)
        pg_losses, v_losses, ent_losses, kls, clipfracs = [], [], [], [], []

        early_stop = False
        for epoch in range(args.update_epochs):
            np.random.shuffle(env_idx)
            for start in range(0, args.num_envs, envs_per_minibatch):
                mb = env_idx[start : start + envs_per_minibatch]
                mb_t = torch.from_numpy(mb).to(device)

                mb_obs = {k: v[:, mb_t] for k, v in obs_buf.items()}
                mb_dones = dones_buf[:, mb_t]
                mb_actions = actions_buf[:, mb_t]
                mb_scalars = scalars_buf[:, mb_t]
                mb_old_logp = logprobs_buf[:, mb_t]
                mb_adv = advantages[:, mb_t]
                mb_ret = returns[:, mb_t]
                mb_h0 = initial_hidden[:, mb_t]

                new_logp, ent, new_value = policy.evaluate(
                    mb_obs, mb_dones, mb_h0, mb_actions, mb_scalars
                )
                log_ratio = new_logp - mb_old_logp
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

                adv = mb_adv
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * (new_value - mb_ret).pow(2).mean()
                ent_loss = ent.mean()

                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
                kls.append(approx_kl.item())
                clipfracs.append(clipfrac.item())

            if args.target_kl is not None and np.mean(kls[-args.num_minibatches :]) > args.target_kl:
                early_stop = True
                break

        # -------- log ----------------------------------------------------
        sps = global_step / (time.time() - start_time)
        log_payload = {
            "train/policy_loss": float(np.mean(pg_losses)),
            "train/value_loss": float(np.mean(v_losses)),
            "train/entropy": float(np.mean(ent_losses)),
            "train/approx_kl": float(np.mean(kls)),
            "train/clipfrac": float(np.mean(clipfracs)),
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/scalar_std": float(policy.scalar_log_std.exp().item()),
            "train/iteration": iteration,
            "train/sps": sps,
            "train/early_stop": int(early_stop),
        }
        if ep_returns_recent:
            log_payload.update({
                "charts/episode_return_mean": float(np.mean(ep_returns_recent)),
                "charts/episode_length_mean": float(np.mean(ep_lengths_recent)),
                "charts/reach_rate": float(np.mean(ep_reached)),
                "charts/built_none_frac": (
                    float(np.mean(none_obj)) if none_obj else 0.0
                ),
                "charts/built_wrong_frac": (
                    float(np.mean(wrong_obj)) if wrong_obj else 0.0
                ),
                "charts/built_correct_frac": (
                    (sum(len(v) for v in match_obj.values()) /
                     max(1, len(ep_returns_recent)))
                ),
            })
        wandb.log(log_payload, step=global_step)

        if iteration % 5 == 0 or iteration == 1:
            er = log_payload.get("charts/episode_return_mean", float("nan"))
            print(
                f"iter={iteration:4d}/{num_iterations}  step={global_step:>9d}  sps={sps:.0f}  "
                f"ep_return={er:+.2f}  policy_loss={log_payload['train/policy_loss']:+.3f}  "
                f"value_loss={log_payload['train/value_loss']:.3f}  "
                f"kl={log_payload['train/approx_kl']:.4f}"
            )

        if iteration % args.save_every_iters == 0:
            ckpt = args.checkpoint_dir / f"{run_name}_iter{iteration}.pt"
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                    "global_step": global_step,
                    "args": vars(args),
                },
                ckpt,
            )
            wandb.save(str(ckpt))
            print(f"saved {ckpt}")

    final_ckpt = args.checkpoint_dir / f"{run_name}_final.pt"
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": num_iterations,
            "global_step": global_step,
            "args": vars(args),
        },
        final_ckpt,
    )
    wandb.save(str(final_ckpt))
    print(f"saved final {final_ckpt}")
    wandb.finish()


if __name__ == "__main__":
    main()
