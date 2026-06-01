#!/usr/bin/env python3
"""PPO + GRU for the bridge_tunnel_commit POMDP, with W&B logging.

The env (``cogniland.bridge_tunnel_commit.BridgeTunnelCommitEnv``) is the
natural-terrain navigation task (midL spawn → central goal door on the right
wall, water/rock obstacles, edge-biased tree forests) plus a **one-shot
build/mine commitment**: the agent must use ``COMMIT_BUILD`` / ``COMMIT_MINE`` to
irreversibly unlock exactly one crossing tool before it can build over water /
mine through rock. Maps come in three categories (balanced / lakes / rocky)
drawn uniformly each reset, so the policy must read the terrain and commit to the
tool the map actually needs. POMDP via an egocentric crop.

Action space is ``Discrete(8)`` (up/down/left/right + BUILD + MINE + COMMIT_BUILD
+ COMMIT_MINE). The policy is a tile-embed (or one-hot) CNN over the minimap
crop, concatenated with the scalar vector (facing one-hot + step fraction +
commit-build/commit-mine flags), fed through a GRU(128) into a Categorical actor
and a value head.

How to run on an RTX 4090
-------------------------

    python scripts/train_ppo_bridge_tunnel_commit.py \\
        --total-timesteps 8_000_000 --num-envs 32 --num-steps 128 \\
        --device cuda --wandb-mode online

Quick smoke:

    python scripts/train_ppo_bridge_tunnel_commit.py \\
        --total-timesteps 50000 --num-envs 8 --wandb-mode disabled
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cogniland.bridge_tunnel_commit import BridgeTunnelCommitEnv  # noqa: E402
from cogniland.bridge_tunnel_commit.tiles import NUM_TILES  # noqa: E402


# =============================================================== vec env

class VecBridgeTunnelCommitEnv:
    """Synchronous vector env around N BridgeTunnelCommitEnv instances, auto-resetting
    and reporting per-episode stats on the done step."""

    def __init__(self, num_envs: int, **env_kwargs):
        base_seed = env_kwargs.pop("seed", 0)
        self.envs = [BridgeTunnelCommitEnv(seed=base_seed + i, **env_kwargs) for i in range(num_envs)]
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

    def step(self, moves: np.ndarray):
        next_obs, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(int(moves[i]))
            done = bool(term or trunc)
            self.ep_returns[i] += r
            self.ep_lengths[i] += 1
            if done:
                info["episode"] = {
                    "return": float(self.ep_returns[i]),
                    "length": int(self.ep_lengths[i]),
                    "reached_target": bool(info["reached_target"]),
                    "commit": int(info.get("commit", 0)),       # 0 none/1 build/2 mine
                    "category": str(info.get("category", "balanced")),
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
        return {
            "minimap": np.stack([o["minimap"] for o in obses]),
            "scalars": np.stack([o["scalars"] for o in obses]),
        }


# =============================================================== policy

def _layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOGRUPolicy(nn.Module):
    """Tile-embed minimap → CNN → MLP(+scalars) → GRU → (Categorical, value)."""

    def __init__(self, obs_space, num_actions: int = 6, gru_hidden: int = 128,
                 embed_dim: int = 256, tile_embed_dim: int = 16,
                 num_tile_classes: int = NUM_TILES, obs_encoding: str = "embed"):
        super().__init__()
        V, _ = obs_space["minimap"].shape
        n_scalars = obs_space["scalars"].shape[0]
        self.view = V
        self.obs_encoding = obs_encoding
        self.num_tile_classes = num_tile_classes
        if obs_encoding == "onehot":
            # categorical observation: one-hot the tile ids (matches the
            # DreamerV3 categorical encoder for a fair comparison). No learned
            # tile embedding — the CNN sees raw per-tile indicator channels.
            self.tile_embed = None
            in_c = num_tile_classes + 2                  # + CoordConv row/col
        else:
            self.tile_embed = nn.Embedding(num_tile_classes, tile_embed_dim)
            nn.init.normal_(self.tile_embed.weight, std=0.5)
            in_c = tile_embed_dim + 2                     # + CoordConv row/col
        self.cnn = nn.Sequential(
            _layer_init(nn.Conv2d(in_c, 32, kernel_size=3, padding=0)), nn.ReLU(),
            _layer_init(nn.Conv2d(32, 32, kernel_size=3, padding=0)), nn.ReLU(),
            _layer_init(nn.Conv2d(32, 32, kernel_size=3, padding=0)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, in_c, V, V)).shape[1]

        self.embed = nn.Sequential(
            _layer_init(nn.Linear(n_flat + n_scalars, embed_dim)), nn.ReLU(),
        )
        self.gru = nn.GRU(embed_dim, gru_hidden, batch_first=False)
        for name, p in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p, 1.0)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)
        self.actor = _layer_init(nn.Linear(gru_hidden, num_actions), std=0.01)
        self.critic = _layer_init(nn.Linear(gru_hidden, 1), std=1.0)
        self.gru_hidden = gru_hidden

    def _encode(self, obs):
        mm = obs["minimap"].long()                      # (B, V, V)
        B, V, _ = mm.shape
        if self.obs_encoding == "onehot":
            emb = torch.nn.functional.one_hot(mm, self.num_tile_classes).float()  # (B,V,V,K)
        else:
            emb = self.tile_embed(mm)                   # (B, V, V, E)
        rr = torch.linspace(-1, 1, V, device=mm.device).view(1, V, 1).expand(B, V, V)
        cc = torch.linspace(-1, 1, V, device=mm.device).view(1, 1, V).expand(B, V, V)
        coords = torch.stack([rr, cc], dim=-1)          # (B, V, V, 2)
        x = torch.cat([emb, coords], dim=-1).permute(0, 3, 1, 2)
        feat = self.cnn(x)
        feat = torch.cat([feat, obs["scalars"].float()], dim=-1)
        return self.embed(feat)

    def _gru_forward(self, obs_seq, done_seq, hidden):
        any_key = next(iter(obs_seq))
        T, B = obs_seq[any_key].shape[:2]
        flat = {k: v.flatten(0, 1) for k, v in obs_seq.items()}
        feat = self._encode(flat).reshape(T, B, -1)
        h = hidden
        outs = []
        for t in range(T):
            mask = (1.0 - done_seq[t].float()).view(1, B, 1)
            h = h * mask
            y, h = self.gru(feat[t:t + 1], h)
            outs.append(y)
        return torch.cat(outs, dim=0), h

    def _heads(self, x):
        return self.actor(x), self.critic(x).squeeze(-1)

    def get_action_and_value(self, obs, hidden, done, action=None):
        obs_seq = {k: v.unsqueeze(0) for k, v in obs.items()}
        gru_out, h_new = self._gru_forward(obs_seq, done.unsqueeze(0), hidden)
        x = gru_out.squeeze(0)
        logits, value = self._heads(x)
        cat = Categorical(logits=logits)
        if action is None:
            action = cat.sample()
        return action, cat.log_prob(action), cat.entropy(), value, h_new

    def evaluate(self, obs_seq, done_seq, hidden, actions):
        gru_out, _ = self._gru_forward(obs_seq, done_seq, hidden)
        T, B = gru_out.shape[:2]
        x = gru_out.reshape(T * B, -1)
        logits, value = self._heads(x)
        cat = Categorical(logits=logits)
        a = actions.reshape(T * B)
        return (cat.log_prob(a).reshape(T, B), cat.entropy().reshape(T, B),
                value.reshape(T, B))


# =============================================================== training

def _to_device(obs: dict, device):
    return {k: torch.from_numpy(v).to(device) for k, v in obs.items()}


def main():
    parser = argparse.ArgumentParser()
    # env / data
    parser.add_argument("--env-size", type=int, default=32, help="map height")
    parser.add_argument("--env-width", type=int, default=None,
                        help="map width (default = env-size, i.e. square)")
    parser.add_argument("--view-size", type=int, default=11)
    parser.add_argument("--orientation", default="natural",
                        choices=("natural",),
                        help="layout: natural (open lakes/mountains/trees, "
                             "midL→right wall). Stripe orientations are retired.")
    parser.add_argument("--categories", nargs="+",
                        default=["balanced", "lakes", "rocky"],
                        choices=("balanced", "lakes", "rocky"),
                        help="map categories drawn uniformly each reset "
                             "(balanced 14/14, lakes 80/20 water:rock, rocky 20/80)")
    parser.add_argument("--tree-frac", type=float, default=0.03,
                        help="natural: impassable tree coverage")
    parser.add_argument("--goal-half", type=int, default=1,
                        help="central goal-door half-height (default 1 = 3-cell door); "
                             "<0 ⇒ whole right wall")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="generous episode timeout — success measures whether "
                             "the agent reaches the target at all, not its speed")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000,
                        help="bridge_tunnel maps converge well before 2M")
    # reward
    parser.add_argument("--slack-penalty", type=float, default=-0.01)
    parser.add_argument("--shaping-coef", type=float, default=0.01)
    parser.add_argument("--reach-bonus", type=float, default=1.0)
    parser.add_argument("--build-cost", type=float, default=0.05,
                        help="extra penalty per successful PLACE/MINE — makes "
                             "crossing an obstacle cost more than walking around it")
    # PPO
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--anneal-ent", action="store_true",
                        help="linearly anneal ent-coef to ~0 over training — explore "
                             "early, sharpen late so the greedy (argmax) policy is crisp")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    # network
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    # infra
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="bridge_tunnel_commit")
    parser.add_argument("--wandb-mode", default="online",
                        choices=("online", "offline", "disabled"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-every-iters", type=int, default=300)
    parser.add_argument("--obs-encoding", choices=("embed", "onehot"), default="embed",
                        help="onehot = categorical one-hot minimap (matches the DreamerV3 "
                             "categorical encoder for a fair comparison); embed = learned tile embedding")
    parser.add_argument("--config", type=Path, default=None)
    args, _ = parser.parse_known_args()
    if args.config is not None:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        unknown = set(cfg) - {a.dest for a in parser._actions}
        if unknown:
            raise SystemExit(f"--config {args.config}: unknown keys {sorted(unknown)}")
        parser.set_defaults(**cfg)
    args = parser.parse_args()

    assert args.num_envs % args.num_minibatches == 0

    run_name = args.run_name or f"ppo_bridge_tunnel_commit_size{args.env_size}_seed{args.seed}_{int(time.time())}"
    wandb.init(
        project=args.wandb_project, name=run_name, config=vars(args),
        mode=args.wandb_mode, save_code=True,
        tags=["algo=ppo_gru", f"map={args.env_size}", "env=bridge_tunnel_commit",
              f"orientation={args.orientation}"],
    )
    device = torch.device(args.device)
    print(f"device={device}  run_name={run_name}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vec = VecBridgeTunnelCommitEnv(
        args.num_envs, size=args.env_size, width=args.env_width,
        view_size=args.view_size, orientation=args.orientation,
        max_steps=args.max_steps, slack_penalty=args.slack_penalty,
        shaping_coef=args.shaping_coef, reach_bonus=args.reach_bonus,
        build_cost=args.build_cost, gamma=args.gamma, seed=args.seed,
        categories=tuple(args.categories), tree_frac=args.tree_frac,
        goal_half=(args.goal_half if args.goal_half >= 0 else None),
    )
    policy = PPOGRUPolicy(
        vec.single_observation_space, num_actions=vec.single_action_space.n,
        gru_hidden=args.gru_hidden, embed_dim=args.embed_dim,
        obs_encoding=args.obs_encoding,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy params: {n_params:,}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)
    if wandb.run is not None:
        wandb.run.tags = list(wandb.run.tags or []) + [f"size={n_params / 1e6:.1f}M"]

    args.checkpoint_dir = args.checkpoint_dir / run_name
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    V = args.view_size
    n_scalars = vec.single_observation_space["scalars"].shape[0]
    obs_buf = {
        "minimap": torch.zeros((args.num_steps, args.num_envs, V, V), dtype=torch.int8, device=device),
        "scalars": torch.zeros((args.num_steps, args.num_envs, n_scalars), dtype=torch.float32, device=device),
    }
    actions_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    next_obs_t = _to_device(vec.reset(), device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    next_hidden = torch.zeros(1, args.num_envs, args.gru_hidden, device=device)

    batch_size = args.num_envs * args.num_steps
    envs_per_minibatch = args.num_envs // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    global_step = 0
    start_time = time.time()
    print(f"num_iterations={num_iterations}  batch_size={batch_size}")

    for iteration in range(1, num_iterations + 1):
        frac = 1.0 - (iteration - 1.0) / num_iterations
        if args.anneal_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = frac * args.learning_rate
        # entropy: full early (explore), → ~0 late (crisp greedy policy)
        ent_coef = args.ent_coef * frac if args.anneal_ent else args.ent_coef

        initial_hidden = next_hidden.clone()
        ep_returns, ep_lengths, ep_reached = [], [], []
        ep_commit, ep_cat_reached = [], {}        # commit choice + per-category success

        for step in range(args.num_steps):
            global_step += args.num_envs
            for k in obs_buf:
                obs_buf[k][step] = next_obs_t[k]
            dones_buf[step] = next_done
            with torch.no_grad():
                action, log_prob, _, value, next_hidden = policy.get_action_and_value(
                    next_obs_t, next_hidden, next_done)
            actions_buf[step] = action
            logprobs_buf[step] = log_prob
            values_buf[step] = value
            next_obs, reward, done, infos = vec.step(action.cpu().numpy())
            rewards_buf[step] = torch.from_numpy(reward).to(device)
            next_obs_t = _to_device(next_obs, device)
            next_done = torch.from_numpy(done.astype(np.float32)).to(device)
            for info in infos:
                if "episode" in info:
                    ep = info["episode"]
                    ep_returns.append(ep["return"])
                    ep_lengths.append(ep["length"])
                    ep_reached.append(float(ep["reached_target"]))
                    ep_commit.append(ep["commit"])
                    ep_cat_reached.setdefault(ep["category"], []).append(
                        float(ep["reached_target"]))

        with torch.no_grad():
            _, _, _, next_value, _ = policy.get_action_and_value(next_obs_t, next_hidden, next_done)
            advantages = torch.zeros_like(rewards_buf)
            last_gae = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_v, next_nonterm = next_value, 1.0 - next_done
                else:
                    next_v, next_nonterm = values_buf[t + 1], 1.0 - dones_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * next_v * next_nonterm - values_buf[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_nonterm * last_gae
                advantages[t] = last_gae
            returns = advantages + values_buf

        env_idx = np.arange(args.num_envs)
        pg_losses, v_losses, ent_losses, kls, clipfracs = [], [], [], [], []
        early_stop = False
        for epoch in range(args.update_epochs):
            np.random.shuffle(env_idx)
            for start in range(0, args.num_envs, envs_per_minibatch):
                mb = torch.from_numpy(env_idx[start:start + envs_per_minibatch]).to(device)
                mb_obs = {k: v[:, mb] for k, v in obs_buf.items()}
                new_logp, ent, new_value = policy.evaluate(
                    mb_obs, dones_buf[:, mb], initial_hidden[:, mb], actions_buf[:, mb])
                log_ratio = new_logp - logprobs_buf[:, mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                adv = advantages[:, mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio,
                                    -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
                v_loss = 0.5 * (new_value - returns[:, mb]).pow(2).mean()
                ent_loss = ent.mean()
                loss = pg_loss + args.vf_coef * v_loss - ent_coef * ent_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                pg_losses.append(pg_loss.item()); v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item()); kls.append(approx_kl.item())
            if args.target_kl is not None and np.mean(kls[-args.num_minibatches:]) > args.target_kl:
                early_stop = True
                break

        sps = global_step / (time.time() - start_time)
        log = {
            "loss/policy": float(np.mean(pg_losses)),
            "loss/value": float(np.mean(v_losses)),
            "loss/entropy": float(np.mean(ent_losses)),
            "train/approx_kl": float(np.mean(kls)),
            "train/clipfrac": float(np.mean(clipfracs)),
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/iteration": iteration,
            "perf/fps": sps,
            "train/early_stop": int(early_stop),
        }
        if ep_returns:
            min_steps = args.env_size + (args.env_width or args.env_size)
            log.update({
                "return/mean": float(np.mean(ep_returns)),
                "return/rolling100": float(np.mean(ep_returns[-100:])),
                "success/mean": float(np.mean(ep_reached)),
                "success/rolling100": float(np.mean(ep_reached[-100:])),
                "rollout/episode_length": float(np.mean(ep_lengths)),
                "return/min_over_steps": float(np.mean([min_steps / max(L, 1) for L in ep_lengths])),
            })
            if ep_commit:
                ec = np.asarray(ep_commit)
                log["commit/frac_build"] = float((ec == 1).mean())
                log["commit/frac_mine"] = float((ec == 2).mean())
                log["commit/frac_none"] = float((ec == 0).mean())
            for cat, vals in ep_cat_reached.items():
                log[f"success/{cat}"] = float(np.mean(vals))
        wandb.log(log, step=global_step)

        if iteration % 5 == 0 or iteration == 1:
            print(f"iter={iteration:4d}/{num_iterations} step={global_step:>9d} sps={sps:.0f} "
                  f"ret={log.get('return/mean', float('nan')):+.2f} "
                  f"succ={log.get('success/mean', float('nan')):.2f} "
                  f"build={log.get('commit/frac_build', float('nan')):.2f} "
                  f"mine={log.get('commit/frac_mine', float('nan')):.2f} "
                  f"len={log.get('rollout/episode_length', float('nan')):.0f} "
                  f"kl={log['train/approx_kl']:.4f}")

        if iteration % args.save_every_iters == 0:
            ckpt = args.checkpoint_dir / f"iter{iteration}.pt"
            torch.save({"policy": policy.state_dict(), "iteration": iteration,
                        "global_step": global_step, "args": vars(args)}, ckpt)
            print(f"saved {ckpt}")

    final = args.checkpoint_dir / "final.pt"
    torch.save({"policy": policy.state_dict(), "iteration": num_iterations,
                "global_step": global_step, "args": vars(args)}, final)
    print(f"saved final {final}")
    wandb.finish()


if __name__ == "__main__":
    main()
