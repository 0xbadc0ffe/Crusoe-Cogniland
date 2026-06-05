#!/usr/bin/env python3
"""PPO + GRU for the Cogniland navigation env, with W&B logging.

Action space + policy heads
---------------------------
The env action space is ``Discrete(6)``: up/down/left/right plus two
terminal build actions, ``build_raft`` and ``build_harness``. The build
object is chosen by the action itself (not a continuous scalar). This
script implements the exact PPO algorithm SB3 uses (clipped surrogate
objective, GAE, multi-epoch minibatch updates, value clipping skipped,
advantage normalisation per minibatch) on a small custom GRU policy:

  CNN trunk → linear → GRU(128) → ┬─ Categorical over 6 moves (actor)
                                  ├─ belief head: tanh scalar in [-1, 1]
                                  └─ value head

The **belief** head is an auxiliary map-recognition probe: a tanh scalar
read off the GRU hidden state and supervised by an MSE loss against the
privileged map_type label (+1 lake, -1 rocky, 0 balanced). It is *not* an
action and is never sent to the env — it only trains the recurrent state
to encode which map the agent is on. The agent still has to commit to the
right build via the categorical move head, learning it through the slip
reward downstream.

`obs["skill_active"]` is the only signal the agent gets that it has
already built — the observable flips from 0 to 1 the moment a build action
fires.

How to run on an RTX 4090
-------------------------

    pip install wandb                            # one-time
    wandb login                                  # one-time, paste your key

    python scripts/crafter/train_ppo_gru.py \\
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
from torch.distributions import Categorical

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
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


class GRUMagnitudePruner:
    """One-shot **magnitude pruning** of the GRU weight matrices.

    This is the PyTorch equivalent of jaxpruner's ``MagnitudePruning``. We
    prune only the two recurrent weight matrices ``weight_ih_l0`` (input→gate)
    and ``weight_hh_l0`` (state→gate); biases are left dense.

    *How it works.* At ``prune()`` we rank each weight by ``|w|`` and build a
    binary mask that keeps the top ``(1 - sparsity)`` fraction (the largest-
    magnitude weights — the ones contributing most to the activations) and
    zeros the rest. The mask is then re-applied (``apply()``) after **every**
    optimizer step for the remainder of training, so the pruned weights stay
    at exactly zero while the surviving weights keep adapting. The masked
    entries receive gradients but are immediately re-zeroed, so the GRU
    "heals" around the sparse skeleton — that fine-tuning is what lets a 90%-
    sparse GRU recover most of its dense performance.
    """

    def __init__(self, gru: nn.GRU, sparsity: float):
        self.gru = gru
        self.sparsity = float(sparsity)
        self.param_names = ["weight_ih_l0", "weight_hh_l0"]
        self.masks: dict[str, torch.Tensor] = {}
        self.active = False

    @torch.no_grad()
    def prune(self) -> None:
        for name in self.param_names:
            w = getattr(self.gru, name)
            flat = w.abs().flatten()
            n_keep = int(round((1.0 - self.sparsity) * flat.numel()))
            mask = torch.zeros_like(flat)
            if n_keep > 0:
                keep_idx = torch.topk(flat, n_keep, largest=True).indices
                mask[keep_idx] = 1.0
            self.masks[name] = mask.view_as(w)
        self.active = True
        self.apply()

    @torch.no_grad()
    def apply(self) -> None:
        if not self.active:
            return
        for name, mask in self.masks.items():
            getattr(self.gru, name).mul_(mask)

    def sparsity_now(self) -> float:
        total = zero = 0
        for name in self.param_names:
            w = getattr(self.gru, name)
            total += w.numel()
            zero += int((w == 0).sum().item())
        return zero / max(total, 1)


class PPOGRUPolicy(nn.Module):
    """Tile-embed → CNN → MLP → GRU → (Categorical move, belief scalar, value).

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
        # ``belief`` is an auxiliary map-recognition probe, NOT an action: a
        # deterministic tanh-bounded estimate of which map the agent is on
        # (+1 lake, -1 rocky, 0 balanced), supervised by an MSE aux loss using
        # the privileged map_type label. It is never sent to the env — it only
        # trains the GRU hidden state to encode the map identity. The build
        # itself is committed via the categorical move head (build_raft /
        # build_harness), learned through the slip reward downstream.
        self.belief_head = _layer_init(nn.Linear(gru_hidden, 1), std=0.01)
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
        belief = torch.tanh(self.belief_head(x)).squeeze(-1)  # (B,) in [-1, 1]
        value = self.critic(x).squeeze(-1)
        return logits, belief, value

    # ---- 1-step path (rollout collection) ------------------------------

    def get_action_and_value(self, obs, hidden, done, action=None):
        # add fake time dim of 1
        obs_seq = {k: v.unsqueeze(0) for k, v in obs.items()}
        gru_out, h_new = self._gru_forward(obs_seq, done.unsqueeze(0), hidden)
        x = gru_out.squeeze(0)
        logits, belief, value = self._heads(x)
        cat = Categorical(logits=logits)
        if action is None:
            action = cat.sample()
        log_prob = cat.log_prob(action)
        entropy = cat.entropy()
        return action, belief, log_prob, entropy, value, h_new

    # ---- T-step path (PPO update) --------------------------------------

    def evaluate(self, obs_seq, done_seq, hidden, actions):
        gru_out, _ = self._gru_forward(obs_seq, done_seq, hidden)
        T, B = gru_out.shape[:2]
        x = gru_out.reshape(T * B, -1)
        logits, belief, value = self._heads(x)
        cat = Categorical(logits=logits)
        actions_flat = actions.reshape(T * B)
        lp_a = cat.log_prob(actions_flat).reshape(T, B)
        ent = cat.entropy().reshape(T, B)
        return lp_a, ent, value.reshape(T, B), belief.reshape(T, B)


# =============================================================== training

def _to_device(obs: dict, device):
    return {k: torch.from_numpy(v).to(device) for k, v in obs.items()}


def main():
    parser = argparse.ArgumentParser()
    # env / data
    parser.add_argument("--env-size", type=int, default=64, choices=(32, 64, 96, 128))
    parser.add_argument("--map-type", default="random",
                        choices=("random", "lake", "rocky", "balanced"))
    parser.add_argument("--generator", default="simplex",
                        help="map generator(s) sampled per reset. One of "
                             "{simplex,components,composed}, or a "
                             "comma-separated mix (e.g. 'simplex,components') "
                             "for an augmented training distribution.")
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
    # reward-shaping overrides (applied to cogniland.nav.skills module globals,
    # which the env reads dynamically) — exposed so a sweep can tune the reward.
    parser.add_argument("--slack-penalty", type=float, default=None,
                        help="flat per-step penalty (default: skills.SLACK_PENALTY)")
    parser.add_argument("--shaping-coef", type=float, default=None,
                        help="PBRS shaping coefficient (default: skills.SHAPING_COEF)")
    parser.add_argument("--reach-bonus", type=float, default=None,
                        help="terminal reach bonus (default: skills.REACH_BONUS)")
    parser.add_argument("--grass-slip-noskill", type=float, default=None,
                        help="grass slip prob while NO skill is committed "
                             "(default: skills.SLIP_PROB_GRASS_NOSKILL = 0.0)")
    parser.add_argument("--clip-neg-shaping", action="store_true",
                        help="clip Δctg at 0 in PBRS shaping; backward steps "
                             "pay only the flat slack, no asymmetric "
                             "−SHAPING penalty.")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--belief-coef", type=float, default=0.5,
                        help="weight for the supervised belief MSE aux loss")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    # network
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    # --- GRU weight pruning (magnitude, one-shot) -----------------------
    parser.add_argument("--pruning-step", type=int, default=None,
                        help="env-step at which to magnitude-prune the GRU "
                             "weights to --sparsity. None = no pruning.")
    parser.add_argument("--sparsity", type=float, default=0.9,
                        help="target fraction of GRU weights set to zero")
    # infra
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="crafter_in_cogniland")
    parser.add_argument("--wandb-mode", default="online",
                        choices=("online", "offline", "disabled"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/ppo_checkpoints"),
                        help="parent dir; each run writes into "
                             "<checkpoint-dir>/<run_name>/{iter<N>.pt,final.pt}")
    parser.add_argument("--save-every-iters", type=int, default=300)
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML file of hyperparameters (arg names with "
                             "underscores); explicit CLI flags still override it")
    # Two-pass parse so a --config YAML sets defaults but CLI flags win.
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

    assert args.num_envs % args.num_minibatches == 0, \
        "num_envs must be divisible by num_minibatches (minibatching is over envs)"

    run_name = args.run_name or (
        f"ppo_gru_size{args.env_size}_seed{args.seed}_{int(time.time())}"
    )
    # tags are populated in the same key=value style as
    # scripts/crafter/dreamerv3_crafter_in_cogniland.py so the cross-algo W&B
    # workspace can filter them by "algo=" and "size=" identically.
    # Size is populated after the policy is constructed (param count
    # depends on env shape) and pushed through wandb.run.tags.
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        mode=args.wandb_mode,
        save_code=True,
        tags=[
            "algo=ppo_gru",
            f"map={args.env_size}",
            "env=cogniland_nav",
        ],
    )
    device = torch.device(args.device)
    print(f"device={device}  run_name={run_name}")

    # Apply reward-shaping overrides to the skills module globals (the env reads
    # them dynamically each step), so they take effect for every env instance.
    from cogniland.nav import skills as _sk
    if args.slack_penalty is not None:
        _sk.SLACK_PENALTY = float(args.slack_penalty)
    if args.shaping_coef is not None:
        _sk.SHAPING_COEF = float(args.shaping_coef)
    if args.reach_bonus is not None:
        _sk.REACH_BONUS = float(args.reach_bonus)
    if args.grass_slip_noskill is not None:
        _sk.SLIP_PROB_GRASS_NOSKILL = float(args.grass_slip_noskill)
    if args.clip_neg_shaping:
        _sk.CLIP_NEG_SHAPING = True
    print(f"reward: slack={_sk.SLACK_PENALTY} shaping={_sk.SHAPING_COEF} "
          f"reach={_sk.REACH_BONUS}  grass_slip_noskill={_sk.SLIP_PROB_GRASS_NOSKILL}  "
          f"clip_neg_shaping={_sk.CLIP_NEG_SHAPING}")

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
        generator=args.generator,
    )
    policy = PPOGRUPolicy(
        vec.single_observation_space,
        num_move_actions=vec.single_action_space.n,
        gru_hidden=args.gru_hidden,
        embed_dim=args.embed_dim,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    # GRU magnitude pruner (no-op unless --pruning-step is set). We keep our
    # own masks rather than torch.nn.utils.prune so the optimizer's parameter
    # references stay valid (prune reparametrises params into *_orig/*_mask).
    pruner = (GRUMagnitudePruner(policy.gru, args.sparsity)
              if args.pruning_step is not None else None)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy params: {n_params:,}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)
    # add a size tag once we know the param count (e.g. "size=1.4M")
    if wandb.run is not None:
        size_str = f"size={n_params / 1e6:.1f}M"
        wandb.run.tags = list(wandb.run.tags or []) + [size_str]

    # Each run writes into its own subdir so the checkpoints/ tree stays
    # readable when many runs share the parent (e.g. during a sweep).
    args.checkpoint_dir = args.checkpoint_dir / run_name
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
    beliefs_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    belief_targets_buf = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.float32, device=device
    )
    _MAP_TYPE_TO_BELIEF = {"lake": 1.0, "rocky": -1.0, "balanced": 0.0}
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
    # Display labels + permutations for the per-iteration skill matrix.
    # Internal indexing matches the JAX trainer:
    #   row: 0=balanced, 1=lake, 2=rocky
    #   col: 0=none, 1=raft, 2=harness
    # Display reorders to rows=(grassland, rocky, lake) ×
    # cols=(noskill, harness, raft). The matrix logged at each iteration
    # reflects only that iteration's finished episodes (not cumulative).
    _ROW_LABELS = ("grassland", "rocky", "lake")
    _ROW_PERM = (0, 2, 1)
    _COL_LABELS = ("noskill", "harness", "raft")
    _COL_PERM = (0, 2, 1)

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
        # Per-iteration (map_type, skill) finished-episode counts. Internal
        # indexing matches the JAX trainer:
        #   row: 0=balanced, 1=lake, 2=rocky
        #   col: 0=none, 1=raft, 2=harness
        # The cumulative version lives on the outer scope (`_skill_matrix_total`).
        iter_skill_counts = np.zeros((3, 3), dtype=np.int64)

        for step in range(args.num_steps):
            global_step += args.num_envs
            # One-shot GRU magnitude pruning once we cross --pruning-step.
            if pruner is not None and not pruner.active and global_step >= args.pruning_step:
                pruner.prune()
                print(f"[prune] GRU magnitude-pruned to {pruner.sparsity_now():.1%} "
                      f"sparsity at step {global_step}")
            for k in obs_buf:
                obs_buf[k][step] = next_obs_t[k]
            dones_buf[step] = next_done

            with torch.no_grad():
                action, belief, log_prob, _, value, next_hidden = policy.get_action_and_value(
                    next_obs_t, next_hidden, next_done
                )
            actions_buf[step] = action
            beliefs_buf[step] = belief
            logprobs_buf[step] = log_prob
            values_buf[step] = value

            # build is a discrete move now (build_raft / build_harness); the
            # belief is aux-only (map-recognition probe), never sent to the env.
            np_moves = action.cpu().numpy()
            next_obs, reward, done, infos = vec.step(np_moves)

            rewards_buf[step] = torch.from_numpy(reward).to(device)
            next_obs_t = _to_device(next_obs, device)
            next_done = torch.from_numpy(done.astype(np.float32)).to(device)
            # supervision target for the belief head (privileged at train time
            # only; the policy still has to infer this from local obs).
            belief_targets_buf[step] = torch.tensor(
                [_MAP_TYPE_TO_BELIEF[infos[i]["map_type"]] for i in range(args.num_envs)],
                dtype=torch.float32, device=device,
            )

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
                # Skill-usage matrix counts.
                row = {"balanced": 0, "lake": 1, "rocky": 2}.get(ep["map_type"])
                col = {"none": 0, "raft": 1, "harness": 2}.get(active)
                if row is not None and col is not None:
                    iter_skill_counts[row, col] += 1

        # -------- bootstrap + GAE ---------------------------------------
        with torch.no_grad():
            _, _, _, _, next_value, _ = policy.get_action_and_value(
                next_obs_t, next_hidden, next_done,
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
        belief_losses, belief_maes = [], []

        early_stop = False
        for epoch in range(args.update_epochs):
            np.random.shuffle(env_idx)
            for start in range(0, args.num_envs, envs_per_minibatch):
                mb = env_idx[start : start + envs_per_minibatch]
                mb_t = torch.from_numpy(mb).to(device)

                mb_obs = {k: v[:, mb_t] for k, v in obs_buf.items()}
                mb_dones = dones_buf[:, mb_t]
                mb_actions = actions_buf[:, mb_t]
                mb_belief_targets = belief_targets_buf[:, mb_t]
                mb_old_logp = logprobs_buf[:, mb_t]
                mb_adv = advantages[:, mb_t]
                mb_ret = returns[:, mb_t]
                mb_h0 = initial_hidden[:, mb_t]

                new_logp, ent, new_value, new_belief = policy.evaluate(
                    mb_obs, mb_dones, mb_h0, mb_actions
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
                belief_loss = (new_belief - mb_belief_targets).pow(2).mean()
                belief_mae = (new_belief - mb_belief_targets).abs().mean()

                loss = (
                    pg_loss
                    + args.vf_coef * v_loss
                    - args.ent_coef * ent_loss
                    + args.belief_coef * belief_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                # re-zero the pruned GRU weights after the optimizer update
                if pruner is not None:
                    pruner.apply()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
                kls.append(approx_kl.item())
                clipfracs.append(clipfrac.item())
                belief_losses.append(belief_loss.item())
                belief_maes.append(belief_mae.item())

            if args.target_kl is not None and np.mean(kls[-args.num_minibatches :]) > args.target_kl:
                early_stop = True
                break

        # -------- log ----------------------------------------------------
        # Shared schema with scripts/crafter/dreamerv3_crafter_in_cogniland.py:
        #   return/mean        — mean episode return
        #   return/rolling100  — same, rolling over last 100 finished episodes
        #   success/mean       — mean reach rate this iteration
        #   success/rolling100 — rolling success rate
        # These are the only keys the cross-algo W&B workspace plots,
        # so keep them stable.
        sps = global_step / (time.time() - start_time)
        log_payload = {
            "loss/policy": float(np.mean(pg_losses)),
            "loss/value": float(np.mean(v_losses)),
            "loss/entropy": float(np.mean(ent_losses)),
            "loss/belief": float(np.mean(belief_losses)),
            "train/approx_kl": float(np.mean(kls)),
            "train/clipfrac": float(np.mean(clipfracs)),
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/belief_mae": float(np.mean(belief_maes)),
            "train/iteration": iteration,
            "perf/fps": sps,
            "train/early_stop": int(early_stop),
        }
        if pruner is not None:
            log_payload["prune/gru_sparsity"] = pruner.sparsity_now()
        if ep_returns_recent:
            ret_mean = float(np.mean(ep_returns_recent))
            ret_rolling = float(np.mean(ep_returns_recent[-100:]))
            succ_mean = float(np.mean(ep_reached))
            succ_rolling = float(np.mean(ep_reached[-100:]))
            # Path-efficiency ratio: `min_steps / num_steps` per episode,
            # where `min_steps = 2 * env_size` is the worst-case Manhattan
            # span across the map. 1.0 means the agent walked exactly that
            # bound; > 1 means it took a shorter path (typical, since
            # spawn/target sit in the corner zones, not opposite corners).
            min_steps = 2 * args.env_size
            min_over_steps = float(np.mean(
                [min_steps / max(L, 1) for L in ep_lengths_recent]
            ))
            min_over_steps_rolling = float(np.mean(
                [min_steps / max(L, 1) for L in ep_lengths_recent[-100:]]
            ))
            log_payload.update({
                "return/mean": ret_mean,
                "return/rolling100": ret_rolling,
                "return/min_over_steps": min_over_steps,
                "return/min_over_steps_rolling100": min_over_steps_rolling,
                "success/mean": succ_mean,
                "success/rolling100": succ_rolling,
                "rollout/episode_length": float(np.mean(ep_lengths_recent)),
                "rollout/built_none_frac": (
                    float(np.mean(none_obj)) if none_obj else 0.0
                ),
                "rollout/built_wrong_frac": (
                    float(np.mean(wrong_obj)) if wrong_obj else 0.0
                ),
                "rollout/built_correct_frac": (
                    (sum(len(v) for v in match_obj.values()) /
                     max(1, len(ep_returns_recent)))
                ),
            })
            # ── Per-iteration skill-usage matrix (3x3 heatmap) ──
            # Row-normalise the count matrix so each row sums to 1 (or
            # zero if no episodes finished on that map type this iter).
            sm = iter_skill_counts.astype(np.float64)
            row_sums = sm.sum(axis=1, keepdims=True)
            norm = np.divide(
                sm, row_sums,
                out=np.zeros_like(sm), where=row_sums > 0,
            )
            norm_disp = norm[np.ix_(_ROW_PERM, _COL_PERM)]
            row_counts = row_sums.flatten()[list(_ROW_PERM)]
            # Per-cell scalars for line charts over time
            for i, row_lbl in enumerate(_ROW_LABELS):
                for j, col_lbl in enumerate(_COL_LABELS):
                    log_payload[f"skill_usage/{row_lbl}/{col_lbl}"] = float(norm_disp[i, j])
            # 3x3 heatmap image — wandb shows this directly in the panel
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(4.0, 3.6))
                im = ax.imshow(norm_disp, cmap="viridis", vmin=0.0, vmax=1.0)
                for i in range(3):
                    for j in range(3):
                        v = float(norm_disp[i, j])
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                color="white" if v < 0.5 else "black",
                                fontsize=11, fontweight="bold")
                ax.set_xticks(range(3))
                ax.set_yticks(range(3))
                ax.set_xticklabels(_COL_LABELS, fontsize=10)
                ax.set_yticklabels(
                    [f"{lbl}\n(n={int(row_counts[i])})"
                     for i, lbl in enumerate(_ROW_LABELS)], fontsize=10)
                ax.set_xlabel("skill built", fontsize=10)
                ax.set_ylabel("map type", fontsize=10)
                ax.set_title(f"skill usage  ·  iter {iteration}", fontsize=11)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                log_payload["skill_usage/matrix"] = wandb.Image(fig)
                plt.close(fig)
            except Exception:
                pass
        wandb.log(log_payload, step=global_step)

        if iteration % 5 == 0 or iteration == 1:
            er = log_payload.get("return/mean", float("nan"))
            succ = log_payload.get("success/mean", float("nan"))
            print(
                f"iter={iteration:4d}/{num_iterations}  step={global_step:>9d}  sps={sps:.0f}  "
                f"ret={er:+.2f}  success={succ:.2f}  "
                f"pg={log_payload['loss/policy']:+.3f}  "
                f"val={log_payload['loss/value']:.3f}  "
                f"kl={log_payload['train/approx_kl']:.4f}  "
                f"belief_mae={log_payload['train/belief_mae']:.3f}"
            )

        if iteration % args.save_every_iters == 0:
            ckpt = args.checkpoint_dir / f"iter{iteration}.pt"
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

    final_ckpt = args.checkpoint_dir / "final.pt"
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
