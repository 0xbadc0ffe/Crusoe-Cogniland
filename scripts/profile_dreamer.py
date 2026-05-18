#!/usr/bin/env python3
"""Per-component wallclock profile of the Dreamer training loop.

Runs a handful of warm-up steps, then times each block separately:
  - env stepping (collect 1 step per env)
  - replay sample
  - WM forward (encode + RSSM rollout + heads)
  - WM backward + optim
  - Imagination rollout (H=15 dreams)
  - AC backward + optim

Reports ms per step + cumulative %. Useful when sps drops from ~70 to ~6
to identify which block is the bottleneck.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Import the symbols we need from the training script
import importlib.util
spec = importlib.util.spec_from_file_location("td", Path(__file__).resolve().parents[0] / "train_dreamer.py")
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)

from cogniland.nav import CognilandNavEnv  # noqa: E402


def fmt(t_ms: float) -> str:
    return f"{t_ms:7.2f}ms"


class Tim:
    def __init__(self, name, sync=True):
        self.name = name
        self.sync = sync
        self.sum = 0.0
        self.n = 0

    def __enter__(self):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.sum += (time.perf_counter() - self.t0) * 1000.0
        self.n += 1

    def avg(self):
        return self.sum / max(self.n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", default="small")
    p.add_argument("--view-size", type=int, default=21)
    p.add_argument("--tile-px", type=int, default=8)
    p.add_argument("--env-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batch-length", type=int, default=64)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--imagine-horizon", type=int, default=15)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if os.environ.get("TF32", "1") == "1":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    # build like main() but smaller
    preset = td.model_size_config(args.model_size)
    hidden = preset["d"]
    deter = preset["deter"]
    cnn_base = preset["cnn_d"]
    codes = preset["codes"]

    envs = [
        CognilandNavEnv(size=args.env_size, view_size=args.view_size,
                        tile_px=args.tile_px, obs_mode="rgb", seed=i)
        for i in range(args.num_envs)
    ]
    env = envs[0]
    image_shape = env.observation_space["image"].shape
    print(f"image_shape={image_shape}  device={device}  size={args.model_size}")

    enc = td.Encoder(image_shape, embed_dim=hidden, base=cnn_base).to(device)
    rssm = td.RSSM(hidden, td.ACTION_DIM, deter=deter, stoch_classes=codes,
                   stoch_dim=codes, hidden=hidden, unimix=0.01).to(device)
    dec = td.Decoder(rssm.feat_dim, image_shape, start_spatial=enc.spatial,
                     base=cnn_base).to(device)
    rew_head = td.MLPHead(rssm.feat_dim, hidden=hidden, num_layers=1,
                          out_dim=255, outscale=0.0).to(device)
    cont_head = td.MLPHead(rssm.feat_dim, hidden=hidden, num_layers=1,
                           out_dim=1, outscale=1.0).to(device)
    actor = td.MLPHead(rssm.feat_dim, hidden=hidden, num_layers=2,
                       out_dim=td.ACTION_DIM, outscale=0.01).to(device)
    critic = td.MLPHead(rssm.feat_dim, hidden=hidden, num_layers=2,
                        out_dim=255, outscale=0.0).to(device)
    slow_critic = td.MLPHead(rssm.feat_dim, hidden=hidden, num_layers=2,
                             out_dim=255, outscale=0.0).to(device)
    slow_critic.load_state_dict(critic.state_dict())
    for q in slow_critic.parameters():
        q.requires_grad_(False)

    wm_params = list(enc.parameters()) + list(rssm.parameters()) \
              + list(dec.parameters()) + list(rew_head.parameters()) \
              + list(cont_head.parameters())
    ac_params = list(actor.parameters()) + list(critic.parameters())
    opt_wm = td.LaProp(wm_params, lr=4e-5, eps=1e-20)
    opt_ac = td.LaProp(ac_params, lr=4e-5, eps=1e-20)

    n_wm = sum(p.numel() for p in wm_params)
    n_ac = sum(p.numel() for p in ac_params)
    print(f"world params: {n_wm:,}  actor+critic: {n_ac:,}")

    # Optionally compile the per-step RSSM forward — the 64-step Python
    # loop launches 64×~10 = ~640 tiny kernels every batch, and even with
    # a fast GPU the per-launch overhead dominates. torch.compile keeps
    # the same Python control flow but produces fused kernels for the
    # body, which on RTX 40-series usually gives 2-3x on this shape.
    if os.environ.get("COMPILE_RSSM", "0") == "1":
        rssm.obs_step = torch.compile(rssm.obs_step, dynamic=False)
        rssm.img_step = torch.compile(rssm.img_step, dynamic=False)
        print("[profile] torch.compile enabled on rssm.{obs_step, img_step}")

    # synthetic batch like replay would produce
    T, B = args.batch_length, args.batch_size
    obs_buf = torch.randint(0, 255, (T, B, *image_shape), dtype=torch.uint8, device=device)
    act_buf = torch.randint(0, td.ACTION_DIM, (T, B), dtype=torch.long, device=device)
    rew_buf = torch.randn(T, B, device=device) * 0.1
    cont_buf = torch.ones(T, B, device=device)
    is_first_buf = torch.zeros(T, B, device=device)
    is_first_buf[0] = 1.0

    timers = {
        "wm/encode":   Tim("wm/encode"),
        "wm/rssm":     Tim("wm/rssm"),
        "wm/heads":    Tim("wm/heads"),
        "wm/backward": Tim("wm/backward"),
        "wm/optim":    Tim("wm/optim"),
        "img/rollout": Tim("img/rollout"),
        "img/heads":   Tim("img/heads"),
        "img/returns": Tim("img/returns"),
        "ac/backward": Tim("ac/backward"),
        "ac/optim":    Tim("ac/optim"),
        "env/step":    Tim("env/step"),
        "env/h2d":     Tim("env/h2d"),
        "env/enc":     Tim("env/enc"),
        "env/rssm":    Tim("env/rssm"),
        "env/actor":   Tim("env/actor"),
        "env/replay_add": Tim("env/replay_add"),
        "replay/sample": Tim("replay/sample"),
        "step/total":  Tim("step/total"),
    }

    class Args:
        free_nats = 1.0
        beta_pred_rec = 1.0; beta_pred_rew = 1.0; beta_pred_cont = 1.0
        beta_dyn = 1.0; beta_rep = 0.1
        beta_actor = 1.0; beta_value = 1.0
        actor_ent_coef = 3e-4
        agc_clip = 0.3
        gamma = 0.997; gae_lambda = 0.95
        imagine_horizon = args.imagine_horizon
        slow_reg_coef = 1.0
        unimix = 0.01
        batch_length = T
        batch_size = B

    retnorm = td.RetNorm()
    a = Args()

    use_bf16 = os.environ.get("BF16", "0") == "1"
    if use_bf16:
        print("[profile] bf16 autocast enabled")
    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)

    # Set up env state for env-step profiling — mirrors train_dreamer.main()
    obs_list = [e.reset()[0] for e in envs]
    is_first = [True] * args.num_envs
    states = [rssm.initial(1, device) for _ in range(args.num_envs)]
    prev_actions = [torch.zeros(1, td.ACTION_DIM, device=device) for _ in range(args.num_envs)]
    # Tiny replay just for add() timing — not actually used
    replay = td.EpisodeReplay(2048, image_shape, device)

    def env_step_profiled(i):
        with timers["env/h2d"]:
            img = torch.from_numpy(obs_list[i]["image"]).unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            with timers["env/enc"]:
                embed = enc(img)
            with timers["env/rssm"]:
                if is_first[i]:
                    states[i] = rssm.initial(1, device)
                    prev_actions[i] = torch.zeros(1, td.ACTION_DIM, device=device)
                _, post = rssm.obs_step(states[i], prev_actions[i], embed)
            states[i] = post
            with timers["env/actor"]:
                logits = actor(rssm.feat(post))
                probs = F.softmax(logits, dim=-1)
                probs = 0.99 * probs + 0.01 / td.ACTION_DIM
                idx = int(torch.distributions.Categorical(probs=probs).sample().item())
        action_vec = F.one_hot(torch.tensor(idx, device=device), td.ACTION_DIM).float().unsqueeze(0)
        prev_actions[i] = action_vec
        env_action = td.env_action_for(idx)
        with timers["env/step"]:
            next_obs, reward, term, trunc, info = envs[i].step(env_action)
        done = term or trunc
        with timers["env/replay_add"]:
            replay.add(obs_list[i]["image"], idx, reward, done, is_first[i])
        if done:
            next_obs, _ = envs[i].reset()
            is_first[i] = True
        else:
            is_first[i] = False
        obs_list[i] = next_obs

    for step in range(args.n_warmup + args.n_steps):
        if step == args.n_warmup:
            print(f"\n--- profiling for {args.n_steps} steps ---")

        with timers["step/total"], autocast:
            # ── env stepping (one step per env, like train_dreamer outer tick) ──
            for ie in range(args.num_envs):
                env_step_profiled(ie)

            with timers["replay/sample"]:
                pass  # using synthetic batch — skip actual sample

            actions_oh = F.one_hot(act_buf, td.ACTION_DIM).float()

            # ── wm forward ──
            with timers["wm/encode"]:
                images_flat = obs_buf.flatten(0, 1)
                embed = enc(images_flat).view(T, B, -1)

            with timers["wm/rssm"]:
                state = rssm.initial(B, device)
                priors_logits, posts_logits = [], []
                posts_deter, posts_stoch = [], []
                for t in range(T):
                    mask = (1.0 - is_first_buf[t]).view(B, 1)
                    state = {k: (v * mask if v.dim() == 2 else v * mask.unsqueeze(-1))
                             for k, v in state.items()}
                    prev_a = actions_oh[t-1] * mask if t > 0 else torch.zeros_like(actions_oh[0])
                    prior, post = rssm.obs_step(state, prev_a, embed[t])
                    priors_logits.append(prior["logits"])
                    posts_logits.append(post["logits"])
                    posts_deter.append(post["deter"])
                    posts_stoch.append(post["stoch"])
                    state = post
                prior_logits = torch.stack(priors_logits)
                post_logits = torch.stack(posts_logits)
                deter = torch.stack(posts_deter)
                stoch = torch.stack(posts_stoch)
                feats = torch.cat([deter, stoch.flatten(-2)], dim=-1)
                feats_flat = feats.flatten(0, 1)

            with timers["wm/heads"]:
                recon = dec(feats_flat)
                image_target = obs_buf.float().flatten(0, 1) / 255.0 - 0.5
                image_loss = ((recon - image_target) ** 2).mean()
                rew_logits = rew_head(feats_flat).view(T, B, -1)
                rew_dist = td.TwoHotDist(rew_logits)
                reward_loss = -rew_dist.log_prob(rew_buf).mean()
                cont_logits = cont_head(feats_flat).view(T, B)
                cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_buf)
                dyn, rep = td.kl_dyn_rep(prior_logits.flatten(0, 1),
                                          post_logits.flatten(0, 1), free=1.0)
                dyn_loss = dyn.mean(); rep_loss = rep.mean()
                wm_loss = image_loss + reward_loss + cont_loss + dyn_loss + 0.1 * rep_loss

            with timers["wm/backward"]:
                opt_wm.zero_grad(set_to_none=True)
                wm_loss.backward()
            with timers["wm/optim"]:
                td.agc_clip_(wm_params, clip=0.3)
                opt_wm.step()

            # ── imagine ──
            with timers["img/rollout"]:
                H = a.imagine_horizon
                starts = {
                    "deter": deter.flatten(0, 1).detach(),
                    "stoch": stoch.flatten(0, 1).detach(),
                    "logits": post_logits.flatten(0, 1).detach(),
                }
                cur = starts
                img_feats = []; img_actions = []; img_log_probs = []; img_entropies = []
                for t in range(H):
                    feat = rssm.feat(cur)
                    img_feats.append(feat)
                    logits = actor(feat)
                    cat = td._categorical_with_unimix(logits, 0.01)
                    idx = cat.sample()
                    log_prob = cat.log_prob(idx)
                    entropy = cat.entropy()
                    img_log_probs.append(log_prob); img_entropies.append(entropy)
                    a_vec = F.one_hot(idx, td.ACTION_DIM).float()
                    img_actions.append(a_vec)
                    cur = rssm.img_step(cur, a_vec)
                img_feats.append(rssm.feat(cur))
                img_feats = torch.stack(img_feats)
                img_log_probs = torch.stack(img_log_probs)
                img_entropies = torch.stack(img_entropies)

            with timers["img/heads"]:
                feats_all = img_feats.flatten(0, 1)
                with torch.no_grad():
                    pred_r = td.TwoHotDist(rew_head(feats_all.detach())).mean().view(H + 1, -1)
                    pred_c = torch.sigmoid(cont_head(feats_all.detach())).view(H + 1, -1)
                    slow_v = td.TwoHotDist(slow_critic(feats_all.detach())).mean().view(H + 1, -1)

            with timers["img/returns"]:
                disc = 0.997 * pred_c
                vals = [slow_v[H]]
                last = slow_v[H]
                for t_ in reversed(range(H)):
                    last = pred_r[t_+1] + disc[t_+1] * ((1 - 0.95) * slow_v[t_+1] + 0.95 * last)
                    vals.append(last)
                vals.reverse()
                returns = torch.stack(vals[:H])
                scale = retnorm.update(returns.detach())
                norm_returns = returns.detach() / scale
                baseline = slow_v[:H] / scale
                advantage = (norm_returns - baseline).detach()
                pg = -(img_log_probs * advantage).mean()
                ent = -3e-4 * img_entropies.mean()
                actor_loss = pg + ent
                critic_in = img_feats[:H].flatten(0, 1).detach()
                critic_logits = critic(critic_in)
                critic_dist = td.TwoHotDist(critic_logits)
                nll_value = -critic_dist.log_prob(returns.detach().flatten()).mean()
                with torch.no_grad():
                    slow_logits = slow_critic(critic_in)
                slow_reg = -(F.softmax(slow_logits.detach(), -1) * critic_dist.log_probs).sum(-1).mean()
                critic_loss = nll_value + slow_reg
                ac_loss = actor_loss + critic_loss

            with timers["ac/backward"]:
                opt_ac.zero_grad(set_to_none=True)
                ac_loss.backward()
            with timers["ac/optim"]:
                td.agc_clip_(ac_params, clip=0.3)
                opt_ac.step()

        if step < args.n_warmup:
            # reset timers after warm-up
            for k, tm in timers.items():
                tm.sum = 0.0; tm.n = 0

    # ── report ──
    print("\n=== profile (per-step averages) ===")
    total = timers["step/total"].avg()
    print(f"{'block':<20} {'avg ms':>10} {'%':>6}")
    print("-" * 40)
    rows = [(k, v.avg()) for k, v in timers.items() if k != "step/total"]
    rows.sort(key=lambda r: -r[1])
    for k, ms in rows:
        pct = 100 * ms / total if total > 0 else 0
        print(f"{k:<20} {fmt(ms)} {pct:5.1f}%")
    print(f"{'TOTAL':<20} {fmt(total)} {100.0:5.1f}%")
    print(f"\nApprox throughput: {1000.0 / total:.2f} updates/sec")


if __name__ == "__main__":
    main()
