#!/usr/bin/env python3
"""DreamerV3 training for Cogniland Nav (PyTorch, single file).

This rewrite follows the DreamerV3 paper (Hafner et al. 2023) hyperparameters
closely. The salient choices:

* **Two-hot reward / critic** heads with symexp/symlog binning — bounded
  outputs prevent the value-target spiral that plagued earlier versions.
* **RetNorm**: percentile-based return normalization for the actor scale —
  S = max(1, Per(R, 95) − Per(R, 5)), updated by EMA.
* **Slow critic**: EMA target + slow-regularizer keeps critic targets
  consistent without bootstrapping into infinity.
* **RMSNorm + SiLU** throughout, **AGC(0.3)** gradient clipping,
  **LaProp(eps=1e-20)** optimizer.
* **Discrete action space** (6 actions: 4 moves + build_raft + build_harness),
  replacing the previous hybrid Categorical+Normal which was the main source
  of policy-gradient pathology.
* **Size presets** (--model-size small/medium/large/xlarge ≈ 12M / 25M / 50M
  / 100M params); default `medium` (25M).

Usage (RTX 4090):
    python scripts/train_dreamer.py \
        --env-size 64 --view-size 21 --tile-px 8 \
        --total-env-steps 1_000_000 \
        --model-size medium \
        --num-envs 4 --batch-size 16 --batch-length 64 \
        --device cuda --wandb-project cogniland-nav

After every `--imagine-every` updates the trainer writes an imagination video.
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
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cogniland.nav import CognilandNavEnv  # noqa: E402


# ─────────────────────────────────────────────────────────────── utils


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


# DreamerV3 paper: actions are 6 discrete = {up, down, left, right,
# build_raft, build_harness}. The env's hybrid (move + build_scalar) is
# packed into this single axis. Discretizing the scalar removes the
# unbounded-log-prob pathology of the previous Categorical+Normal setup.
ACTION_DIM = 6


def env_action_for(idx: int) -> dict:
    """Map discrete action idx → env action dict."""
    if idx < 4:
        return {"move": int(idx), "build_scalar": np.array([0.0], np.float32)}
    scalar = 1.0 if idx == 4 else -1.0  # 4=raft (+), 5=harness (-)
    return {"move": 4, "build_scalar": np.array([scalar], np.float32)}


# ──────────────────────────────────────────────────────── model presets


def model_size_config(name: str) -> dict:
    """Return (d, deter, cnn_d, codes) for a paper-aligned size preset.

    See DreamerV3 Table 3 — paper uses block GRU which trims the GRU
    param count by ~8x. Our naive GRUCell makes the recurrent layer
    larger, so we shrink `deter` here to keep total params near the
    paper target on Cogniland's 168x168 input. `d` is hidden width,
    `deter` the GRU state, `cnn_d` the base CNN channel count, and
    `codes` the number of one-hot codes per latent dimension.
    """
    table = {
        "small":  dict(d=256, deter=1024, cnn_d=16, codes=16),   # ≈7M wm
        "medium": dict(d=384, deter=2048, cnn_d=24, codes=24),   # ≈25M wm
        "large":  dict(d=512, deter=3072, cnn_d=32, codes=32),   # ≈55M wm
        "xlarge": dict(d=768, deter=4096, cnn_d=48, codes=48),   # ≈110M wm
    }
    if name not in table:
        raise ValueError(f"unknown size '{name}', expected one of {list(table)}")
    return table[name]


# ────────────────────────────────────────────────────── building blocks


def _init_linear(layer, std=1.0, bias=0.0):
    if hasattr(layer, "weight") and layer.weight is not None and layer.weight.dim() >= 2:
        nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias)
    return layer


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.scale


def _norm_act(features, act=True):
    layers = [RMSNorm(features)]
    if act:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


# ───────────────────────────────────────────────────────────── networks


class Encoder(nn.Module):
    """4 stride-2 convs (kernel 4) + 1×1 channel-bottleneck + Dense projection.

    Uses kernel=4, stride=2, padding=1 — for H=168 this gives
    168→84→42→21→10 (the last layer rounds down). A 1×1 bottleneck
    after the conv stack keeps the flattening tractable: e.g. d=256 with
    cnn_d=16 has 128 channels at 10×10 = 12800 flat values; bottlenecking
    to 8 channels gives 800 values and a much smaller dense projection.
    The decoder mirrors this exactly.
    """

    def __init__(self, image_shape, embed_dim: int = 1024, base: int = 24,
                 bottleneck: int = 8):
        super().__init__()
        C, H, W = image_shape
        depths = (base, base * 2, base * 4, base * 8)
        layers = []
        in_c = C
        for d in depths:
            layers += [
                nn.Conv2d(in_c, d, 4, stride=2, padding=1, bias=False),
                _norm_to_3d(d),
                nn.SiLU(),
            ]
            in_c = d
        # channel bottleneck
        layers += [
            nn.Conv2d(in_c, bottleneck, 1, bias=False),
            _norm_to_3d(bottleneck),
            nn.SiLU(),
        ]
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, C, H, W)).flatten(1).shape[1]
            spatial = self.cnn(torch.zeros(1, C, H, W)).shape[-2:]
        self.spatial = spatial
        self.bottleneck = bottleneck
        self.proj = nn.Linear(n_flat, embed_dim, bias=False)
        self.norm = RMSNorm(embed_dim)
        self.out_dim = embed_dim
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=1.0, a=-2, b=2)
        _init_linear(self.proj, std=1.0)

    def forward(self, image_uint8: torch.Tensor) -> torch.Tensor:
        x = image_uint8.float() / 255.0 - 0.5
        x = self.cnn(x).flatten(1)
        return F.silu(self.norm(self.proj(x)))


class _Norm2d(nn.Module):
    """RMSNorm over the channel dim of a (B, C, H, W) tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        norm = x.float().pow(2).mean(1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.scale


def _norm_to_3d(c):
    return _Norm2d(c)


class Decoder(nn.Module):
    """Inverse of Encoder: (feat_dim,) → (C, H, W) prediction.

    Mirrors the encoder's 4-layer stride-2 downsampling. To recover a
    target side length that isn't a pure power of two (e.g. 168 vs the
    encoder's 10×10 output), the first transpose conv may use
    ``output_padding=1`` to bump from 10→21 before the remaining
    doublings 21→42→84→168.
    """

    def __init__(self, feat_dim: int, target_shape, start_spatial=(10, 10),
                 base: int = 24, bottleneck: int = 8):
        super().__init__()
        C, H_t, W_t = target_shape
        depths = (base * 8, base * 4, base * 2, base)
        s_h, s_w = start_spatial
        self.start_h, self.start_w = s_h, s_w
        self.first_depth = depths[0]
        self.bottleneck = bottleneck
        self.target_h, self.target_w = H_t, W_t
        # mirror encoder: fc → (bottleneck × s_h × s_w) → 1×1 expand
        self.fc = nn.Linear(feat_dim, bottleneck * s_h * s_w, bias=False)
        self.fc_norm = _Norm2d(bottleneck)
        self.fc_expand = nn.Conv2d(bottleneck, depths[0], 1, bias=False)
        self.fc_expand_norm = _Norm2d(depths[0])
        # ratio H_target / start_h must be ~16. If exact 16: no extra
        # padding. If 16 < r < 32: pad first layer to bump by 1.
        ratio_h = H_t / s_h
        ratio_w = W_t / s_w
        # Each layer doubles unless it's the first and the ratio is odd-ish.
        # We compute output_padding per stage so the final spatial == target.
        sizes_h = [s_h]
        sizes_w = [s_w]
        for k in range(4):
            # target halved k+1 times from the end
            divisor = 2 ** (3 - k)
            sizes_h.append(int(round(H_t / divisor)))
            sizes_w.append(int(round(W_t / divisor)))
        # but our progression should at least double each step; clamp
        # to "next size that's a doubling or doubling+1"
        out_paddings_h, out_paddings_w = [], []
        cur_h, cur_w = s_h, s_w
        target_sizes_h, target_sizes_w = [], []
        for k in range(4):
            divisor = 2 ** (3 - k)
            t_h = int(round(H_t / divisor))
            t_w = int(round(W_t / divisor))
            # ConvTranspose2d kernel=4 stride=2 padding=1: out = 2*in + op
            op_h = t_h - 2 * cur_h
            op_w = t_w - 2 * cur_w
            op_h = max(0, min(1, op_h))
            op_w = max(0, min(1, op_w))
            out_paddings_h.append(op_h)
            out_paddings_w.append(op_w)
            target_sizes_h.append(2 * cur_h + op_h)
            target_sizes_w.append(2 * cur_w + op_w)
            cur_h = 2 * cur_h + op_h
            cur_w = 2 * cur_w + op_w
        layers = []
        in_c = depths[0]
        for i, d in enumerate(depths[1:] + (base,)):
            layers += [
                nn.ConvTranspose2d(
                    in_c, d, 4, stride=2, padding=1,
                    output_padding=(out_paddings_h[i], out_paddings_w[i]),
                    bias=False,
                ),
                _Norm2d(d),
                nn.SiLU(),
            ]
            in_c = d
        layers += [nn.Conv2d(in_c, C, 1, bias=True)]
        self.deconv = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.trunc_normal_(m.weight, std=1.0, a=-2, b=2)
        _init_linear(self.fc, std=1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.fc(x).view(B, self.bottleneck, self.start_h, self.start_w)
        x = F.silu(self.fc_norm(x))
        x = F.silu(self.fc_expand_norm(self.fc_expand(x)))
        x = self.deconv(x)
        # centre-crop / pad to target H/W (no-op if already exact)
        if x.shape[-1] > self.target_w or x.shape[-2] > self.target_h:
            sh = (x.shape[-2] - self.target_h) // 2
            sw = (x.shape[-1] - self.target_w) // 2
            x = x[:, :, sh:sh + self.target_h, sw:sw + self.target_w]
        elif x.shape[-1] < self.target_w or x.shape[-2] < self.target_h:
            ph = self.target_h - x.shape[-2]
            pw = self.target_w - x.shape[-1]
            x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        return x  # logits / pre-sigmoid


class RSSM(nn.Module):
    """Recurrent state-space model with discrete stochastic latents.

    Prior: p(z' | h, a). Posterior: q(z' | h, x). Discrete latent is
    `stoch_classes` × `stoch_dim` one-hots. Both prior and posterior
    logits are mixed with a small uniform component (`unimix`) to keep
    them away from degeneracy — the standard DreamerV3 trick.
    """

    def __init__(self, embed_dim: int, action_dim: int,
                 deter: int = 3072, stoch_classes: int = 24, stoch_dim: int = 24,
                 hidden: int = 384, unimix: float = 0.01):
        super().__init__()
        self.deter = deter
        self.classes = stoch_classes
        self.dim = stoch_dim
        self.unimix = unimix
        z_size = stoch_classes * stoch_dim

        self.action_dim = action_dim
        self.img_in = nn.Sequential(
            nn.Linear(z_size + action_dim, hidden, bias=False),
            RMSNorm(hidden), nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden, deter)
        self.img_out = nn.Sequential(
            nn.Linear(deter, hidden, bias=False),
            RMSNorm(hidden), nn.SiLU(),
        )
        self.prior_logits = nn.Linear(hidden, z_size)
        self.obs_out = nn.Sequential(
            nn.Linear(deter + embed_dim, hidden, bias=False),
            RMSNorm(hidden), nn.SiLU(),
        )
        self.post_logits = nn.Linear(hidden, z_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m, std=1.0)
        # output heads use small init
        nn.init.zeros_(self.prior_logits.weight); nn.init.zeros_(self.prior_logits.bias)
        nn.init.zeros_(self.post_logits.weight); nn.init.zeros_(self.post_logits.bias)
        # GRU bias-init: reset/update gates closed slightly so the model
        # tends to *retain* state — empirically much more stable on long
        # sequences than random init.
        nn.init.orthogonal_(self.gru.weight_ih)
        nn.init.orthogonal_(self.gru.weight_hh)
        nn.init.zeros_(self.gru.bias_ih); nn.init.zeros_(self.gru.bias_hh)

    def _logits_to_dist(self, logits):
        # apply unimix: a small uniform prior to avoid collapse
        probs = F.softmax(logits, dim=-1)
        probs = (1.0 - self.unimix) * probs + self.unimix / self.dim
        logits = torch.log(probs + 1e-10)
        return torch.distributions.OneHotCategoricalStraightThrough(logits=logits), logits

    def initial(self, batch: int, device) -> dict[str, torch.Tensor]:
        return {
            "deter": torch.zeros(batch, self.deter, device=device),
            "stoch": torch.zeros(batch, self.classes, self.dim, device=device),
            "logits": torch.zeros(batch, self.classes, self.dim, device=device),
        }

    def _stoch_to_flat(self, stoch):
        return stoch.reshape(stoch.shape[0], -1)

    def img_step(self, prev_state, prev_action):
        z_flat = self._stoch_to_flat(prev_state["stoch"])
        x = self.img_in(torch.cat([z_flat, prev_action], dim=-1))
        deter = self.gru(x, prev_state["deter"])
        feat = self.img_out(deter)
        prior_logits = self.prior_logits(feat).view(-1, self.classes, self.dim)
        dist, prior_logits = self._logits_to_dist(prior_logits)
        stoch = dist.rsample()
        return {"deter": deter, "stoch": stoch, "logits": prior_logits}

    def obs_step(self, prev_state, prev_action, embed):
        prior = self.img_step(prev_state, prev_action)
        post_in = self.obs_out(torch.cat([prior["deter"], embed], dim=-1))
        post_logits = self.post_logits(post_in).view(-1, self.classes, self.dim)
        dist, post_logits = self._logits_to_dist(post_logits)
        stoch = dist.rsample()
        return prior, {"deter": prior["deter"], "stoch": stoch, "logits": post_logits}

    def feat(self, state):
        return torch.cat([state["deter"], self._stoch_to_flat(state["stoch"])], dim=-1)

    @property
    def feat_dim(self) -> int:
        return self.deter + self.classes * self.dim


# ──────────────────────────────────────────────────── two-hot distribution


class TwoHotDist:
    """Two-hot encoding for symlog-spaced real-valued targets.

    Used for reward and value heads. Predictions live in the symlog space
    (so the model can learn over many orders of magnitude). The bins are
    a fixed grid in symlog space; the loss is cross-entropy between the
    head's logits and the two-hot encoding of symlog(target).
    """

    def __init__(self, logits: torch.Tensor, low: float = -20.0, high: float = 20.0):
        self.logits = logits
        self.n_bins = logits.shape[-1]
        self.bins = torch.linspace(low, high, self.n_bins, device=logits.device, dtype=logits.dtype)
        self.log_probs = F.log_softmax(logits, dim=-1)

    def mean(self) -> torch.Tensor:
        probs = self.log_probs.exp()
        return symexp((probs * self.bins).sum(-1))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        y = symlog(x)
        y_clamp = y.clamp(min=float(self.bins[0]), max=float(self.bins[-1]))
        # find the two surrounding bin indices
        below = (self.bins[None, :] <= y_clamp.unsqueeze(-1)).sum(-1) - 1
        below = below.clamp(0, self.n_bins - 1)
        above = (below + 1).clamp(0, self.n_bins - 1)
        # interpolation weights
        bins_below = self.bins[below]
        bins_above = self.bins[above]
        denom = (bins_above - bins_below).clamp(min=1e-8)
        w_below = (bins_above - y_clamp) / denom
        w_above = 1.0 - w_below
        # two-hot target
        # mask edge case where below == above (clamped)
        equal = (below == above)
        w_below = torch.where(equal, torch.ones_like(w_below), w_below)
        w_above = torch.where(equal, torch.zeros_like(w_above), w_above)
        # gather log_probs
        lp_below = self.log_probs.gather(-1, below.unsqueeze(-1)).squeeze(-1)
        lp_above = self.log_probs.gather(-1, above.unsqueeze(-1)).squeeze(-1)
        return w_below * lp_below + w_above * lp_above


class MLPHead(nn.Module):
    """Multi-layer MLP head with optional logits-output (n_bins) or scalar."""

    def __init__(self, feat_dim: int, hidden: int = 384, num_layers: int = 1,
                 out_dim: int = 1, outscale: float = 0.0):
        super().__init__()
        layers = []
        in_d = feat_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_d, hidden, bias=False), RMSNorm(hidden), nn.SiLU()]
            in_d = hidden
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(in_d, out_dim)
        for m in self.body.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m, std=1.0)
        if outscale == 0.0:
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        else:
            _init_linear(self.out, std=outscale)

    def forward(self, x):
        return self.out(self.body(x))


# ───────────────────────────────────────────────────────── replay buffer


class EpisodeReplay:
    """Cyclic buffer of single-step transitions; supports random subseq sampling."""

    def __init__(self, capacity: int, image_shape, device):
        self.capacity = capacity
        self.device = device
        self.image_shape = image_shape
        self.obs = np.zeros((capacity,) + image_shape, dtype=np.uint8)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.cont = np.zeros((capacity,), dtype=np.float32)   # 1 - done
        self.is_first = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(self, obs_img, action_idx, reward, done, is_first):
        i = self.ptr
        self.obs[i] = obs_img
        self.act[i] = int(action_idx)
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
        act = torch.from_numpy(self.act[idx]).to(self.device)
        rew = torch.from_numpy(self.rew[idx]).to(self.device)
        cont = torch.from_numpy(self.cont[idx]).to(self.device)
        is_first = torch.from_numpy(self.is_first[idx]).to(self.device)
        return {
            "image": obs.transpose(0, 1),
            "action": act.transpose(0, 1),
            "reward": rew.transpose(0, 1),
            "cont": cont.transpose(0, 1),
            "is_first": is_first.transpose(0, 1),
        }


# ─────────────────────────────────────────────────────── LaProp optimizer


class LaProp(torch.optim.Optimizer):
    """LaProp — gradient is normalised by sqrt(EMA(g²)) *before* the
    momentum update. Eps is added inside the sqrt-divisor to make it
    safe to use eps=1e-20 (as the DreamerV3 paper specifies).
    """

    def __init__(self, params, lr=4e-5, betas=(0.9, 0.999), eps=1e-20):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["m"], state["v"]
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                t = state["step"]
                v_hat = v / (1 - beta2 ** t)
                normed = g / (v_hat.sqrt() + eps)
                m.mul_(beta1).add_(normed, alpha=1 - beta1)
                m_hat = m / (1 - beta1 ** t)
                p.add_(m_hat, alpha=-lr)
        return loss


def agc_clip_(params, clip: float = 0.3, eps: float = 1e-3):
    """Adaptive gradient clipping (Brock et al. 2021) — per-parameter,
    scaled by the parameter norm. Closely mirrors optax's adaptive_grad_clip.
    """
    for p in params:
        if p.grad is None:
            continue
        param_norm = p.detach().norm().clamp(min=eps)
        grad_norm = p.grad.detach().norm()
        max_norm = clip * param_norm
        if grad_norm > max_norm:
            p.grad.mul_(max_norm / grad_norm.clamp(min=1e-8))


# ──────────────────────────────────────────────────────────── RetNorm


class RetNorm:
    """Percentile-EMA based return normalization for the actor scale.

    S = max(L, Per(R, hi) − Per(R, lo)), with `low` / `high` tracking
    EMAs of the 5th and 95th percentiles of imagined returns.
    """

    def __init__(self, lo: float = 0.05, hi: float = 0.95,
                 decay: float = 0.99, limit: float = 1.0):
        self.lo = lo
        self.hi = hi
        self.decay = decay
        self.limit = limit
        self.low = 0.0
        self.high = 0.0

    @torch.no_grad()
    def update(self, returns: torch.Tensor):
        flat = returns.flatten()
        if flat.numel() == 0:
            return self.scale()
        lo = torch.quantile(flat, self.lo).item()
        hi = torch.quantile(flat, self.hi).item()
        self.low = self.decay * self.low + (1 - self.decay) * lo
        self.high = self.decay * self.high + (1 - self.decay) * hi
        return self.scale()

    def scale(self) -> float:
        return max(self.limit, self.high - self.low)


# ────────────────────────────────────────────────────────── main loop


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
    parser.add_argument("--total-env-steps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--train-ratio", type=int, default=32,
                        help="model updates per env step (per env)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-length", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=200_000,
                        help="capacity in frames; 168x168 RGB → ~16GB at 200k")
    parser.add_argument("--prefill", type=int, default=4_096)
    # losses + optim (paper defaults)
    parser.add_argument("--lr-wm", type=float, default=4e-5)
    parser.add_argument("--lr-ac", type=float, default=4e-5)
    parser.add_argument("--gamma", type=float, default=0.997)         # 1/(1−γ)=333
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument("--free-nats", type=float, default=1.0)
    parser.add_argument("--beta-rep", type=float, default=0.1)
    parser.add_argument("--beta-dyn", type=float, default=1.0)
    parser.add_argument("--beta-pred-rew", type=float, default=1.0)
    parser.add_argument("--beta-pred-cont", type=float, default=1.0)
    parser.add_argument("--beta-pred-rec", type=float, default=1.0)
    parser.add_argument("--beta-actor", type=float, default=1.0)
    parser.add_argument("--beta-value", type=float, default=1.0)
    parser.add_argument("--actor-ent-coef", type=float, default=3e-4)
    parser.add_argument("--slow-ema-decay", type=float, default=0.98)
    parser.add_argument("--slow-reg-coef", type=float, default=1.0)
    parser.add_argument("--unimix", type=float, default=0.01)
    parser.add_argument("--num-reward-bins", type=int, default=255)
    parser.add_argument("--agc-clip", type=float, default=0.3)
    parser.add_argument("--opt-eps", type=float, default=1e-20)
    # model
    parser.add_argument("--model-size", default="medium",
                        choices=("small", "medium", "large", "xlarge"))
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="override; defaults to model-size preset")
    parser.add_argument("--deter", type=int, default=None)
    parser.add_argument("--stoch-classes", type=int, default=None)
    parser.add_argument("--stoch-dim", type=int, default=None)
    parser.add_argument("--cnn-base", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    # infra
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="cogniland-nav-dreamer")
    parser.add_argument("--wandb-mode", default="online",
                        choices=("online", "offline", "disabled"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-every-updates", type=int, default=5000)
    parser.add_argument("--imagine-every", type=int, default=2000)
    parser.add_argument("--imagine-dir", type=Path, default=Path("imagine"))
    parser.add_argument("--imagine-batch", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--compile", action="store_true", default=False,
                        help="torch.compile RSSM steps (~2x faster, +20s startup)")
    parser.add_argument("--no-tf32", action="store_true", default=False,
                        help="disable TF32 matmul (on by default)")
    args = parser.parse_args()

    # ── resolve model size preset ────────────────────────────────────
    preset = model_size_config(args.model_size)
    args.hidden = args.hidden or preset["d"]
    args.deter = args.deter or preset["deter"]
    args.cnn_base = args.cnn_base or preset["cnn_d"]
    args.stoch_classes = args.stoch_classes or preset["codes"]
    args.stoch_dim = args.stoch_dim or preset["codes"]
    args.embed_dim = args.embed_dim or args.hidden

    run_name = args.run_name or (
        f"dreamer_{args.model_size}_size{args.env_size}_seed{args.seed}_{int(time.time())}"
    )
    wandb.init(project=args.wandb_project, name=run_name,
               config=vars(args), mode=args.wandb_mode, save_code=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if not args.no_tf32 and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.imagine_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} run={run_name} size={args.model_size} "
          f"(d={args.hidden}, deter={args.deter}, cnn_d={args.cnn_base}, "
          f"codes={args.stoch_classes})")

    envs = [
        CognilandNavEnv(
            size=args.env_size, map_type=args.map_type, view_size=args.view_size,
            tile_px=args.tile_px, obs_mode="rgb", seed=args.seed + i,
            max_steps=args.max_steps,
        )
        for i in range(args.num_envs)
    ]
    image_shape = envs[0].observation_space["image"].shape  # (3, H, W)

    enc = Encoder(image_shape, embed_dim=args.embed_dim, base=args.cnn_base).to(device)
    rssm = RSSM(args.embed_dim, ACTION_DIM,
                deter=args.deter, stoch_classes=args.stoch_classes,
                stoch_dim=args.stoch_dim, hidden=args.hidden,
                unimix=args.unimix).to(device)
    # decoder spatial = the encoder's output spatial dim
    dec = Decoder(rssm.feat_dim, image_shape, start_spatial=enc.spatial,
                  base=args.cnn_base).to(device)
    rew_head = MLPHead(rssm.feat_dim, hidden=args.hidden, num_layers=1,
                       out_dim=args.num_reward_bins, outscale=0.0).to(device)
    cont_head = MLPHead(rssm.feat_dim, hidden=args.hidden, num_layers=1,
                        out_dim=1, outscale=1.0).to(device)
    actor = MLPHead(rssm.feat_dim, hidden=args.hidden, num_layers=2,
                    out_dim=ACTION_DIM, outscale=0.01).to(device)
    critic = MLPHead(rssm.feat_dim, hidden=args.hidden, num_layers=2,
                     out_dim=args.num_reward_bins, outscale=0.0).to(device)
    slow_critic = MLPHead(rssm.feat_dim, hidden=args.hidden, num_layers=2,
                          out_dim=args.num_reward_bins, outscale=0.0).to(device)
    slow_critic.load_state_dict(critic.state_dict())
    for p in slow_critic.parameters():
        p.requires_grad_(False)

    if args.compile:
        # Compile the per-step RSSM forwards — the 64-step Python loop is
        # the single biggest non-backward bottleneck. ~2x speedup at the
        # cost of ~20s startup compile time.
        rssm.obs_step = torch.compile(rssm.obs_step, dynamic=False)
        rssm.img_step = torch.compile(rssm.img_step, dynamic=False)
        print("torch.compile enabled on rssm.{obs_step, img_step}")

    wm_params = list(enc.parameters()) + list(rssm.parameters()) \
              + list(dec.parameters()) + list(rew_head.parameters()) \
              + list(cont_head.parameters())
    ac_params = list(actor.parameters()) + list(critic.parameters())

    opt_wm = LaProp(wm_params, lr=args.lr_wm, eps=args.opt_eps)
    opt_ac = LaProp(ac_params, lr=args.lr_ac, eps=args.opt_eps)

    n_wm = sum(p.numel() for p in wm_params)
    n_ac = sum(p.numel() for p in ac_params)
    print(f"world params: {n_wm:,}  actor+critic: {n_ac:,}")

    replay = EpisodeReplay(args.replay_capacity, image_shape, device)
    retnorm = RetNorm()

    # Per-env state
    obs_list = [env.reset()[0] for env in envs]
    is_first = [True] * args.num_envs
    ep_returns = [0.0] * args.num_envs
    ep_lens = [0] * args.num_envs
    ep_buffer = deque(maxlen=200)

    states = [rssm.initial(1, device) for _ in range(args.num_envs)]
    prev_actions = [torch.zeros(1, ACTION_DIM, device=device)
                    for _ in range(args.num_envs)]

    env_steps = 0
    updates = 0
    start_time = time.time()

    def actor_action(state, deterministic: bool = False):
        feat = rssm.feat(state)
        logits = actor(feat)
        # unimix
        probs = F.softmax(logits, dim=-1)
        probs = (1.0 - args.unimix) * probs + args.unimix / ACTION_DIM
        if deterministic:
            idx = int(probs.argmax(-1).item())
        else:
            idx = int(torch.distributions.Categorical(probs=probs).sample().item())
        return idx

    def env_step_one(i, deterministic=False):
        obs = obs_list[i]
        img = torch.from_numpy(obs["image"]).unsqueeze(0).to(device)
        with torch.no_grad():
            embed = enc(img)
            if is_first[i]:
                states[i] = rssm.initial(1, device)
                prev_actions[i] = torch.zeros(1, ACTION_DIM, device=device)
            _, post = rssm.obs_step(states[i], prev_actions[i], embed)
        states[i] = post
        if env_steps < args.prefill:
            idx = int(np.random.randint(0, ACTION_DIM))
        else:
            with torch.no_grad():
                idx = actor_action(post, deterministic=deterministic)
        action_vec = F.one_hot(torch.tensor(idx, device=device),
                               ACTION_DIM).float().unsqueeze(0)
        prev_actions[i] = action_vec
        env_action = env_action_for(idx)
        next_obs, reward, term, trunc, info = envs[i].step(env_action)
        done = term or trunc
        replay.add(obs["image"], idx, reward, done, is_first[i])
        ep_returns[i] += reward
        ep_lens[i] += 1
        if done:
            ep_buffer.append({
                "return": ep_returns[i],
                "length": ep_lens[i],
                "reached": bool(info["reached_target"]),
                "active_object": info.get("active_object", 0),
                "map_type": info.get("map_type", "random"),
            })
            ep_returns[i] = 0.0; ep_lens[i] = 0
            next_obs, _ = envs[i].reset()
            is_first[i] = True
        else:
            is_first[i] = False
        obs_list[i] = next_obs

    # ─── prefill ──────────────────────────────────────────────────────
    print("filling replay …")
    while env_steps < args.prefill:
        for i in range(args.num_envs):
            env_step_one(i)
            env_steps += 1
            if env_steps >= args.prefill:
                break
    print(f"prefill done — {env_steps} env steps; starting training")

    # ─── main loop ────────────────────────────────────────────────────
    train_debt = 0.0
    while env_steps < args.total_env_steps:
        for i in range(args.num_envs):
            env_step_one(i)
            env_steps += 1

        # how many train updates to run this outer tick?
        train_debt += args.train_ratio
        n_updates_now = int(train_debt)
        train_debt -= n_updates_now

        for _ in range(n_updates_now):
            batch = replay.sample(args.batch_size, args.batch_length)
            if batch is None:
                continue
            updates += 1
            metrics = train_step(
                batch, enc, rssm, dec, rew_head, cont_head,
                actor, critic, slow_critic,
                opt_wm, opt_ac, wm_params, ac_params,
                retnorm, args, device,
            )
            # slow critic EMA
            with torch.no_grad():
                d = args.slow_ema_decay
                for tp, p in zip(slow_critic.parameters(), critic.parameters()):
                    tp.mul_(d).add_(p, alpha=1 - d)

            if updates % args.log_every == 0:
                ep_ret = float(np.mean([e["return"] for e in ep_buffer])) if ep_buffer else float("nan")
                reach = float(np.mean([e["reached"] for e in ep_buffer])) if ep_buffer else 0.0
                sps = env_steps / max(time.time() - start_time, 1e-6)
                print(f"upd={updates:6d} env_step={env_steps:7d} sps={sps:.0f} "
                      f"wm={metrics['wm/loss']:+.3f} pol={metrics['actor/loss']:+.3f} "
                      f"val={metrics['critic/loss']:.3f} ret={ep_ret:+.2f} reach={reach:.2f}")
                wandb.log({
                    **metrics,
                    "charts/episode_return_mean": ep_ret,
                    "charts/reach_rate": reach,
                    "charts/env_steps": env_steps,
                    "charts/updates_per_sec": updates / max(time.time() - start_time, 1e-6),
                    "charts/sps": sps,
                    "charts/retnorm_low": retnorm.low,
                    "charts/retnorm_high": retnorm.high,
                }, step=env_steps)

            if updates % args.imagine_every == 0 and updates > 0:
                try:
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
                except Exception as e:
                    print(f"[warn] video render failed: {e}")

            if updates % args.save_every_updates == 0 and updates > 0:
                ckpt = args.checkpoint_dir / f"{run_name}_upd{updates}.pt"
                torch.save({
                    "enc": enc.state_dict(), "rssm": rssm.state_dict(),
                    "dec": dec.state_dict(), "rew": rew_head.state_dict(),
                    "cont": cont_head.state_dict(),
                    "actor": actor.state_dict(), "critic": critic.state_dict(),
                    "slow_critic": slow_critic.state_dict(),
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
        "slow_critic": slow_critic.state_dict(),
        "args": vars(args), "env_steps": env_steps, "updates": updates,
    }, final)
    wandb.save(str(final))
    wandb.finish()


# ──────────────────────────────────────── one model + AC update step


def _categorical_with_unimix(logits: torch.Tensor, unimix: float):
    probs = F.softmax(logits, dim=-1)
    n = probs.shape[-1]
    probs = (1.0 - unimix) * probs + unimix / n
    return torch.distributions.Categorical(probs=probs)


def kl_dyn_rep(prior_logits, post_logits, free: float):
    """Dyn KL = KL(stop_grad(q) || p), Rep KL = KL(q || stop_grad(p))."""
    # OneHot dim is the inner (D,) — sum over it after KL.
    prior = torch.distributions.OneHotCategorical(logits=prior_logits)
    post = torch.distributions.OneHotCategorical(logits=post_logits)
    prior_sg = torch.distributions.OneHotCategorical(logits=prior_logits.detach())
    post_sg = torch.distributions.OneHotCategorical(logits=post_logits.detach())
    # KL is computed per latent group (the OneHot dim); the .sum(-1) then
    # sums across the `classes` groups.
    dyn = torch.distributions.kl_divergence(post_sg, prior).sum(-1)
    rep = torch.distributions.kl_divergence(post, prior_sg).sum(-1)
    dyn = dyn.clamp(min=free)
    rep = rep.clamp(min=free)
    return dyn, rep


def train_step(batch, enc, rssm, dec, rew_head, cont_head,
                actor, critic, slow_critic,
                opt_wm, opt_ac, wm_params, ac_params,
                retnorm, args, device):
    T = args.batch_length
    B = args.batch_size

    images = batch["image"]        # (T, B, C, H, W) uint8
    actions = batch["action"]      # (T, B) int64
    rewards = batch["reward"]      # (T, B)
    conts = batch["cont"]          # (T, B)
    is_first = batch["is_first"]   # (T, B)

    actions_oh = F.one_hot(actions, ACTION_DIM).float()  # (T, B, A)

    # ── 1. encode all images
    images_flat = images.flatten(0, 1)
    embed = enc(images_flat).view(T, B, -1)

    # ── 2. roll RSSM
    state = rssm.initial(B, device)
    priors_logits, posts_logits = [], []
    posts_deter, posts_stoch = [], []
    for t in range(T):
        mask = (1.0 - is_first[t]).view(B, 1)
        state = {k: (v * mask if v.dim() == 2 else v * mask.unsqueeze(-1))
                 for k, v in state.items()}
        prev_a = actions_oh[t - 1] * mask if t > 0 else torch.zeros_like(actions_oh[0])
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

    # ── 3. WM losses
    recon = dec(feats_flat)                                  # logits
    image_target = images.float().flatten(0, 1) / 255.0 - 0.5
    image_loss = ((recon - image_target) ** 2).mean()

    rew_logits = rew_head(feats_flat).view(T, B, -1)
    rew_dist = TwoHotDist(rew_logits)
    reward_loss = -rew_dist.log_prob(rewards).mean()

    cont_logits = cont_head(feats_flat).view(T, B)
    cont_loss = F.binary_cross_entropy_with_logits(cont_logits, conts)

    dyn_kl, rep_kl = kl_dyn_rep(prior_logits.flatten(0, 1),
                                 post_logits.flatten(0, 1),
                                 free=args.free_nats)
    dyn_loss = dyn_kl.mean()
    rep_loss = rep_kl.mean()

    wm_loss = (
        args.beta_pred_rec * image_loss
        + args.beta_pred_rew * reward_loss
        + args.beta_pred_cont * cont_loss
        + args.beta_dyn * dyn_loss
        + args.beta_rep * rep_loss
    )

    opt_wm.zero_grad(set_to_none=True)
    wm_loss.backward()
    agc_clip_(wm_params, clip=args.agc_clip)
    opt_wm.step()

    # ── 4. imagined rollouts for AC
    H = args.imagine_horizon
    starts = {
        "deter": deter.flatten(0, 1).detach(),
        "stoch": stoch.flatten(0, 1).detach(),
        "logits": post_logits.flatten(0, 1).detach(),
    }
    cur = starts
    img_feats = []
    img_actions = []
    img_log_probs = []
    img_entropies = []
    for t in range(H):
        feat = rssm.feat(cur)
        img_feats.append(feat)
        logits = actor(feat)
        cat = _categorical_with_unimix(logits, args.unimix)
        idx = cat.sample()
        log_prob = cat.log_prob(idx)
        entropy = cat.entropy()
        img_log_probs.append(log_prob)
        img_entropies.append(entropy)
        a_vec = F.one_hot(idx, ACTION_DIM).float()
        img_actions.append(a_vec)
        cur = rssm.img_step(cur, a_vec)
    img_feats.append(rssm.feat(cur))   # bootstrap
    img_feats = torch.stack(img_feats)
    img_log_probs = torch.stack(img_log_probs)
    img_entropies = torch.stack(img_entropies)

    feats_all = img_feats.flatten(0, 1)
    # Reward & continue use the *just-updated* world-model heads; we detach
    # feats so AC grads don't leak back into the WM.
    with torch.no_grad():
        pred_r = TwoHotDist(rew_head(feats_all.detach())).mean().view(H + 1, -1)
        pred_c = torch.sigmoid(cont_head(feats_all.detach())).view(H + 1, -1)
        slow_v = TwoHotDist(slow_critic(feats_all.detach())).mean().view(H + 1, -1)

    # discount per step (gamma × predicted continue), with first column = 1
    disc = args.gamma * pred_c              # (H+1, *)
    # λ-returns (Hafner '23 style)
    target_v = slow_v.clone()
    vals = [target_v[H]]
    last = target_v[H]
    for t in reversed(range(H)):
        last = pred_r[t + 1] + disc[t + 1] * (
            (1 - args.gae_lambda) * target_v[t + 1] + args.gae_lambda * last
        )
        vals.append(last)
    vals.reverse()
    returns = torch.stack(vals[:H])  # (H, *)
    # ─── RetNorm + actor loss ────────────────────────────────────────
    scale = retnorm.update(returns.detach())
    norm_returns = returns.detach() / scale
    # baseline = slow critic value at the same timestep
    baseline = slow_v[:H] / scale
    advantage = (norm_returns - baseline).detach()

    pg = -(img_log_probs * advantage).mean()
    ent = -args.actor_ent_coef * img_entropies.mean()
    actor_loss = args.beta_actor * (pg + ent)

    # ─── Critic loss ─────────────────────────────────────────────────
    # Critic regresses (in two-hot space) on the λ-return *targets*. We
    # use the imagination feats[:H] (detached) — the critic only needs
    # grads through its own params.
    critic_in = img_feats[:H].flatten(0, 1).detach()
    critic_logits = critic(critic_in)
    critic_dist = TwoHotDist(critic_logits)
    nll_value = -critic_dist.log_prob(returns.detach().flatten()).mean()
    # slow-critic regularization: keep the critic close to its EMA twin
    with torch.no_grad():
        slow_logits = slow_critic(critic_in)
    slow_dist = TwoHotDist(slow_logits)
    slow_reg = -(F.softmax(slow_logits.detach(), -1) * critic_dist.log_probs).sum(-1).mean()
    critic_loss = args.beta_value * (nll_value + args.slow_reg_coef * slow_reg)

    ac_loss = actor_loss + critic_loss

    opt_ac.zero_grad(set_to_none=True)
    ac_loss.backward()
    agc_clip_(ac_params, clip=args.agc_clip)
    opt_ac.step()

    return {
        "wm/loss": wm_loss.item(),
        "wm/image_loss": image_loss.item(),
        "wm/reward_loss": reward_loss.item(),
        "wm/cont_loss": cont_loss.item(),
        "wm/dyn_loss": dyn_loss.item(),
        "wm/rep_loss": rep_loss.item(),
        "actor/loss": actor_loss.item(),
        "actor/pg": pg.item(),
        "actor/entropy": img_entropies.mean().item(),
        "actor/adv_mean": advantage.mean().item(),
        "actor/adv_std": advantage.std().item(),
        "critic/loss": critic_loss.item(),
        "critic/value_mean": TwoHotDist(critic_logits.detach()).mean().mean().item(),
        "critic/slow_reg": slow_reg.item(),
        "imag/return_mean": returns.mean().item(),
        "imag/return_std": returns.std().item(),
        "imag/reward_mean": pred_r.mean().item(),
        "imag/cont_mean": pred_c.mean().item(),
        "retnorm/scale": scale,
    }


if __name__ == "__main__":
    main()
