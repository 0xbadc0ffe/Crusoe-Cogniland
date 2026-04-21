"""Model sanity diagnostics — fresh PPO-RNN on a 4-env balanced-biome env.

  1. forward_pass   — logits/value/carry ranges at init (uniform policy check)
  2. rollout_gae    — collect 128-step rollout, compute GAE, dump stats
  3. grad_norms     — one PPO update; print per-layer grad norms pre/post clip
  4. param_count    — total + per-layer parameters

Usage:
    python scripts/diagnose_model.py
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from cogniland.envs.registry import make_env
from cogniland.agents.registry import load_agent
from cogniland.agents.utils import count_parameters


def _build():
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/env/cogniland.yaml"),
        OmegaConf.load("configs/agent/ppo_rnn.yaml"),
    )
    cfg.env.num_parallel_envs = 4
    env = make_env(cfg.env_id, cfg, train=True)
    agent = load_agent(cfg)
    state = agent.init(jax.random.PRNGKey(0))
    return env, agent, state, cfg


def param_count(state):
    print("\n[4] param_count")
    total = count_parameters(state.train_state.params)
    print(f"  total params: {total}")
    # Per-layer
    params = state.train_state.params["params"]
    for name, val in params.items():
        n = sum(x.size for x in jax.tree.leaves(val))
        print(f"    {name:<30s} {n:>10d}")
    return total


def forward_pass(env, agent, state):
    print("\n[1] forward_pass — logits/value/carry at init")
    obs = env.reset()
    rng = jax.random.PRNGKey(1)
    # Low-level forward via the stored network apply
    params = state.train_state.params
    minimap = jnp.asarray(obs["minimap"])
    scalars = jnp.asarray(obs["scalars"])
    task_emb = jnp.asarray(obs["task_embedding"])
    carry = (jnp.zeros((4, 128)), jnp.zeros((4, 128)))
    logits, value, new_carry = state.train_state.apply_fn(
        params, minimap, scalars, task_emb, carry
    )
    probs = jax.nn.softmax(logits)
    log_probs = jax.nn.log_softmax(logits)
    entropy = float(-(probs * log_probs).sum(axis=-1).mean())
    print(f"  logits min/max: {float(logits.min()):+.3f} / {float(logits.max()):+.3f}")
    print(f"  probs mean/std: {float(probs.mean()):.3f} / {float(probs.std()):.3f}")
    print(f"  entropy: {entropy:.3f} (log(8) = {float(jnp.log(8)):.3f})")
    print(f"  value range: {float(value.min()):+.3f} ... {float(value.max()):+.3f}")
    print(f"  carry_h std: {float(new_carry[0].std()):.3f} carry_c std: {float(new_carry[1].std()):.3f}")


def rollout_gae(env, agent, state):
    print("\n[2] rollout_gae — 128-step rollout + GAE sanity")
    # Easier to just run agent.train for a tiny number of frames and inspect
    # metrics. It's JIT-first, so force CPU plays nice.
    T = 128
    obs = env.reset()
    B = env.num_envs
    rng = jax.random.PRNGKey(2)
    params = state.train_state.params
    apply_fn = state.train_state.apply_fn

    carry = (jnp.zeros((B, 128)), jnp.zeros((B, 128)))
    rewards = []
    values = []
    dones = []
    for t in range(T):
        minimap = jnp.asarray(obs["minimap"])
        scalars = jnp.asarray(obs["scalars"])
        task_emb = jnp.asarray(obs["task_embedding"])
        rng, act_rng = jax.random.split(rng)
        logits, value, new_carry = apply_fn(params, minimap, scalars, task_emb, carry)
        action = jax.random.categorical(act_rng, logits)
        action_np = np.asarray(action)
        obs, r, d, _ = env.step(action_np)
        rewards.append(r.copy()); values.append(np.asarray(value)); dones.append(d.copy())
        d_mask = jnp.asarray(d).reshape(-1, 1)
        carry = jax.tree.map(lambda c: jnp.where(d_mask, 0.0, c), new_carry)

    rewards = np.array(rewards)       # [T, B]
    values = np.array(values)          # [T, B]
    dones = np.array(dones).astype(np.float32)

    # Minimal GAE (matches ppo_rnn._compute_gae)
    gamma = 0.99
    gae_lambda = 0.95
    last_value = 0.0  # bootstrap approximation
    T_, B_ = rewards.shape
    adv = np.zeros_like(rewards)
    last_gae = np.zeros(B_, dtype=np.float32)
    for t in reversed(range(T_)):
        next_v = last_value if t == T_ - 1 else values[t + 1]
        next_done = 0.0 if t == T_ - 1 else dones[t + 1]
        delta = rewards[t] + gamma * next_v * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        adv[t] = last_gae

    rets = adv + values
    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
    done_frac = dones.mean()
    print(f"  reward min/max/mean: {rewards.min():+.3f} / {rewards.max():+.3f} / {rewards.mean():+.3f}")
    print(f"  value min/max/mean : {values.min():+.3f} / {values.max():+.3f} / {values.mean():+.3f}")
    print(f"  adv_raw  mean/std  : {adv.mean():+.3f} / {adv.std():.3f}")
    print(f"  adv_norm mean/std  : {adv_norm.mean():+.3f} / {adv_norm.std():.3f} (expect ~0/1)")
    print(f"  return   mean/std  : {rets.mean():+.3f} / {rets.std():.3f}")
    print(f"  done fraction      : {done_frac:.3f}  ({int(dones.sum())} terminations across {T}x{B} cells)")
    assert np.isfinite(adv).all(), "adv has NaN/Inf"
    assert np.isfinite(rets).all(), "returns have NaN/Inf"


def grad_norms(env, agent, state):
    print("\n[3] grad_norms — per-layer gradient norms pre/post global-norm clip")
    obs = env.reset()
    B = env.num_envs
    params = state.train_state.params
    apply_fn = state.train_state.apply_fn

    minimap = jnp.asarray(obs["minimap"])
    scalars = jnp.asarray(obs["scalars"])
    task_emb = jnp.asarray(obs["task_embedding"])
    carry = (jnp.zeros((B, 128)), jnp.zeros((B, 128)))
    # Fake targets to get a non-trivial loss
    returns = jnp.ones(B, dtype=jnp.float32)
    actions = jnp.zeros(B, dtype=jnp.int32)
    adv = jnp.ones(B, dtype=jnp.float32)

    def loss_fn(p):
        logits, value, _ = apply_fn(p, minimap, scalars, task_emb, carry)
        log_probs = jax.nn.log_softmax(logits)
        new_lp = log_probs[jnp.arange(B), actions]
        pg = -(adv * new_lp).mean()
        vloss = 0.5 * ((value - returns) ** 2).mean()
        probs = jax.nn.softmax(logits)
        entropy = -(probs * log_probs).sum(axis=-1).mean()
        return pg + 0.5 * vloss - 5e-3 * entropy

    grads = jax.grad(loss_fn)(params)
    pre = {}
    for name, val in grads["params"].items():
        gn = float(jnp.sqrt(sum((g ** 2).sum() for g in jax.tree.leaves(val))))
        pre[name] = gn
    total_pre = float(jnp.sqrt(sum(v ** 2 for v in pre.values())))

    # Apply global-norm clip at 0.5 (matches ppo_rnn.yaml)
    clip_val = 0.5
    scale = min(1.0, clip_val / (total_pre + 1e-8))
    post = {k: v * scale for k, v in pre.items()}

    print(f"  total grad norm (pre clip)  : {total_pre:.4f}")
    print(f"  total grad norm (post clip) : {total_pre * scale:.4f}  (scale={scale:.3f})")
    print(f"  per-layer:")
    for name in sorted(pre.keys()):
        print(f"    {name:<32s} pre={pre[name]:.4f}  post={post[name]:.4f}")


def main():
    env, agent, state, _ = _build()
    param_count(state)
    forward_pass(env, agent, state)
    rollout_gae(env, agent, state)
    grad_norms(env, agent, state)


if __name__ == "__main__":
    main()
