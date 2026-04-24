"""Break down rollout time into components to find the FPS bottleneck.

Runs a short PPO rollout while timing each sub-stage:
  (a) obs host->device transfer          (jnp.asarray on agent side)
  (b) policy forward + sample            (_sample_action)
  (c) action device->host sync           (np.asarray)
  (d) env.step (CPU numpy)               (movement, drain, rewards)
  (e) env._get_obs (incl. jit minimap)   (CTG lookups + 8 h2d inside kernel)
  (f) transition bookkeeping (Python)
  (g) LSTM carry reset on done

Reports total and per-step μs for each stage, plus rollout fps.
"""
from __future__ import annotations

from cogniland.config import setup_environment
setup_environment()

import time
import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from cogniland.agents import load_agent
from cogniland.envs.registry import make_env


def _sync(x):
    if isinstance(x, (tuple, list)):
        for y in x:
            _sync(y)
    elif hasattr(x, "block_until_ready"):
        x.block_until_ready()


def main():
    cfg = OmegaConf.load("configs/env/cogniland.yaml")
    acfg = OmegaConf.load("configs/agent/ppo_rnn.yaml")
    cfg = OmegaConf.merge(cfg, acfg)
    cfg.env.num_parallel_envs = 128
    cfg.env.biome_filter = ["balanced"]
    cfg.offline = True

    env = make_env(cfg.env_id, cfg, train=True)
    task_ids = np.zeros(env.num_envs, dtype=np.int32)
    env.set_tasks(task_ids)

    agent = load_agent(cfg)
    rng = jax.random.PRNGKey(0)
    state = agent.init(rng)

    num_envs = env.num_envs
    lstm_size = cfg.agent.lstm_size
    carry = (jnp.zeros((num_envs, lstm_size)), jnp.zeros((num_envs, lstm_size)))
    task_emb_jax = jnp.asarray(
        np.eye(7, dtype=np.float32)[task_ids]
    )

    params = state.train_state.params
    apply_fn = state.train_state.apply_fn

    # The agent's select path — replicate what's inside the training loop.
    @jax.jit
    def sample_action(params, mm, sc, te, carry, rng):
        logits, value, new_carry = apply_fn(params, mm, sc, te, carry)
        act_rng = rng
        actions = jax.random.categorical(act_rng, logits)
        log_prob = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), actions]
        return actions, log_prob, value, new_carry

    # Warm-up: a few steps to compile jits.
    obs = env.reset()
    for _ in range(5):
        mm = jnp.asarray(obs["minimap"])
        sc = jnp.asarray(obs["scalars"])
        rng, k = jax.random.split(rng)
        actions_jax, _, _, new_carry = sample_action(params, mm, sc, task_emb_jax, carry, k)
        _sync(actions_jax)
        actions_np = np.asarray(actions_jax)
        obs, _, _, _ = env.step(actions_np)
        carry = new_carry

    num_steps = 256
    t = {k: 0.0 for k in ("h2d", "forward", "d2h", "env_step", "carry_reset", "store")}
    storage: list = []
    t0 = time.perf_counter()

    for _ in range(num_steps):
        tt = time.perf_counter()
        mm = jnp.asarray(obs["minimap"])
        sc = jnp.asarray(obs["scalars"])
        _sync(mm); _sync(sc)
        t["h2d"] += time.perf_counter() - tt

        tt = time.perf_counter()
        rng, k = jax.random.split(rng)
        actions_jax, log_prob, value, new_carry = sample_action(
            params, mm, sc, task_emb_jax, carry, k,
        )
        _sync(actions_jax); _sync(log_prob); _sync(value); _sync(new_carry)
        t["forward"] += time.perf_counter() - tt

        tt = time.perf_counter()
        actions_np = np.asarray(actions_jax)
        t["d2h"] += time.perf_counter() - tt

        tt = time.perf_counter()
        obs, rewards, dones, info = env.step(actions_np)
        # Ensure minimap jax compute inside env._get_obs actually finishes
        if hasattr(obs["minimap"], "block_until_ready"):
            obs["minimap"].block_until_ready()
        t["env_step"] += time.perf_counter() - tt

        tt = time.perf_counter()
        done_mask = jnp.asarray(dones).reshape(-1, 1)
        carry = jax.tree.map(lambda c: jnp.where(done_mask, 0.0, c), new_carry)
        _sync(carry)
        t["carry_reset"] += time.perf_counter() - tt

        tt = time.perf_counter()
        storage.append((mm, sc, actions_jax, log_prob, value))
        t["store"] += time.perf_counter() - tt

    total = time.perf_counter() - t0
    frames = num_steps * num_envs
    fps = frames / total
    print(f"\n=== rollout: {num_steps} steps x {num_envs} envs = {frames} frames ===")
    print(f"total: {total:.3f}s   throughput: {fps:,.0f} fps")
    print()
    total_timed = sum(t.values())
    for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
        us_per_step = 1e6 * v / num_steps
        pct = 100 * v / total
        print(f"  {k:14s}  {v*1000:8.1f}ms  ({pct:5.1f}%)   {us_per_step:6.1f}μs/step")
    unaccounted = total - total_timed
    print(f"  {'unaccounted':14s}  {unaccounted*1000:8.1f}ms  "
          f"({100*unaccounted/total:5.1f}%)")


if __name__ == "__main__":
    main()
