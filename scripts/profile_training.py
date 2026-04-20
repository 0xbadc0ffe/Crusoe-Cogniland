"""Profile the PPO-RNN training loop to find GPU-utilization bottlenecks.

Measures, per num_envs ∈ {128, 256, 512, 1024}:
  * env.reset()                          — one-shot, Dijkstra-dominated
  * obs -> jnp.asarray  (host -> device)
  * _sample_action       (GPU forward)
  * jax.block_until_ready on actions
  * np.asarray(actions)  (device -> host)
  * env.step              (CPU numpy)
  * Full unsync'd rollout wall time (what training actually sees)
  * PPO update step (forward + backward + optimizer)

Prints a breakdown table and computes:
  * % of wall time the GPU is actually busy
  * steps/sec throughput
  * projected speedup from (a) jit'd rollout, (b) bigger batch, (c) env on GPU.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from omegaconf import OmegaConf

from cogniland.config.env import setup_environment

setup_environment()

import jax
import jax.numpy as jnp

from cogniland.envs.registry import make_env
from cogniland.agents.registry import load_agent


def sync():
    (jax.numpy.zeros(1) + 0).block_until_ready()


@contextmanager
def timed(name, store):
    sync()
    t0 = time.perf_counter()
    yield
    sync()
    store[name] = store.get(name, 0.0) + (time.perf_counter() - t0)


@dataclass
class PhaseTimes:
    host_to_device: float = 0.0
    forward_gpu: float = 0.0
    device_to_host: float = 0.0
    env_step: float = 0.0
    bookkeeping: float = 0.0
    rollout_wall: float = 0.0
    ppo_update: float = 0.0
    reset_wall: float = 0.0
    compile_wall: float = 0.0


def build_config(num_envs: int):
    env_cfg = OmegaConf.load("configs/env/cogniland.yaml")
    agent_cfg = OmegaConf.load("configs/agent/ppo_rnn.yaml")
    cfg = OmegaConf.merge(env_cfg, agent_cfg)
    cfg.env.num_parallel_envs = num_envs
    return cfg


def profile_num_envs(num_envs: int, num_steps: int = 64, warmup: int = 8,
                     measure: int = 32) -> PhaseTimes:
    print(f"\n{'='*60}\nProfiling num_envs={num_envs}, num_steps={num_steps}")
    print('='*60)

    cfg = build_config(num_envs)

    # Env (registry builds CognilandEnv + MultiTaskEnvWrapper)
    env = make_env(cfg.env_id, cfg, train=True)

    # Agent
    agent = load_agent(cfg)
    rng = jax.random.PRNGKey(cfg.seed)
    state = agent.init(rng)

    # Measure reset (this is where Dijkstra cost-to-go is computed)
    t = PhaseTimes()
    sync()
    t0 = time.perf_counter()
    obs = env.reset()
    sync()
    t.reset_wall = time.perf_counter() - t0
    print(f"env.reset() wall time: {t.reset_wall*1000:.1f} ms "
          f"(Dijkstra for {num_envs} envs)")

    # Task embedding (one-hot, task 0 for all)
    task_emb = jnp.zeros((num_envs, cfg.task_embedding_dim), dtype=jnp.float32)
    task_emb = task_emb.at[:, 0].set(1.0)

    # LSTM carry
    lstm_size = cfg.agent.lstm_size
    carry = (jnp.zeros((num_envs, lstm_size)), jnp.zeros((num_envs, lstm_size)))

    # Pull jitted functions from the closure
    # ppo_rnn stores them as `_sample_action` inside make_ppo_rnn's closure,
    # but they're not directly accessible. Re-create via select_action path.
    # Instead, we access the agent's select_action with training=True.

    # ---- Warmup (compile) ----
    sync()
    t0 = time.perf_counter()
    for _ in range(warmup):
        minimap_jax = jnp.asarray(obs["minimap"])
        scalars_jax = jnp.asarray(obs["scalars"])
        rng, act_rng = jax.random.split(rng)
        # Use _sample_action via the private closure — grab it from the agent
        actions_np, state = agent.select_action(
            state, obs, act_rng, is_first=None, training=True,
        )
        actions_jax = jnp.asarray(actions_np)
        next_obs, rewards, dones, info = env.step(actions_np)
        obs = next_obs
    sync()
    t.compile_wall = time.perf_counter() - t0
    print(f"warmup/compile:        {t.compile_wall*1000:.1f} ms "
          f"({warmup} steps)")

    # ---- Measurement: fine-grained per-phase ----
    # We need a raw handle on the _sample_action — grab it via monkey-patch
    # from ppo_rnn module.
    from cogniland.agents import ppo_rnn as ppo_rnn_mod  # noqa
    # The Agent's `select_action` wraps all the pieces; we time that as the
    # "forward+transfer" combo, and also time the raw sub-phases manually.

    # Raw sub-phase measurement: re-do it with explicit syncs
    for _ in range(measure):
        # host -> device
        sync(); h0 = time.perf_counter()
        minimap_jax = jnp.asarray(obs["minimap"])
        scalars_jax = jnp.asarray(obs["scalars"])
        minimap_jax.block_until_ready()
        t.host_to_device += time.perf_counter() - h0

        # GPU forward
        rng, act_rng = jax.random.split(rng)
        sync(); h0 = time.perf_counter()
        actions_np, state = agent.select_action(
            state, obs, act_rng, is_first=None, training=True,
        )
        # select_action already calls np.asarray(), forcing the sync
        t.forward_gpu += time.perf_counter() - h0
        # (device->host already inside select_action via np.asarray)

        # env step (CPU)
        h0 = time.perf_counter()
        next_obs, rewards, dones, info = env.step(actions_np)
        t.env_step += time.perf_counter() - h0

        # bookkeeping (what the real rollout also does)
        h0 = time.perf_counter()
        _ = info.get("returned_episode", None)
        t.bookkeeping += time.perf_counter() - h0

        obs = next_obs

    # ---- Unsync'd rollout wall time (how fast training actually runs) ----
    sync()
    t0 = time.perf_counter()
    for _ in range(measure):
        rng, act_rng = jax.random.split(rng)
        actions_np, state = agent.select_action(
            state, obs, act_rng, is_first=None, training=True,
        )
        next_obs, rewards, dones, info = env.step(actions_np)
        obs = next_obs
    sync()
    t.rollout_wall = time.perf_counter() - t0

    # ---- PPO update timing ----
    # Simulate a full rollout of num_steps, then one update.
    storage_min, storage_scl, storage_act, storage_lp, storage_v, storage_r, storage_d = \
        [], [], [], [], [], [], []
    storage_ch, storage_cc = [], []
    carry_sim = (jnp.zeros((num_envs, lstm_size)), jnp.zeros((num_envs, lstm_size)))
    for _ in range(num_steps):
        rng, act_rng = jax.random.split(rng)
        minimap_jax = jnp.asarray(obs["minimap"])
        scalars_jax = jnp.asarray(obs["scalars"])
        actions_np, state = agent.select_action(
            state, obs, act_rng, is_first=None, training=True,
        )
        storage_min.append(minimap_jax)
        storage_scl.append(scalars_jax)
        next_obs, rewards, dones, info = env.step(actions_np)
        obs = next_obs
    sync()

    # Stack into [T*B, ...] for a single ppo-update-sized minibatch
    flat_mm = jnp.concatenate(storage_min, axis=0)
    flat_sc = jnp.concatenate(storage_scl, axis=0)
    flat_size = flat_mm.shape[0]
    minibatch_size = flat_size // cfg.agent.num_minibatches

    # Full-segment timing. Call once first to trigger _run_all_updates
    # JIT compilation (multi-second the first time), then time the second
    # call. Subtract measured rollout cost to get a clean ppo_update time.
    seg_frames = num_envs * num_steps
    task_ids_np = np.zeros(num_envs, dtype=np.int32)

    # Warm-up call to compile _run_all_updates
    sync()
    state, _m = agent.train(state, env, rng, seg_frames, task_ids=task_ids_np)
    sync()

    sync()
    t0 = time.perf_counter()
    state, _m = agent.train(state, env, rng, seg_frames, task_ids=task_ids_np)
    sync()
    one_seg_wall = time.perf_counter() - t0
    expected_rollout = t.rollout_wall * num_steps
    t.ppo_update = max(0.0, one_seg_wall - expected_rollout)

    # ---- Normalise to per-step ----
    t.host_to_device /= measure
    t.forward_gpu /= measure
    t.env_step /= measure
    t.bookkeeping /= measure
    t.rollout_wall /= measure

    return t


def print_report(results: dict[int, PhaseTimes]):
    print(f"\n{'='*90}\nBREAKDOWN (time per env.step, ms):")
    print('='*90)
    header = f"{'num_envs':>10} {'h->d':>8} {'forward':>10} {'env.step':>10} " \
             f"{'bookkeep':>10} {'wall':>8} {'frames/s':>12} {'reset(ms)':>10} {'ppo_upd(ms)':>12}"
    print(header)
    print('-'*len(header))
    for n, t in results.items():
        fps = n / t.rollout_wall if t.rollout_wall > 0 else 0.0
        print(f"{n:>10} {t.host_to_device*1000:>8.2f} {t.forward_gpu*1000:>10.2f} "
              f"{t.env_step*1000:>10.2f} {t.bookkeeping*1000:>10.2f} "
              f"{t.rollout_wall*1000:>8.2f} {fps:>12.0f} "
              f"{t.reset_wall*1000:>10.1f} {t.ppo_update*1000:>12.1f}")

    print(f"\n{'='*90}")
    print("INTERPRETATION:")
    print('='*90)
    for n, t in results.items():
        total = t.host_to_device + t.forward_gpu + t.env_step
        if total == 0:
            continue
        gpu_frac = t.forward_gpu / total * 100
        cpu_frac = (t.env_step + t.host_to_device) / total * 100
        print(f"\n num_envs={n}:")
        print(f"   GPU busy:     {gpu_frac:5.1f}% of step time")
        print(f"   CPU busy:     {cpu_frac:5.1f}% of step time")
        print(f"   Rollout wall: {t.rollout_wall*1000:.2f} ms/step ({n/t.rollout_wall:,.0f} fps)")


if __name__ == "__main__":
    import sys
    configs = [128, 256, 512, 1024] if len(sys.argv) == 1 else [int(x) for x in sys.argv[1:]]
    results = {}
    for n in configs:
        try:
            results[n] = profile_num_envs(n)
        except Exception as e:
            print(f"Failed at num_envs={n}: {e}")
            import traceback; traceback.print_exc()
    print_report(results)
