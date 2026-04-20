"""Time full segments of agent.train() — the metric that matches real training."""
from __future__ import annotations

import time
import numpy as np
from omegaconf import OmegaConf

from cogniland.config.env import setup_environment
setup_environment()

import jax
from cogniland.envs.registry import make_env
from cogniland.agents.registry import load_agent


def sync():
    (jax.numpy.zeros(1) + 0).block_until_ready()


def main(num_envs=256, num_steps=64, num_segments=5):
    env_cfg = OmegaConf.load("configs/env/cogniland.yaml")
    agent_cfg = OmegaConf.load("configs/agent/ppo_rnn.yaml")
    cfg = OmegaConf.merge(env_cfg, agent_cfg)
    cfg.env.num_parallel_envs = num_envs

    env = make_env(cfg.env_id, cfg, train=True)
    agent = load_agent(cfg)
    rng = jax.random.PRNGKey(cfg.seed)
    state = agent.init(rng)
    task_ids = np.zeros(num_envs, dtype=np.int32)
    seg_frames = num_envs * num_steps

    # Warm-up: first segment triggers full JIT compilation
    print(f"\nWarm-up segment (compiling)...")
    sync(); t0 = time.perf_counter()
    state, _ = agent.train(state, env, rng, seg_frames, task_ids=task_ids)
    sync(); warmup_time = time.perf_counter() - t0
    print(f"Warm-up segment: {warmup_time:.2f} s")

    # Measurement: N segments
    print(f"Timing {num_segments} segments ({seg_frames} frames each)...")
    sync(); t0 = time.perf_counter()
    for _ in range(num_segments):
        state, _ = agent.train(state, env, rng, seg_frames, task_ids=task_ids)
    sync(); total_time = time.perf_counter() - t0
    per_seg = total_time / num_segments
    fps = seg_frames / per_seg

    print(f"\n=== num_envs={num_envs}, num_steps={num_steps} ===")
    print(f"Per-segment wall: {per_seg*1000:.1f} ms")
    print(f"Frames per seg:   {seg_frames}")
    print(f"Throughput:       {fps:.0f} fps")
    print(f"5M frames ETA:    {5_000_000 / fps / 60:.1f} min")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    main(n)
