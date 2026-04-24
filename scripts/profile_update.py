"""Profile the PPO update cost alone.

Measures the time of one call to ``_run_all_updates`` (4 epochs * 4 minibatches
= 16 gradient steps on 16384 samples).
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
    state = agent.init(jax.random.PRNGKey(0))

    # Collect one rollout segment (this also triggers all jits).
    print("Running one training segment to measure end-to-end timing...")
    rng = jax.random.PRNGKey(1)
    t0 = time.perf_counter()
    # ``agent.train`` returns after running full rollout + PPO update
    # for num_train_frames. Measure one segment of 16384 frames
    # (= one rollout, one update set).
    new_state, metrics = agent.train(
        state, env, rng, 16384, progress_bar=None, task_ids=task_ids,
    )
    # Block on any pending device work.
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, new_state)
    t1 = time.perf_counter()
    seg_dt = t1 - t0
    frames = 16384
    print(f"one segment (rollout+update): {seg_dt:.3f}s   {frames/seg_dt:,.0f} fps")

    # Now run 3 more segments and average
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        new_state, metrics = agent.train(
            new_state, env, rng, 16384, progress_bar=None, task_ids=task_ids,
        )
        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, new_state)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"  seg {i+2}: {t1-t0:.3f}s   {frames/(t1-t0):,.0f} fps")

    avg = sum(times) / len(times)
    print(f"\navg of 3 post-warmup segments: {avg:.3f}s   {frames/avg:,.0f} fps")

    # Break out: how much time is the PPO update vs the rollout?
    # Strip agent.train to do ONLY rollout (no update). This is a rough
    # estimate — we re-run the inner loop manually.
    print("\n--- isolating rollout vs update ---")
    import importlib.util
    # Re-run a pure rollout (no update)
    obs = env.reset()
    from cogniland.agents.ppo_rnn import make_ppo_rnn  # noqa

    # Warm-up a compiled _sample_action
    # Using agent.select_action for simplicity:
    t0 = time.perf_counter()
    rng2 = jax.random.PRNGKey(2)
    num_steps = 128
    carry_reset_all = None
    for _ in range(num_steps):
        acts, state = agent.select_action(
            new_state, obs, rng2, training=True,
        )
        obs, _, _, _ = env.step(acts)
    # block
    t1 = time.perf_counter()
    rollout_dt = t1 - t0
    print(f"rollout only (128 steps): {rollout_dt:.3f}s   {num_steps*128/rollout_dt:,.0f} fps")
    print(f"update time estimate: {avg - rollout_dt:.3f}s  "
          f"(~{100*(avg-rollout_dt)/avg:.0f}% of segment)")


if __name__ == "__main__":
    main()
