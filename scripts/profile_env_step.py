"""Profile the inside of env.step() — break it into sub-phases."""
from __future__ import annotations

import time
import numpy as np
from omegaconf import OmegaConf

from cogniland.config.env import setup_environment
setup_environment()

from cogniland.envs.env import CognilandEnv, _compute_minimap_batch
from cogniland.envs.registry import make_env


def main(num_envs: int = 256, num_steps: int = 200):
    env_cfg = OmegaConf.load("configs/env/cogniland.yaml")
    agent_cfg = OmegaConf.load("configs/agent/ppo_rnn.yaml")
    cfg = OmegaConf.merge(env_cfg, agent_cfg)
    cfg.env.num_parallel_envs = num_envs

    env_wrap = make_env(cfg.env_id, cfg, train=True)
    env_inner = env_wrap.env  # CognilandEnv

    obs = env_wrap.reset()

    rng = np.random.default_rng(0)

    # Warm up
    for _ in range(20):
        actions = rng.integers(0, 8, size=num_envs).astype(np.int32)
        env_wrap.step(actions)

    # Time wrapped step
    t0 = time.perf_counter()
    for _ in range(num_steps):
        actions = rng.integers(0, 8, size=num_envs).astype(np.int32)
        env_wrap.step(actions)
    wrap_total = (time.perf_counter() - t0) / num_steps

    # Time inner step
    env_wrap.reset()
    for _ in range(20):
        actions = rng.integers(0, 8, size=num_envs).astype(np.int32)
        env_inner.step(actions)
    t0 = time.perf_counter()
    for _ in range(num_steps):
        actions = rng.integers(0, 8, size=num_envs).astype(np.int32)
        env_inner.step(actions)
    inner_total = (time.perf_counter() - t0) / num_steps

    # Time just _get_obs
    t0 = time.perf_counter()
    for _ in range(num_steps):
        env_inner._get_obs()
    get_obs_time = (time.perf_counter() - t0) / num_steps

    # Time just _compute_minimap_batch
    t0 = time.perf_counter()
    for _ in range(num_steps):
        _compute_minimap_batch(
            env_inner._rgb, env_inner._heightmap, env_inner._terrain_idx,
            env_inner._berry_mask,
            env_inner.map_idx, env_inner.pos_r, env_inner.pos_c,
            env_inner.yes_r, env_inner.yes_c,
            env_inner.no_r, env_inner.no_c,
            env_inner._vis_per_terrain,
            vis_lut_packed=env_inner._vis_lut_packed,
            disk_stack=env_inner._disk_stack,
            occlude=env_inner._occlude,
        )
    minimap_time = (time.perf_counter() - t0) / num_steps

    # Time action branches only: reset + step with forced actions of each type
    env_wrap.reset()
    # Only movement
    t0 = time.perf_counter()
    for _ in range(num_steps):
        actions = rng.integers(0, 4, size=num_envs).astype(np.int32)
        env_inner.step(actions)
    move_time = (time.perf_counter() - t0) / num_steps

    env_wrap.reset()
    t0 = time.perf_counter()
    for _ in range(num_steps):
        actions = np.full(num_envs, 4, dtype=np.int32)
        env_inner.step(actions)
    forage_time = (time.perf_counter() - t0) / num_steps

    env_wrap.reset()
    t0 = time.perf_counter()
    for _ in range(num_steps):
        actions = np.full(num_envs, 5, dtype=np.int32)
        env_inner.step(actions)
    craft_time = (time.perf_counter() - t0) / num_steps

    # Time _reset_envs cost (ctg dijkstra)
    # Trigger many resets: kill all envs and step
    env_wrap.reset()
    env_inner.hp[:] = 1.0  # will die on first step
    t0 = time.perf_counter()
    actions = rng.integers(0, 4, size=num_envs).astype(np.int32)
    env_inner.step(actions)  # Will trigger full reset
    reset_driven_step = time.perf_counter() - t0

    print(f"\n=== num_envs={num_envs} ===")
    print(f"env_wrap.step:             {wrap_total*1000:6.2f} ms")
    print(f"env_inner.step:            {inner_total*1000:6.2f} ms  (wrapper adds {(wrap_total-inner_total)*1000:.2f} ms)")
    print(f"   └─ all moves only:      {move_time*1000:6.2f} ms")
    print(f"   └─ all forage only:     {forage_time*1000:6.2f} ms")
    print(f"   └─ all craft only:      {craft_time*1000:6.2f} ms")
    print(f"_get_obs:                  {get_obs_time*1000:6.2f} ms")
    print(f"   └─ _compute_minimap:    {minimap_time*1000:6.2f} ms")
    print(f"step with full auto-reset: {reset_driven_step*1000:6.2f} ms (256 Dijkstra)")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    main(n)
