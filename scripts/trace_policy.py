"""Dump 1 episode of actions + state for a trained checkpoint or fresh model.

Usage:
    python scripts/trace_policy.py                      # fresh model
    python scripts/trace_policy.py <checkpoint_dir>     # load weights
"""

from __future__ import annotations
import sys
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from cogniland.envs.registry import make_env
from cogniland.agents.registry import load_agent


def main():
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/env/cogniland.yaml"),
        OmegaConf.load("configs/agent/ppo_rnn.yaml"),
    )
    cfg.env.num_parallel_envs = 1
    env = make_env(cfg.env_id, cfg, train=True)
    agent = load_agent(cfg)
    state = agent.init(jax.random.PRNGKey(123))

    print(f"param count: {sum(x.size for x in jax.tree.leaves(state.train_state.params))}")

    # Per-step greedy trace
    obs = env.reset()
    print(f"map idx={env.env.map_idx.tolist()} biome={env.env._biomes[env.env.map_idx].tolist()}")
    print(f"spawn=({env.env.spawn_r[0]},{env.env.spawn_c[0]}) yes=({env.env.yes_r[0]},{env.env.yes_c[0]}) "
          f"hp_init={env.env.hp[0]} ctg_spawn={env.env.ctg_spawn[0]}")

    actions_taken = np.zeros(8, dtype=np.int32)
    ep_return = 0.0
    max_steps = 80
    for t in range(max_steps):
        mm = jnp.asarray(obs["minimap"])
        sc = jnp.asarray(obs["scalars"])
        te = jnp.asarray(obs["task_embedding"])
        # Forward pass - greedy
        logits, value, _ = state.train_state.apply_fn(
            state.train_state.params, mm, sc, te,
            (jnp.zeros((1, 128)), jnp.zeros((1, 128)))
        )
        probs = jax.nn.softmax(logits)
        a_greedy = int(jnp.argmax(logits, axis=-1)[0])
        a_sample = int(jax.random.categorical(jax.random.PRNGKey(t), logits)[0])
        action = np.array([a_sample], dtype=np.int32)
        actions_taken[a_sample] += 1
        obs, r, d, info = env.step(action)
        ep_return += float(r[0])
        if t < 10 or bool(d[0]):
            print(f"  t={t:2d} sampled={a_sample} greedy={a_greedy} probs={np.asarray(probs[0]).round(3).tolist()}  "
                  f"value={float(value[0]):+.3f} r={float(r[0]):+.3f} hp={env.env.hp[0]:.1f} "
                  f"pos=({env.env.pos_r[0]},{env.env.pos_c[0]}) ctg={float(info['ctg_curr'][0]):.1f}")
        if bool(d[0]):
            print(f"  episode over at t={t}, alive={bool(info['alive'][0])}, reached={bool(info['reached'][0])}")
            break
    print(f"\naction distribution: {actions_taken.tolist()} (sampled)")
    print(f"ep_return: {ep_return:+.3f}")


if __name__ == "__main__":
    main()
