"""Evaluate a trained checkpoint on the berry-detour test conditions.

Loads weights from `results/<run_id>/checkpoints/cogniland-v0/best` and runs
deterministic rollouts on balanced maps with ``min_spawn_target_manhattan=80``,
reporting the success rate on the hard-detour-required distribution.

Usage:
    python scripts/eval_berry_test.py results/injx92sl
    python scripts/eval_berry_test.py results/injx92sl --min-manhattan 80 \
        --num-episodes 200 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from cogniland.envs.registry import make_env
from cogniland.agents.registry import load_agent
from cogniland.trainer.checkpoint import CheckpointManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--min-manhattan", type=int, default=80)
    ap.add_argument("--num-episodes", type=int, default=200)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--which", type=str, default="best",
                    choices=("best", "last"))
    args = ap.parse_args()

    run_dir = Path(args.results_dir).resolve()
    ckpt_dir = run_dir / "checkpoints" / "cogniland-v0"
    if not ckpt_dir.exists():
        sys.exit(f"No checkpoint dir at {ckpt_dir}")

    # Build env at eval conditions: balanced + forced far spawn
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/env/cogniland.yaml"),
        OmegaConf.load("configs/agent/ppo_rnn.yaml"),
    )
    cfg.env.num_parallel_envs = args.num_envs
    cfg.env.num_parallel_envs_eval = args.num_envs
    cfg.env.spawn_distance_range = None
    cfg.env.min_spawn_target_manhattan = args.min_manhattan
    cfg.seed = args.seed

    env = make_env(cfg.env_id, cfg, train=False)
    agent = load_agent(cfg)

    # Load checkpoint
    mgr = CheckpointManager(
        checkpoint_dir=str(ckpt_dir),
        keep_last=3,
        save_best=True,
    )
    state_dict, _, meta = mgr.load(load_best=(args.which == "best"))
    print(f"Loaded {args.which} checkpoint step={meta.get('step', '?')} "
          f"metrics={meta.get('metrics', {})}")

    # Init fresh state to get the right runtime, then overwrite params
    state = agent.init(jax.random.PRNGKey(args.seed))
    state = agent.state_from_checkpoint(state_dict, state.runtime)
    print(f"Loaded checkpoint from {ckpt_dir / args.which}")

    # Run deterministic evaluation
    task_ids = np.zeros(env.num_envs, dtype=np.int32)
    env.set_tasks(task_ids)
    rng = jax.random.PRNGKey(args.seed + 1)

    returns, successes, lengths, distances = [], [], [], []
    import sys as _sys

    while len(successes) < args.num_episodes:
        # One evaluate call may collect many episodes; cap per-call frames to
        # keep the harness snappy.
        metrics = agent.evaluate(
            state, env, rng,
            num_eval_frames=20000,
            task_ids=task_ids,
        )
        info = metrics.get("episode_info", {})
        if not info:
            break
        done = np.asarray(jnp.array(info.get("returned_episode", [])).reshape(-1)).astype(bool)
        if not done.any():
            continue
        r = np.asarray(jnp.array(info["returned_episode_returns"]).reshape(-1))[done]
        l = np.asarray(jnp.array(info["returned_episode_lengths"]).reshape(-1))[done]
        s = np.asarray(jnp.array(info["task_success"]).reshape(-1))[done].astype(np.float32)
        returns.extend(r.tolist())
        lengths.extend(l.tolist())
        successes.extend(s.tolist())

    n = len(successes)
    if n == 0:
        sys.exit("No episodes completed — was min_manhattan too high?")

    success_rate = float(np.mean(successes))
    mean_ret = float(np.mean(returns))
    mean_len = float(np.mean(lengths))
    print(f"\n=== berry test: min_manhattan={args.min_manhattan}, episodes={n} ===")
    print(f"  success_rate: {success_rate*100:.1f}%")
    print(f"  mean_return:  {mean_ret:+.2f}")
    print(f"  mean_length:  {mean_len:.1f}")


if __name__ == "__main__":
    main()
