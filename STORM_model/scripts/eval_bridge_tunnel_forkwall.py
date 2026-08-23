"""Held-out evaluation of a trained STORM checkpoint on the bridge_tunnel
fork_wall task, using the SAME held-out map split (forkwall6k/test.pkl) that
the released PPO fork_wall agent is scored on (REGISTRY.md: 100% success),
so the reported number is directly comparable across PPO / Dreamer / STORM.

Usage:
    python -m scripts.eval_bridge_tunnel_forkwall \\
        --results-dir results/<run_id> --episodes 600 [--step N | --best]
"""
from argparse import ArgumentParser

from cl.config import setup_environment
setup_environment()

import numpy as np
from omegaconf import OmegaConf

from cl.agents import load_agent
from cl.environments import make_environment
from cl.trainer.checkpoint import load_checkpoint
from cl.trainer.utils import RNGManager


def get_args():
    p = ArgumentParser()
    p.add_argument("--results-dir", required=True,
                    help="results/<run_id> directory produced by scripts.train")
    p.add_argument("--env-name", default="BridgeTunnel/forkwall")
    p.add_argument("--maps-path", default="data/bridge_tunnel/forkwall6k/test.pkl")
    p.add_argument("--num-envs", type=int, default=24)
    p.add_argument("--episodes", type=int, default=600,
                    help="approx. number of held-out episodes to evaluate")
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    p.add_argument("--best", action="store_true", help="load the 'best' checkpoint instead")
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def main():
    args = get_args()

    run_config = OmegaConf.load(f"{args.results_dir}/checkpoints/run_config.yaml")
    config = OmegaConf.merge(run_config, OmegaConf.create({
        "seed": args.seed,
        "env": {
            "num_parallel_envs": args.num_envs,
            "num_parallel_envs_eval": args.num_envs,
            "maps_path": args.maps_path,
        },
    }))

    agent = load_agent(config)

    rng_manager = RNGManager(seed=args.seed)
    state = agent.init(rng_manager.get_key())

    ckpt_dir = f"{args.results_dir}/checkpoints/{args.env_name}"
    checkpoint_data, _ckpt_cfg, metadata = load_checkpoint(
        checkpoint_dir=ckpt_dir, step=args.step, load_best=args.best,
    )
    state = agent.state_from_checkpoint(checkpoint_data, state.runtime)
    print(f"Loaded checkpoint step={metadata.get('step')} metrics={metadata.get('metrics')}")

    env_config = OmegaConf.create({
        "seed": args.seed,
        "env": OmegaConf.to_container(config.env, resolve=True),
    })
    env = make_environment(args.env_name, env_config)

    num_eval_frames = args.episodes * args.max_steps
    print(f"Evaluating on {args.maps_path} for ~{num_eval_frames:,} frames "
          f"({args.num_envs} envs)...")

    metrics = agent.evaluate(state, env, rng_manager.get_key(), num_eval_frames)

    episode_info = metrics.get("episode_info")
    if episode_info is None:
        print("No episodes completed -- increase --episodes.")
        return

    returns = np.asarray(episode_info["returned_episode_returns"]).reshape(-1)
    lengths = np.asarray(episode_info["returned_episode_lengths"]).reshape(-1)
    successes = (returns > 0).astype(np.float32)

    print(f"episodes           : {len(returns)}")
    print(f"success rate        : {successes.mean():.4f}")
    print(f"mean return         : {returns.mean():.4f}")
    print(f"mean episode length : {lengths.mean():.1f}")


if __name__ == "__main__":
    main()
