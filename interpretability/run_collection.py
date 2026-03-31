#!/usr/bin/env python3
"""CLI for data collection: run a trained model on maps and save trajectories.

Usage::

    python interpretability/run_collection.py \
        --model-path artifacts/good_old/ckpt_best.pt \
        --output-dir interpretability/data/ \
        --episodes-per-map 3 \
        --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cogniland.env.types import EnvConfig
from cogniland.models.ppo import ActorCritic
from cogniland.utils import load_checkpoint
from interpretability.collect import TrajectoryCollector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect trajectories with neural activations.")
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to model checkpoint (e.g., artifacts/good_old/ckpt_best.pt)")
    p.add_argument("--test-maps", type=str, default="data/test_seed42_n16.pt",
                   help="Path to test maps .pt file")
    p.add_argument("--behavioral-maps", type=str, default="data/test_behavior.pt",
                   help="Path to behavioral test maps")
    p.add_argument("--output-dir", type=str, default="interpretability/data/",
                   help="Output directory for HDF5 and CSV")
    p.add_argument("--episodes-per-map", type=int, default=3,
                   help="Number of episodes per test map")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device to run on (cpu or cuda)")
    p.add_argument("--store-minimaps", action="store_true",
                   help="Store full minimap observations (large)")

    # Architecture overrides — auto-detected from checkpoint if not specified
    p.add_argument("--scalar-dim", type=int, default=None)
    p.add_argument("--minimap-channels", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--action-dim", type=int, default=None)
    p.add_argument("--cnn-channels", type=int, default=None)
    p.add_argument("--cnn-out-spatial", type=int, default=None)
    p.add_argument("--scalar-hidden", type=int, default=None)
    p.add_argument("--seed", type=int, default=1042)

    return p.parse_args()


def _infer_arch_from_checkpoint(ckpt_path: str, device: str = "cpu") -> dict:
    """Infer ActorCritic architecture params from checkpoint weight shapes."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]

    # cnn.0.weight: [cnn_channels//2, minimap_channels, 3, 3]
    minimap_channels = sd["cnn.0.weight"].shape[1]
    cnn_channels = sd["cnn.3.weight"].shape[0]  # cnn.3 is the second conv → full cnn_channels

    # scalar_net.0.weight: [scalar_hidden, scalar_dim]
    scalar_hidden = sd["scalar_net.0.weight"].shape[0]
    scalar_dim = sd["scalar_net.0.weight"].shape[1]

    # trunk.0.weight: [hidden_dim, cnn_out + scalar_hidden]
    hidden_dim = sd["trunk.0.weight"].shape[0]
    trunk_in = sd["trunk.0.weight"].shape[1]
    cnn_out = trunk_in - scalar_hidden
    # cnn_out = cnn_channels * cnn_out_spatial^2
    import math
    cnn_out_spatial = int(math.isqrt(cnn_out // cnn_channels))

    # actor.weight: [action_dim, hidden_dim]
    action_dim = sd["actor.weight"].shape[0]

    arch = dict(
        scalar_dim=scalar_dim, minimap_channels=minimap_channels,
        hidden_dim=hidden_dim, action_dim=action_dim,
        cnn_channels=cnn_channels, cnn_out_spatial=cnn_out_spatial,
        scalar_hidden=scalar_hidden,
    )
    print(f"  Auto-detected architecture: {arch}")
    return arch


def main():
    args = parse_args()

    # Load model — auto-detect architecture from checkpoint
    print(f"Loading model from {args.model_path}")
    arch = _infer_arch_from_checkpoint(args.model_path, args.device)

    # Allow CLI overrides
    for k in arch:
        cli_val = getattr(args, k.replace("-", "_"), None)
        if cli_val is not None:
            arch[k] = cli_val

    model = ActorCritic(**arch).to(args.device)

    load_checkpoint(args.model_path, model, device=args.device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Environment config
    env_config = EnvConfig(device=args.device)

    # Collector
    collector = TrajectoryCollector(
        model, env_config, device=args.device,
        store_minimaps=args.store_minimaps,
    )

    all_trajectories = []

    # Test maps
    if Path(args.test_maps).exists():
        print(f"\nLoading test maps from {args.test_maps}")
        test_data = torch.load(args.test_maps, map_location="cpu", weights_only=True)
        test_maps = test_data["maps"]
        print(f"  {test_maps.shape[0]} maps, {args.episodes_per_map} episodes each")
        trajs = collector.collect_from_test_maps(
            test_maps, episodes_per_map=args.episodes_per_map, seed=args.seed,
        )
        all_trajectories.extend(trajs)
        print(f"  Collected {len(trajs)} test trajectories")
    else:
        print(f"Warning: test maps not found at {args.test_maps}")

    # Behavioral maps
    if Path(args.behavioral_maps).exists():
        print(f"\nLoading behavioral maps from {args.behavioral_maps}")
        trajs = collector.collect_from_behavioral_maps(
            behavioral_path=args.behavioral_maps, seed=args.seed + 5000,
        )
        all_trajectories.extend(trajs)
        print(f"  Collected {len(trajs)} behavioral trajectories")
    else:
        print(f"Warning: behavioral maps not found at {args.behavioral_maps}")

    # Save
    print(f"\nSaving {len(all_trajectories)} total trajectories to {args.output_dir}")
    collector.save(all_trajectories, output_dir=args.output_dir)

    # Quick summary
    outcomes = {}
    for t in all_trajectories:
        outcomes[t.outcome] = outcomes.get(t.outcome, 0) + 1
    print(f"\nOutcomes: {outcomes}")
    lengths = [t.episode_length for t in all_trajectories]
    print(f"Episode lengths: mean={sum(lengths)/len(lengths):.0f}, "
          f"min={min(lengths)}, max={max(lengths)}")


if __name__ == "__main__":
    main()
