#!/usr/bin/env python3
"""Load a Dreamer checkpoint and render an imagined-trajectory video.

This is the offline counterpart to the periodic videos that
``train_dreamer.py`` writes during training. Point it at a checkpoint
(produced by the trainer's ``--save-every-updates`` flag) and a few
short episodes of real env interaction will be replayed into the world
model, then the model dreams forward and the result is written to mp4.

Usage
-----

    python scripts/imagine_video.py \\
        --checkpoint checkpoints/dreamer_size64_seed0_<ts>_upd5000.pt \\
        --out imagine/manual_inspect.mp4 \\
        --episodes 4 --horizon 64
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cogniland.nav import CognilandNavEnv  # noqa: E402
from cogniland.nav_dreamer_video import render_imagined  # noqa: E402

# Re-use the network classes from the trainer.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "train_dreamer", str(Path(__file__).parent / "train_dreamer.py")
)
_td = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_td)  # type: ignore[union-attr]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--prefix-steps", type=int, default=12,
                        help="real env steps per episode before the dream starts")
    parser.add_argument("--horizon", type=int, default=64,
                        help="imagined frames after the prefix")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--map-type", default="random",
                        choices=("random", "lake", "rocky", "balanced"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    tr_args = argparse.Namespace(**ckpt["args"])
    device = torch.device(args.device)

    # rebuild env to know image shape
    env = CognilandNavEnv(
        size=tr_args.env_size, map_type=args.map_type,
        view_size=tr_args.view_size, tile_px=tr_args.tile_px,
        obs_mode="rgb", seed=args.seed, max_steps=tr_args.max_steps,
    )
    image_shape = env.observation_space["image"].shape
    action_dim = 6

    enc = _td.Encoder(image_shape, embed_dim=tr_args.embed_dim).to(device)
    rssm = _td.RSSM(tr_args.embed_dim, action_dim,
                    deter=tr_args.deter, stoch_classes=tr_args.stoch_classes,
                    stoch_dim=tr_args.stoch_dim).to(device)
    dec = _td.Decoder(rssm.feat_dim, image_shape).to(device)
    actor = _td.Actor(rssm.feat_dim, num_moves=5).to(device)
    enc.load_state_dict(ckpt["enc"])
    rssm.load_state_dict(ckpt["rssm"])
    dec.load_state_dict(ckpt["dec"])
    actor.load_state_dict(ckpt["actor"])
    enc.eval(); rssm.eval(); dec.eval(); actor.eval()

    # Collect a few prefix episodes into a tiny replay-like buffer.
    replay = _td.EpisodeReplay(
        capacity=args.episodes * args.prefix_steps * 2 + 16,
        image_shape=image_shape, action_dim=action_dim, device=device,
    )
    for ep in range(args.episodes):
        obs, _ = env.reset()
        is_first = True
        for t in range(args.prefix_steps):
            # Use a uniform random policy for the prefix — gives the model
            # an honest "anchor"; the imagined rollout will follow the
            # actor from there.
            move = int(np.random.randint(0, 5))
            scalar = float(np.random.uniform(-1.0, 1.0))
            action_vec = _td.encode_action(move, scalar)
            action = {"move": move, "build_scalar": np.array([scalar], np.float32)}
            replay.add(obs["image"], float(obs["skill_active"][0]),
                       action_vec, 0.0, False, is_first)
            obs, *_ = env.step(action)
            is_first = False

    render_imagined(
        replay, enc, rssm, dec, actor, device,
        path=args.out, batch=args.episodes,
        prefix=args.prefix_steps, horizon=args.horizon, fps=args.fps,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
