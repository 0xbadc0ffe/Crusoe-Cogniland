"""Render Dreamer imagined trajectories as mp4 videos.

Given a trained Dreamer world model + policy, this module:

1. Samples a small batch of context frames from the replay buffer.
2. Encodes them through the encoder → posterior RSSM rollout to land in a
   latent state anchored to *real* observations.
3. Imagines forward for ``horizon`` steps using the actor + the prior
   dynamics (so the imagined frames are pure model generations).
4. Decodes every step back to an RGB frame and writes a grid video.

The video stacks ``batch`` rollouts in a 2-row grid:

  top row    = actual observations (the prefix that anchored the posterior)
  bottom row = imagined observations (model dream)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ACTION_DIM = 6


@torch.no_grad()
def render_imagined(replay, encoder, rssm, decoder, actor, device,
                    path: Path, batch: int = 4, prefix: int = 8,
                    horizon: int = 32, fps: int = 8) -> None:
    """Sample, anchor, imagine, decode, and write an mp4 to ``path``."""
    import imageio

    sample = replay.sample(batch, prefix + 1)
    if sample is None:
        print("[render_imagined] not enough data yet")
        return
    images = sample["image"]                                  # (T, B, C, H, W)
    actions = sample["action"]                                # (T, B) int
    is_first = sample["is_first"]
    B = images.shape[1]
    C, H, W = images.shape[2:]

    actions_oh = F.one_hot(actions.long(), ACTION_DIM).float()  # (T, B, A)

    # ── anchor: posterior rollout over the prefix
    state = rssm.initial(B, device)
    embed_seq = encoder(images.flatten(0, 1).to(device)).view(prefix + 1, B, -1)
    for t in range(prefix):
        mask = (1.0 - is_first[t].to(device)).view(B, 1)
        state = {k: (v * mask if v.dim() == 2 else v * mask.unsqueeze(-1))
                 for k, v in state.items()}
        prev_a = actions_oh[t].to(device) if t > 0 else torch.zeros_like(actions_oh[t]).to(device)
        _, state = rssm.obs_step(state, prev_a, embed_seq[t])

    real_frames = images[:prefix].cpu().numpy()              # (prefix, B, C, H, W)

    # ── imagine forward
    imagined_frames = []
    cur = state
    for _ in range(horizon):
        feat = rssm.feat(cur)
        logits = actor(feat)
        idx = logits.argmax(-1)
        a_vec = F.one_hot(idx, ACTION_DIM).float()
        cur = rssm.img_step(cur, a_vec)
        recon = decoder(rssm.feat(cur))                       # logits, centred around 0
        rgb = (recon + 0.5).clamp(0, 1)
        frame = (rgb * 255.0).byte().cpu().numpy()
        imagined_frames.append(frame)
    imagined = np.stack(imagined_frames)

    pad_len = max(0, horizon - prefix)
    if pad_len > 0:
        pad = np.repeat(real_frames[-1:], pad_len, axis=0)
        real_full = np.concatenate([real_frames, pad], axis=0)
    else:
        real_full = real_frames[:horizon]

    out_frames = []
    for t in range(horizon):
        top = np.concatenate([real_full[t, b].transpose(1, 2, 0) for b in range(B)], axis=1)
        bot = np.concatenate([imagined[t, b].transpose(1, 2, 0) for b in range(B)], axis=1)
        sep = np.full((4, top.shape[1], 3), 80, dtype=np.uint8)
        grid = np.concatenate([top, sep, bot], axis=0)
        out_frames.append(grid)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    for f in out_frames:
        writer.append_data(f)
    writer.close()
