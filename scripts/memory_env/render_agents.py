#!/usr/bin/env python
"""Render videos of trained R2-Dreamer MemoryEnv agents acting greedily.

For each provided checkpoint (2cue/3cue/4cue) we roll out one greedy episode on
each of the 4 cue types and write ONE mp4 per model. Episode seeds are shared
across models, so all three face identical layouts -> directly comparable. Each
frame is ``[ full top-down grid | agent POV ]`` with a header strip showing
model / cue / step / cumulative return / outcome.

The policy needs a GPU (reuses eval_r2dreamer.build_act_fn, which carries RSSM
state per episode and resets on is_first); env rendering is headless via
MiniGrid get_frame.

    PYTHONPATH=src python scripts/memory_env/render_agents.py \
        --ckpt-2cue r2dreamer_model/runs/memory_2cue/latest.pt \
        --ckpt-3cue r2dreamer_model/runs/memory_3cue/latest.pt \
        --ckpt-4cue r2dreamer_model/runs/memory_4cue/latest.pt \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "external" / "r2dreamer"))
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))

from cogniland.memory_env import MemoryEnv, MemoryEnvConfig  # noqa: E402
from datasets import ALL_CUES  # noqa: E402
from eval_r2dreamer import build_act_fn  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

MODELS = ["2cue", "3cue", "4cue"]
OUTDIR = _REPO / "outputs" / "report" / "videos"


def _upscale(img, k):
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def _header(width, lines, height=46):
    img = Image.new("RGB", (width, height), (18, 18, 18))
    d = ImageDraw.Draw(img)
    d.text((6, 5), lines[0], fill=(255, 255, 255))
    if len(lines) > 1:
        d.text((6, 25), lines[1], fill=(180, 180, 180))
    return np.asarray(img)


def composite(env, obs, header_lines, tile_size):
    """[full grid | gap | upscaled agent POV] with a header strip on top."""
    grid = np.asarray(env._mg.get_frame(tile_size=tile_size, agent_pov=False, highlight=True))
    hg = grid.shape[0]
    k = max(1, hg // obs.shape[0])
    pov = _upscale(obs, k)
    if pov.shape[0] < hg:
        pov = np.vstack([pov, np.zeros((hg - pov.shape[0], pov.shape[1], 3), np.uint8)])
    elif pov.shape[0] > hg:
        pov = pov[:hg]
    gap = np.full((hg, 8, 3), 40, np.uint8)
    body = np.hstack([grid, gap, pov])
    return np.vstack([_header(body.shape[1], header_lines), body])


def _normalize(frames):
    """Pad all frames to a common size rounded up to a multiple of 16 (libx264)."""
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)
    H = (h + 15) // 16 * 16
    W = (w + 15) // 16 * 16
    out = []
    for f in frames:
        c = np.zeros((H, W, 3), np.uint8)
        c[: f.shape[0], : f.shape[1]] = f
        out.append(c)
    return out


def rollout(act_fn, cue, seed, tile_size, max_steps, model_name):
    c = MemoryEnvConfig(cue_distribution="custom", custom_cues=[cue])
    env = MemoryEnv(c)
    obs, info = env.reset(seed=seed)
    cum, t, done, success = 0.0, 0, False, False

    def hdr():
        l1 = (f"{model_name} model    cue={cue}    "
              f"(correct: branch={info.get('correct_branch')} door={info.get('target_door_color')})")
        tail = ("DONE - " + ("SUCCESS" if success else "fail")) if done else ""
        return [l1, f"t={t:>3}   return={cum:+.2f}   {tail}"]

    frames = [composite(env, obs, hdr(), tile_size)]
    while not done and t < max_steps:
        a = act_fn(obs, info)
        obs, r, term, trunc, info = env.step(a)
        cum += r
        t += 1
        done = bool(term or trunc)
        success = bool(info.get("success", False))
        frames.append(composite(env, obs, hdr(), tile_size))
    frames += [frames[-1]] * 10  # hold the outcome
    return frames, cum, success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-2cue")
    ap.add_argument("--ckpt-3cue")
    ap.add_argument("--ckpt-4cue")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-size", default="size25M")
    ap.add_argument("--cues", nargs="+", default=list(ALL_CUES))
    ap.add_argument("--episodes-per-cue", type=int, default=1)
    ap.add_argument("--tile-size", type=int, default=24)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpts = {"2cue": args.ckpt_2cue, "3cue": args.ckpt_3cue, "4cue": args.ckpt_4cue}

    written = []
    for model in MODELS:
        ck = ckpts[model]
        if not ck:
            continue
        print(f"== {model}: building act_fn from {ck}", flush=True)
        act_fn = build_act_fn(ck, model, device=args.device, model_size=args.model_size)
        allf = []
        for ci, cue in enumerate(args.cues):
            for ep in range(args.episodes_per_cue):
                seed = args.seed_base + ci * 1000 + ep
                fr, cum, succ = rollout(act_fn, cue, seed, args.tile_size, args.max_steps, model)
                print(f"   {cue} ep{ep} seed={seed}: return={cum:+.2f} success={succ} frames={len(fr)}",
                      flush=True)
                allf.extend(fr)
        out = outdir / f"memoryenv_{model}_play.mp4"
        imageio.mimwrite(out, _normalize(allf), fps=args.fps, codec="libx264",
                         quality=8, macro_block_size=1)
        print(f"   wrote {out} ({len(allf)} frames)", flush=True)
        written.append(str(out))

    print("VIDEOS:", *written, sep="\n", flush=True)


if __name__ == "__main__":
    main()
