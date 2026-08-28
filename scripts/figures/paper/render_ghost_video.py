#!/usr/bin/env python3
"""The video version of figure 7.5: 20 ghost agents playing one map at once.

One video per (agent, map type), nine in total. Each shows the full Crafter-
textured world with the twenty sampled episodes of figure 7.5 played
simultaneously as translucent ghosts, in the same per-episode colours.

Design notes:

* CAMERA. A square 32x32-tile window (the map is 32 rows by 64 columns, so the
  window is the full height and half the width) pans left to right at constant
  speed, arriving at the right edge as the episodes end. Panning is sub-tile:
  the world is blitted at a fractional offset, so the motion is smooth rather
  than stepping one tile at a time.

* ONE WORLD, MANY AGENTS. The twenty episodes each modify their own copy of the
  map, but the video shows a single world. Changed cells therefore take
  priority: as soon as ANY ghost bridges water or mines rock, that cell is drawn
  changed for the rest of the video. This is why the world only ever gets more
  open, and why a cell never flickers back.

* ANIMATION. Every tool event fires the pygame demo's own effect at the cell it
  changed -- yellow rubble for a mine, white sparkles for a build -- so tool use
  is visible rather than a silent tile swap.

* TIME. Episodes differ in length. Frames are interpolated between env steps, so
  ghosts glide; a ghost that has finished fades out and stops being drawn.

  PYTHONPATH=src python scripts/figures/paper/render_ghost_video.py
  ... --agents ppo --biomes lakes        # subset while iterating
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio.v2 as imageio
import numpy as np
import pygame
import matplotlib

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "figures"))

from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from paper_rollouts_textured import Effect, load_sprites  # noqa: E402

GH = REPO / "outputs/ghost_videos"
OUT = REPO / "paper/figures/belief_report/videos/ghosts"
SPRITE_DIR = REPO / "src/cogniland/assets/sprites"

BASE = {T.GRASS: "grass", T.WATER: "water", T.ROCK: "stone", T.WOOD: "path",
        T.TREE: "tree", T.SAND: "sand", T.DIRT: "path", T.TARGET: "grass"}
FACE_SPRITE = {0: "player-up", 1: "player-down", 2: "player-left", 3: "player-right"}
NAMES = ["grass", "water", "stone", "sand", "tree", "lava", "path", "flag",
         "diamond", "player", "player-up", "player-down", "player-left", "player-right"]

CAM = 32              # camera is a square window, in tiles
TP = 24               # pixels per tile  -> 768x768 video
SUB = 3               # rendered frames per env step
FPS = 24
TRAIL = 14            # ghost trail length, in env steps
CMAP = matplotlib.colormaps["turbo"]     # same colours as figure 7.5
AGENTS = ["ppo", "dreamer", "storm"]
BIOMES = ["lakes", "rocky", "balanced"]
LABEL = {"ppo": "PPO + GRU", "dreamer": "DreamerV3", "storm": "STORM"}


def ghost_colour(j, n):
    r, g, b, _ = CMAP(0.06 + 0.88 * j / max(n - 1, 1))
    return (int(r * 255), int(g * 255), int(b * 255))


def tint(sprite, colour, alpha):
    """Tint a sprite toward `colour` without washing it out.

    A plain additive fill drives every sprite to white, which is why the ghosts
    have to keep their own shading and take their identity from the disc drawn
    behind them instead."""
    s = sprite.copy()
    s.fill(colour + (255,), special_flags=pygame.BLEND_RGBA_MULT)
    s.set_alpha(alpha)
    return s


def disc(radius, colour, alpha):
    d = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(d, colour + (alpha,), (radius, radius), radius)
    return d


def build_world_timeline(rolls):
    """-> (events, n_steps). events[t] = list of (kind, r, c) applied at step t.

    Changed cells win over the base map, and the first change to a cell is the
    one that counts, so later duplicates by other ghosts are dropped."""
    n_steps = max(len(r["steps"]) for r in rolls)
    events = [[] for _ in range(n_steps + 1)]
    seen = set()
    for t in range(n_steps):
        for roll in rolls:
            if t >= len(roll["steps"]):
                continue
            ev = roll["steps"][t]["ev"]
            if ev and (ev["r"], ev["c"]) not in seen:
                seen.add((ev["r"], ev["c"]))
                events[t].append((ev["kind"], ev["r"], ev["c"]))
    return events, n_steps


def render(agent, biome, data, pool, sprites, args):
    d = data[biome]
    rec = pool[d["map_id"]]
    rolls = d["rollouts"]
    H, W = rec.terrain.shape
    events, n_steps = build_world_timeline(rolls)
    n_ghosts = len(rolls)
    colours = [ghost_colour(j, n_ghosts) for j in range(n_ghosts)]

    world = np.array(rec.terrain, dtype=np.int16)          # mutated as events fire
    # The camera tracks the PACK. Calibrating the pan to the longest episode
    # lets a single wandering ghost stretch the video (one lakes episode runs
    # 350 steps against a median of 98), so the pan and the video length follow
    # the 90th percentile and later stragglers are simply cut.
    lengths = sorted(len(r["steps"]) for r in rolls)
    # p75, not p90: two stragglers out of twenty are enough to drag p90 far past
    # the pack (DreamerV3 on lakes has a median of 115 steps and a p90 of 269),
    # which would stretch the video to watch one wandering ghost.
    n_video = int(np.percentile(lengths, 75)) + 12
    total_sub = n_video * SUB
    pan_span = max(W - CAM, 0)

    # Pan SPEED is set by the pack, not by the video length. The agents cross
    # all W columns while the camera only has W-CAM to travel, so a pan spread
    # over the whole episode is outrun and the ghosts leave the frame. Instead
    # the camera keeps the median ghost centred, at constant speed: fit the
    # window that the median column needs, then move linearly across it and
    # clamp at both ends.
    med = np.array([np.median([r["steps"][min(t, len(r["steps"]) - 1)]["c"]
                               for r in rolls]) for t in range(n_video)])
    ideal = np.clip(med - CAM / 2.0, 0, pan_span)
    moving = np.flatnonzero((ideal > 0) & (ideal < pan_span))
    if len(moving):
        t_start, t_end = int(moving[0]), int(moving[-1]) + 1
    else:
        t_start, t_end = 0, max(n_video - 1, 1)

    # pre-rendered per-ghost sprites, one per facing, tinted and translucent
    ghost_sprites = [{f: tint(sprites[FACE_SPRITE[f]], colours[j], 235) for f in range(4)}
                     for j in range(n_ghosts)]

    surf = pygame.Surface((CAM * TP, CAM * TP))
    frames, effects = [], []
    rng = np.random.default_rng(0)

    for fi in range(total_sub):
        t = min(fi // SUB, n_steps - 1)
        frac = (fi % SUB) / SUB

        for kind, r, c in events[t] if fi % SUB == 0 and t < len(events) else []:
            world[r, c] = T.WOOD if kind == "build" else T.GRASS
            effects.append(Effect((r, c), kind, rng))

        # camera: constant speed over [t_start, t_end], clamped outside
        tt = fi / SUB
        cam_x = pan_span * np.clip((tt - t_start) / max(t_end - t_start, 1), 0.0, 1.0)
        x0 = int(cam_x * TP)

        # ---- terrain ----
        surf.fill((8, 10, 8))
        c_lo = max(int(cam_x) - 1, 0)
        c_hi = min(int(cam_x) + CAM + 2, W)
        for r in range(H):
            for c in range(c_lo, c_hi):
                px = c * TP - x0
                if -TP < px < CAM * TP:
                    surf.blit(sprites[BASE.get(int(world[r, c]), "grass")], (px, r * TP))
        # doors
        for cells, name in ((rec.top_goal_cells, "top"), (rec.bottom_goal_cells, "bottom")):
            good = rec.correct_target in ("either", name)
            for (r, c) in cells:
                px = c * TP - x0
                if -TP < px < CAM * TP:
                    surf.blit(sprites["flag"], (px, r * TP))
                    pygame.draw.rect(surf, (34, 197, 94) if good else (239, 68, 68),
                                     (px, r * TP, TP, TP), 2)

        # ---- ghosts ----
        for j, roll in enumerate(rolls):
            steps = roll["steps"]
            if t >= len(steps) - 1:
                continue                                  # finished: stop drawing
            s0, s1 = steps[t], steps[min(t + 1, len(steps) - 1)]
            gr = s0["r"] + (s1["r"] - s0["r"]) * frac
            gc = s0["c"] + (s1["c"] - s0["c"]) * frac
            # trail
            for k in range(max(t - TRAIL, 0), t):
                a = (k - (t - TRAIL)) / max(TRAIL, 1)
                px = steps[k]["c"] * TP - x0 + TP // 2
                if -TP < px < CAM * TP:
                    rad = max(1, int(TP * 0.16 * a))
                    dot = pygame.Surface((rad * 2, rad * 2), pygame.SRCALPHA)
                    pygame.draw.circle(dot, colours[j] + (int(150 * a),), (rad, rad), rad)
                    surf.blit(dot, (px - rad, steps[k]["r"] * TP + TP // 2 - rad))
            px = gc * TP - x0
            if -TP < px < CAM * TP:
                rad = int(TP * 0.44)
                surf.blit(disc(rad, colours[j], 150),
                          (int(px + TP / 2 - rad), int(gr * TP + TP / 2 - rad)))
                surf.blit(ghost_sprites[j][int(s0["facing"])], (int(px), int(gr * TP)))

        # ---- tool animations ----
        for e in list(effects):
            px = e.cell[1] * TP - x0 + TP // 2
            if -TP < px < CAM * TP:
                e.draw(surf, int(px), e.cell[0] * TP + TP // 2, TP)
            e.t += 1
            if not e.alive():
                effects.remove(e)

        frames.append(np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)).copy())

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"ghosts_{agent}_{biome}_map{d['map_id']}.mp4"
    imageio.mimwrite(p, frames, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    n_ev = sum(len(e) for e in events)
    print(f"  {agent:8s} {biome:9s} map {d['map_id']:4d}: {len(frames)} frames, "
          f"{n_steps} steps, {n_ev} world changes -> {p.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default=",".join(AGENTS))
    ap.add_argument("--biomes", default=",".join(BIOMES))
    a = ap.parse_args()

    pygame.init()
    pygame.display.set_mode((1, 1))
    sprites = load_sprites(TP)
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))

    for agent in a.agents.split(","):
        f = GH / f"ghost_{agent}.json"
        if not f.exists():
            print(f"  {agent}: no {f.name} yet, skipping")
            continue
        data = json.loads(f.read_text())
        for biome in a.biomes.split(","):
            render(agent, biome, data, pool, sprites, a)


if __name__ == "__main__":
    main()
