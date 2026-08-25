#!/usr/bin/env python3
"""Textured rollout videos: what the agent *actually sees*, in Crafter sprites.

Renders each episode as two synced panels:

  left   the agent's OBSERVATION -- the 21x21 egocentric crop, drawn with the
         Crafter sprite set (the same tiles//sprites the pygame demo uses), with
         out-of-bounds cells blacked out. This is the agent's entire input.
  right  the world map for the reader's orientation, with the trajectory trail
         and the two doors marked. The agent never sees this panel.

Mining and building are animated in the observation panel with the demo's own
`Effect` class (mine = yellow rubble burst, build = white sparkles), so tool use
is visible rather than a silent tile swap.

Runs headless (SDL dummy driver). Same agent adapters and map ids as
paper_rollouts.py, so the textured videos line up with the flat ones.

  PYTHONPATH=src python scripts/figures/paper_rollouts_textured.py --agent ppo
  PYTHONPATH=src:r2dreamer_model  ... --agent dreamer
  (from STORM_model/)  PYTHONPATH=.:..:../src python ../scripts/figures/paper_rollouts_textured.py --agent storm
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "figures"))

from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelEnv  # noqa: E402
from paper_rollouts import (  # noqa: E402
    FORKWALL_KWARGS, make_dreamer, make_ppo, make_storm,
)

SPRITE_DIR = REPO / "src/cogniland/assets/sprites"
# tile id -> sprite, matching scripts/bridge_tunnel/play_bridge_tunnel.py
BASE = {T.GRASS: "grass", T.WATER: "water", T.ROCK: "stone", T.WOOD: "path",
        T.TREE: "tree", T.SAND: "sand", T.DIRT: "path", T.TARGET: "grass"}
OVERLAY = {T.TARGET: "flag"}
FACE_SPRITE = {0: "player-up", 1: "player-down", 2: "player-left", 3: "player-right"}
A_BUILD, A_MINE = 4, 5
NAMES = ["grass", "water", "stone", "sand", "tree", "lava", "path", "flag",
         "diamond", "player", "player-up", "player-down", "player-left", "player-right"]

TP = 22          # px per tile in the observation panel
CELL = 7         # px per tile in the world panel
HUD_H = 46
PAD = 14
BG = (17, 20, 16)
INK = (232, 236, 226)
DIM = (140, 152, 138)


class Effect:
    """Cell animation, ported from the pygame demo: mine = rubble, build = sparkles."""
    DUR = 8

    def __init__(self, cell, kind, rng):
        self.cell, self.kind, self.t = cell, kind, 0
        n = 14 if kind == "build" else 7
        ang = rng.uniform(0, 2 * math.pi, n)
        rad = rng.uniform(0.25, 1.0, n) ** 0.5
        self.dx, self.dy = np.cos(ang) * rad, np.sin(ang) * rad
        self.delay = rng.uniform(0.0, 0.35, n)

    def alive(self):
        return self.t < self.DUR

    def draw(self, surf, cx, cy, tp):
        f = self.t / self.DUR
        if self.kind == "mine":
            for k in range(len(self.dx)):
                d = f * tp * 0.7
                s = max(1, int(tp * 0.18 * (1 - f)))
                pygame.draw.rect(surf, (235, 215, 70),
                                 (int(cx + self.dx[k] * d) - s,
                                  int(cy + self.dy[k] * d) - s, 2 * s, 2 * s))
        else:
            for k in range(len(self.dx)):
                ff = min(1.0, max(0.0, f - self.delay[k]) / max(1e-3, 1 - self.delay[k]))
                if ff <= 0:
                    continue
                d = ff * tp * 0.85
                px, py = int(cx + self.dx[k] * d), int(cy + self.dy[k] * d)
                b = int(255 * (1 - ff))
                s = max(1, int(tp * 0.08 * (1 - ff) + 1))
                pygame.draw.rect(surf, (255, 255, 255), (px - s, py - s, 2 * s, 2 * s))
                if b > 60:
                    pygame.draw.line(surf, (b, b, b), (px - s - 1, py), (px + s + 1, py))


def load_sprites(tp):
    return {n: pygame.transform.scale(
        pygame.image.load(str(SPRITE_DIR / f"{n}.png")).convert_alpha(), (tp, tp))
        for n in NAMES}


def draw_obs(surf, ox, oy, crop, facing, sprites, effects, agent_rc, view):
    """Left panel: the egocentric crop in sprites, agent fixed at centre."""
    half = view // 2
    for r in range(view):
        for c in range(view):
            t = int(crop[r, c])
            x, y = ox + c * TP, oy + r * TP
            if t == T.OOB:
                pygame.draw.rect(surf, (8, 10, 8), (x, y, TP, TP))
                continue
            surf.blit(sprites[BASE.get(t, "grass")], (x, y))
            if t in OVERLAY:
                surf.blit(sprites[OVERLAY[t]], (x, y))
    # tool animations, mapped from world coords into the crop
    ar, ac = agent_rc
    for e in effects:
        er, ec = e.cell
        rr, cc = er - ar + half, ec - ac + half
        if 0 <= rr < view and 0 <= cc < view:
            e.draw(surf, ox + cc * TP + TP // 2, oy + rr * TP + TP // 2, TP)
    surf.blit(sprites[FACE_SPRITE[facing]], (ox + half * TP, oy + half * TP))
    pygame.draw.rect(surf, (60, 68, 58), (ox - 1, oy - 1, view * TP + 2, view * TP + 2), 1)


def draw_world(surf, wx, wy, env, rec, traj):
    """Right panel: the true map (reader-only) with trail, doors and view box."""
    img = T.TILE_COLORS[env._terrain]
    s = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    H, W = env._terrain.shape
    s = pygame.transform.scale(s, (W * CELL, H * CELL))
    surf.blit(s, (wx, wy))
    for cells, name in ((rec.top_goal_cells, "top"), (rec.bottom_goal_cells, "bottom")):
        good = rec.correct_target in ("either", name)
        for (r, c) in cells:
            pygame.draw.rect(surf, (34, 197, 94) if good else (239, 68, 68),
                             (wx + c * CELL - 1, wy + r * CELL - 1, CELL + 2, CELL + 2), 2)
    for i, (r, c) in enumerate(traj[-160:]):
        a = 0.25 + 0.75 * (i + 1) / min(len(traj), 160)
        pygame.draw.rect(surf, (int(255 * a), int(255 * a), int(255 * a)),
                         (wx + c * CELL + CELL // 3, wy + r * CELL + CELL // 3,
                          max(1, CELL // 3), max(1, CELL // 3)))
    ar, ac = env._pos
    half = env.view_size // 2
    r0, c0 = max(0, ar - half), max(0, ac - half)
    r1, c1 = min(H, ar + half + 1), min(W, ac + half + 1)
    pygame.draw.rect(surf, (250, 220, 60),
                     (wx + c0 * CELL, wy + r0 * CELL,
                      (c1 - c0) * CELL, (r1 - r0) * CELL), 1)
    pygame.draw.rect(surf, (255, 255, 255), (wx + ac * CELL, wy + ar * CELL, CELL, CELL))
    pygame.draw.rect(surf, (60, 68, 58), (wx - 1, wy - 1, W * CELL + 2, H * CELL + 2), 1)


def rollout(act, reset, rec, agent_name, out_mp4, fps=16, hold=0.9, seed=0):
    # All three agents sample, so without an explicit seed these clips drift on
    # every regeneration and the captions in the report stop matching them.
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset()
    reset()
    view = env.view_size
    sprites = load_sprites(TP)
    fnt = pygame.font.SysFont("dejavusans", 15)
    fnt_s = pygame.font.SysFont("dejavusans", 12)
    H, W = env._terrain.shape

    ow, oh = view * TP, view * TP
    ww, wh = W * CELL, H * CELL
    sw = PAD * 3 + ow + ww
    sh = max(oh, wh) + HUD_H + PAD * 2
    surf = pygame.Surface((sw, sh))
    rng = np.random.default_rng(seed)

    frames, traj, effects, ret = [], [env._pos], [], 0.0
    n_build = n_mine = 0

    def compose(step):
        surf.fill(BG)
        draw_obs(surf, PAD, PAD + HUD_H, np.asarray(obs["minimap"]), env._facing,
                 sprites, effects, env._pos, view)
        draw_world(surf, PAD * 2 + ow, PAD + HUD_H, env, rec, traj)
        surf.blit(fnt.render(
            f"{agent_name}   category: {rec.category}   rewarded door: {rec.correct_target}",
            True, INK), (PAD, PAD - 2))
        surf.blit(fnt_s.render(
            f"step {step}    return {ret:+.2f}    built {n_build}    mined {n_mine}",
            True, DIM), (PAD, PAD + 19))
        cap_y = PAD + HUD_H + max(oh, wh) + 3
        surf.blit(fnt_s.render("agent observation (21x21, Crafter tiles)", True, DIM),
                  (PAD, cap_y))
        surf.blit(fnt_s.render("world state - not observed by the agent", True, DIM),
                  (PAD * 2 + ow, cap_y))
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)).copy()

    for t in range(FORKWALL_KWARGS["max_steps"]):
        frames.append(compose(t))
        a = act(obs, False)
        facing_before = env._facing
        obs, r, term, trunc, info = env.step(a)
        ret += float(r)
        traj.append(env._pos)

        if a in (A_BUILD, A_MINE) and (info.get("placed") or info.get("mined")):
            dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[facing_before]
            cell = (env._pos[0] + dr, env._pos[1] + dc)
            kind = "build" if info.get("placed") else "mine"
            effects.append(Effect(cell, kind, rng))
            n_build += int(kind == "build")
            n_mine += int(kind == "mine")

        # Age the bursts *alongside* the replay instead of pausing for them. An
        # earlier version blocked until each burst finished, which stalled the
        # playback for half a second on every tool use and read as lag. The dust
        # now settles at the cell it belongs to while the agent walks on.
        for e in effects:
            e.t += 1
        effects = [e for e in effects if e.alive()]

        if term or trunc:
            for _ in range(int(fps * hold)):
                frames.append(compose(t + 1))
                for e in effects:
                    e.t += 1
                effects = [e for e in effects if e.alive()]
            break

    success = env._pos in (env._correct_cells or set())
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_mp4, frames, fps=fps, codec="libx264",
                     output_params=["-pix_fmt", "yuv420p", "-crf", "26"],
                     macro_block_size=1)
    return dict(agent=agent_name, category=rec.category, success=bool(success),
                steps=len(traj) - 1, ret=round(ret, 3),
                builds=n_build, mines=n_mine, frames=len(frames))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, choices=["ppo", "dreamer", "storm"])
    p.add_argument("--seed", type=int, default=0,
                   help="policy seed; a timed-out clip retries with seed+1, +2, ...")
    p.add_argument("--maps", default=str(REPO / "data/bridge_tunnel/forkwall6k/test.pkl"))
    p.add_argument("--map-ids", default="0,5,7")
    p.add_argument("--out", default=str(REPO / "paper/figures/forkwall_paper"))
    p.add_argument("--ppo-ckpt", default=str(REPO / "final_models/ppo/ppo_plain.pt"))
    p.add_argument("--storm-bundle", default=str(REPO / "final_models/storm"))
    p.add_argument("--storm-step", type=int, default=624489)
    p.add_argument("--dreamer-ckpt", default=str(REPO / "final_models/dreamer/dreamer_25M_bl64.pt"))
    p.add_argument("--dreamer-size", default="size25M")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    pygame.init()
    pygame.display.set_mode((1, 1))
    out = Path(args.out)
    with open(args.maps, "rb") as f:
        pool = pickle.load(f)

    if args.agent == "ppo":
        act, reset = make_ppo(args.ppo_ckpt)
    elif args.agent == "storm":
        act, reset = make_storm(args.storm_bundle, args.storm_step)
    else:
        act, reset = make_dreamer(args.dreamer_ckpt, args.device, args.dreamer_size)

    rows = []
    for i in [int(x) for x in args.map_ids.split(",")]:
        rec = pool[i]
        mp4 = out / "videos_textured" / f"{args.agent}_obs_map{i}_{rec.category}.mp4"
        # A timed-out episode is 800 steps = 50 s of unwatchable video and never
        # illustrates anything. Re-roll it; genuine wrong-door failures are kept,
        # because those are the interesting ones.
        for sd in range(args.seed, args.seed + 12):
            row = rollout(act, reset, rec, args.agent.upper(), mp4, seed=sd)
            if row["steps"] < FORKWALL_KWARGS["max_steps"]:
                break
            print(f"   map {i}: timeout at seed {sd}, re-rolling")
        row["map_id"] = i
        row["seed"] = sd
        rows.append(row)
        print(f"map {i:5d} {rec.category:9s} ok={row['success']!s:5s} steps={row['steps']:3d} "
              f"build={row['builds']} mine={row['mines']} -> {mp4.name}")

    jf = out / f"rollouts_textured_{args.agent}.json"
    jf.write_text(json.dumps(rows, indent=1))
    print("wrote", jf)


if __name__ == "__main__":
    main()
