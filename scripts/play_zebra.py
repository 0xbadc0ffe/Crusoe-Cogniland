#!/usr/bin/env python3
"""Playable pygame demo for the zebra_nav env — human or trained AI.

Works with the current maps (diagonal / vertical / natural) and both action
modes (absolute 4-move, or relative turn/forward). Renders the full map (the
agent itself only sees the egocentric crop, drawn as a white box), the agent
with a facing arrow, and short **mining** (rock → rubble → grass) and
**bridge-building** (water → planks + ripple) animations.

Human controls
--------------
  absolute mode : ↑ ↓ ← →  move/face,  B = build (place on water),  M = mine
  relative mode : ← → turn,  ↑ forward,  B = build,  M = mine
  common        : R = new map,  Q/Esc = quit
AI controls (when --checkpoint is given)
  A = toggle AI auto-play,  Space = single AI step,  + / - = AI speed

    python scripts/play_zebra.py --orientation natural --env-width 64 --view-size 21
    python scripts/play_zebra.py --checkpoint checkpoints/zebra_agents/vertical_cuefollower.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.zebra_nav.env import (  # noqa: E402
    ZebraNavEnv, _FACE_DELTA, F_UP, F_DOWN, F_LEFT, F_RIGHT,
    A_UP, A_DOWN, A_LEFT, A_RIGHT, A_PLACE, A_MINE,
    R_TURN_LEFT, R_TURN_RIGHT, R_FORWARD, R_PLACE, R_MINE,
)
from cogniland.zebra_nav import tiles as T  # noqa: E402


def _facing_cell(env):
    dr, dc = _FACE_DELTA[env._facing]
    return env._pos[0] + dr, env._pos[1] + dc


class Effect:
    """A short cell animation (mining rubble or bridge planks + ripple)."""
    DUR = 12

    def __init__(self, cell, kind):
        self.cell, self.kind, self.t = cell, kind, 0

    def alive(self):
        return self.t < self.DUR

    def draw(self, surf, px):
        r, c = self.cell
        x, y, f = c * px, r * px, self.t / self.DUR
        cx, cy = x + px // 2, y + px // 2
        if self.kind == "mine":
            # grey rubble chunks flying outward, fading
            rng = np.random.default_rng(r * 131 + c)
            for k in range(6):
                ang = 2 * np.pi * k / 6 + rng.uniform(0, 1)
                d = f * px * 0.6
                rx, ry = int(cx + np.cos(ang) * d), int(cy + np.sin(ang) * d)
                s = max(1, int(px * 0.22 * (1 - f)))
                pygame.draw.rect(surf, (90, 90, 90), (rx - s, ry - s, 2 * s, 2 * s))
        else:  # place: expanding ripple ring + wood plank growing in
            ring = int(f * px * 0.7)
            if ring > 0:
                col = (200, 220, 255)
                pygame.draw.circle(surf, col, (cx, cy), ring, max(1, int(px * 0.08)))
            w = int(px * 0.7 * f)
            if w > 0:
                pygame.draw.rect(surf, (140, 90, 50),
                                 (cx - w // 2, cy - px // 4, w, px // 2))
        self.t += 1


def _draw(screen, env, px, effects, font, mode_txt, ai_on, status):
    H, W = env._terrain.shape
    # base tiles (effect cells are drawn by the effect instead)
    eff_cells = {e.cell for e in effects}
    img = T.TILE_COLORS[env._terrain]
    for r in range(H):
        for c in range(W):
            if (r, c) in eff_cells:
                pygame.draw.rect(screen, (110, 173, 86), (c * px, r * px, px, px))  # grass base
                continue
            screen.fill(tuple(int(v) for v in img[r, c]), (c * px, r * px, px, px))
    for e in effects:
        e.draw(screen, px)
    # egocentric view box
    v = env.view_size // 2
    ar, ac = env._pos
    pygame.draw.rect(screen, (255, 255, 255),
                     ((ac - v) * px, (ar - v) * px, env.view_size * px, env.view_size * px), 1)
    # agent + facing arrow
    cx, cy = ac * px + px // 2, ar * px + px // 2
    pygame.draw.circle(screen, (20, 20, 20), (cx, cy), max(2, px // 2 - 1))
    pygame.draw.circle(screen, (255, 255, 255), (cx, cy), max(2, px // 2 - 1), 1)
    dr, dc = _FACE_DELTA[env._facing]
    pygame.draw.line(screen, (255, 80, 80), (cx, cy),
                     (cx + dc * px // 2, cy + dr * px // 2), max(2, px // 5))
    if font:
        txt = f"{mode_txt} | {'AI' if ai_on else 'HUMAN'} | {status}"
        screen.blit(font.render(txt, True, (255, 255, 255), (0, 0, 0)), (4, 4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orientation", default="natural",
                   choices=("diagonal", "vertical", "natural"))
    p.add_argument("--env-size", type=int, default=32)
    p.add_argument("--env-width", type=int, default=64)
    p.add_argument("--view-size", type=int, default=21)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--action-mode", default="relative", choices=("absolute", "relative"))
    p.add_argument("--tile-px", type=int, default=16)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--maps", type=Path, default=None,
                   help="pickled validation-map set (from make_zebra_val_maps.py); the "
                        "demo cycles through these fixed maps instead of random ones")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # fixed validation/demo map set (so the demo plays exactly the validation maps)
    val_records = None
    if args.maps is not None:
        import pickle
        with open(args.maps, "rb") as f:
            blob = pickle.load(f)
        val_records = blob["records"]
        args.orientation = blob.get("orientation", args.orientation)
        H, W = val_records[0].terrain.shape
        args.env_size, args.env_width = H, W

    policy = device = None
    if args.checkpoint is not None:
        import torch
        from train_ppo_zebra import PPOGRUPolicy
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ca = ck["args"]
        args.orientation = ca.get("orientation", args.orientation)
        args.env_size = ca.get("env_size", args.env_size)
        args.env_width = ca.get("env_width") or args.env_size
        args.view_size = ca.get("view_size", args.view_size)
        args.action_mode = ca.get("action_mode", "absolute")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ZebraNavEnv(size=args.env_size, width=args.env_width, view_size=args.view_size,
                      orientation=args.orientation, max_steps=args.max_steps,
                      action_mode=args.action_mode, seed=args.seed)
    map_idx = [0]

    def _load_next_map():
        """If a fixed validation set is given, cycle through it; else random."""
        if val_records:
            env._fixed_record = val_records[map_idx[0] % len(val_records)]
            map_idx[0] += 1
        return env.reset()[0]

    obs = _load_next_map()

    if args.checkpoint is not None:
        import torch
        n_tiles = int(ck["policy"]["tile_embed.weight"].shape[0])
        n_act = int(ck["policy"]["actor.weight"].shape[0])
        policy = PPOGRUPolicy(env.observation_space, num_actions=n_act,
                              gru_hidden=ca.get("gru_hidden", 128),
                              embed_dim=ca.get("embed_dim", 256),
                              num_tile_classes=n_tiles).to(device)
        policy.load_state_dict(ck["policy"]); policy.eval()
        hidden = torch.zeros(1, 1, policy.gru_hidden, device=device)

    pygame.init()
    H, W = env._terrain.shape
    px = args.tile_px
    screen = pygame.display.set_mode((W * px, H * px))
    pygame.display.set_caption("zebra_nav")
    font = pygame.font.SysFont("monospace", max(10, px))
    clock = pygame.time.Clock()
    effects = []
    rel = args.action_mode == "relative"
    S = {"obs": obs, "status": "go!", "ai_on": policy is not None, "period": 6}
    if policy is not None:
        S["hidden"] = hidden
    KEYMAP_ABS = {pygame.K_UP: A_UP, pygame.K_DOWN: A_DOWN, pygame.K_LEFT: A_LEFT,
                  pygame.K_RIGHT: A_RIGHT, pygame.K_b: A_PLACE, pygame.K_m: A_MINE}
    KEYMAP_REL = {pygame.K_LEFT: R_TURN_LEFT, pygame.K_RIGHT: R_TURN_RIGHT,
                  pygame.K_UP: R_FORWARD, pygame.K_b: R_PLACE, pygame.K_m: R_MINE}

    def reset_episode():
        S["obs"] = _load_next_map()
        if policy is not None:
            import torch
            S["hidden"] = torch.zeros(1, 1, policy.gru_hidden, device=device)
        effects.clear()

    def do_action(a):
        fcell = _facing_cell(env)
        o, r, term, trunc, info = env.step(a)
        S["obs"] = o
        if info["mined"]:
            effects.append(Effect(fcell, "mine"))
        if info["placed"]:
            effects.append(Effect(fcell, "place"))
        S["status"] = f"step {info['step']}  ret {info['episode_return']:+.2f}"
        if term or trunc:
            S["status"] = ("REACHED!" if term else "timeout") + " — new map"
            reset_episode()

    def ai_step():
        import torch
        with torch.no_grad():
            mm = torch.from_numpy(S["obs"]["minimap"])[None][None].to(device)
            sc = torch.from_numpy(S["obs"]["scalars"])[None][None].to(device)
            gout, h = policy._gru_forward({"minimap": mm, "scalars": sc},
                                          torch.zeros(1, 1, device=device), S["hidden"])
            S["hidden"] = h
            logits, _ = policy._heads(gout.squeeze(0))
            a = int(torch.distributions.Categorical(logits=logits).sample())
        do_action(a)

    running, frame = True, 0
    while running:
        frame += 1
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_r:
                    reset_episode()
                elif ev.key == pygame.K_a and policy is not None:
                    S["ai_on"] = not S["ai_on"]
                elif ev.key == pygame.K_SPACE and policy is not None:
                    ai_step()
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    S["period"] = max(1, S["period"] - 1)
                elif ev.key == pygame.K_MINUS:
                    S["period"] += 1
                elif not (policy is not None and S["ai_on"]):
                    km = KEYMAP_REL if rel else KEYMAP_ABS
                    if ev.key in km:
                        do_action(km[ev.key])

        if policy is not None and S["ai_on"] and frame % S["period"] == 0:
            ai_step()

        screen.fill((0, 0, 0))
        effects[:] = [e for e in effects if e.alive()]
        _draw(screen, env, px, effects, font,
              f"{args.orientation}/{args.action_mode}", S["ai_on"], S["status"])
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
