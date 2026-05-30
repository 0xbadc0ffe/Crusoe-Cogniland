#!/usr/bin/env python3
"""Playable pygame demo for the zebra_nav env — human or trained AI.

Crafter-sprite egocentric main view (left) + small pixel minimap (right), in the
style of play_cogniland.py. Starts on a menu where you pick Human/AI and the map
(a curated validation map, or Random), then plays with mining / bridge-building
animations.

    python scripts/play_zebra.py            # uses models/zebra_nav/natural_agent.pt
                                            # + data/zebra_nav/val_maps.pkl if present
    python scripts/play_zebra.py --checkpoint models/zebra_nav/vertical_cuefollower.pt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.zebra_nav.env import (  # noqa: E402
    ZebraNavEnv, _FACE_DELTA, F_UP, F_DOWN, F_LEFT, F_RIGHT,
    A_UP, A_DOWN, A_LEFT, A_RIGHT, A_PLACE, A_MINE,
)
from cogniland.zebra_nav import generate_zebra_map, tiles as T  # noqa: E402

_SPRITE_DIR = Path(__file__).resolve().parents[1] / "src/cogniland/assets/sprites"
# zebra tile -> (base sprite, optional overlay sprite)
_BASE = {T.GRASS: "grass", T.WATER: "water", T.ROCK: "stone", T.WOOD: "path",
         T.OBSIDIAN: "lava", T.TREE: "tree", T.SAND: "sand", T.DIRT: "path",
         T.TARGET: "grass", T.CUE_WATER_THIN: "grass", T.CUE_ROCK_THIN: "grass"}
_OVERLAY = {T.TARGET: "flag", T.CUE_WATER_THIN: "diamond", T.CUE_ROCK_THIN: "diamond"}
_FACE_SPRITE = {F_UP: "player-up", F_DOWN: "player-down",
                F_LEFT: "player-left", F_RIGHT: "player-right"}
_PANEL_W = 280


def _load_sprites(tp):
    names = ["grass", "water", "stone", "sand", "tree", "lava", "path", "flag",
             "diamond", "player", "player-up", "player-down", "player-left", "player-right"]
    out = {}
    for n in names:
        img = pygame.image.load(str(_SPRITE_DIR / f"{n}.png")).convert_alpha()
        out[n] = pygame.transform.scale(img, (tp, tp))
    return out


class Effect:
    """Short cell animation in world coords: mine (yellow burst) / place (ripple)."""
    DUR = 11

    def __init__(self, cell, kind):
        self.cell, self.kind, self.t = cell, kind, 0

    def alive(self):
        return self.t < self.DUR

    def draw(self, surf, cx, cy, tp):
        f = self.t / self.DUR
        if self.kind == "mine":
            for k in range(6):
                a = 2 * np.pi * k / 6
                d = f * tp * 0.7
                s = max(1, int(tp * 0.16 * (1 - f)))
                pygame.draw.rect(surf, (235, 215, 70),
                                 (int(cx + np.cos(a) * d) - s, int(cy + np.sin(a) * d) - s, 2 * s, 2 * s))
        else:
            ring = int(f * tp * 0.8)
            if ring > 0:
                pygame.draw.circle(surf, (235, 70, 70), (cx, cy), ring, max(1, tp // 8))
        self.t += 1


def _draw_main(win, env, sprites, tp, effects):
    """Egocentric sprite view; player centred with facing sprite."""
    V = env.view_size
    crop = env._egocentric_crop()
    for vr in range(V):
        for vc in range(V):
            t = int(crop[vr, vc]); x, y = vc * tp, vr * tp
            if t == T.OOB:
                win.fill((0, 0, 0), (x, y, tp, tp)); continue
            win.blit(sprites[_BASE.get(t, "grass")], (x, y))
            if t in _OVERLAY:
                win.blit(sprites[_OVERLAY[t]], (x, y))
    win.blit(sprites[_FACE_SPRITE[env._facing]], (V // 2 * tp, V // 2 * tp))
    # animations (world cell → view cell relative to the centred agent)
    ar, ac = env._pos
    for e in effects:
        er, ec = e.cell
        vr, vc = er - ar + V // 2, ec - ac + V // 2
        if 0 <= vr < V and 0 <= vc < V:
            e.draw(win, vc * tp + tp // 2, vr * tp + tp // 2, tp)


def _draw_minimap(win, env, ox, oy, cell):
    H, W = env._terrain.shape
    img = T.TILE_COLORS[env._terrain]
    surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    surf = pygame.transform.scale(surf, (W * cell, H * cell))
    pygame.draw.rect(win, (70, 70, 80), (ox - 2, oy - 2, W * cell + 4, H * cell + 4), 1)
    win.blit(surf, (ox, oy))
    ar, ac = env._pos
    pygame.draw.rect(win, (255, 255, 255), (ox + ac * cell - 1, oy + ar * cell - 1, cell + 2, cell + 2))
    v = env.view_size // 2
    pygame.draw.rect(win, (255, 255, 0),
                     (ox + (ac - v) * cell, oy + (ar - v) * cell, env.view_size * cell, env.view_size * cell), 1)
    return H * cell


def _text(win, font, lines, x, y, color=(235, 235, 235)):
    for i, ln in enumerate(lines):
        win.blit(font.render(ln, True, color), (x, y + i * (font.get_height() + 2)))


# ───────────────────────────── AI helper ──────────────────────────────────

def _ai_action(policy, obs, state, device, deterministic=False):
    import torch
    with torch.no_grad():
        mm = torch.from_numpy(obs["minimap"])[None][None].to(device)
        sc = torch.from_numpy(obs["scalars"])[None][None].to(device)
        gout, h = policy._gru_forward({"minimap": mm, "scalars": sc},
                                      torch.zeros(1, 1, device=device), state["hidden"])
        state["hidden"] = h
        logits, _ = policy._heads(gout.squeeze(0))
        if deterministic:
            return int(torch.argmax(logits, -1))
        return int(torch.distributions.Categorical(logits=logits).sample())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("models/zebra_nav/natural_agent.pt"))
    p.add_argument("--maps", type=Path, default=Path("data/zebra_nav/val_maps.pkl"))
    p.add_argument("--orientation", default="natural", choices=("diagonal", "vertical", "natural"))
    p.add_argument("--env-size", type=int, default=32)
    p.add_argument("--env-width", type=int, default=64)
    p.add_argument("--view-size", type=int, default=21)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--main-px", type=int, default=600, help="target pixel size of the main view")
    args = p.parse_args()

    # optional checkpoint (AI) — its config drives env shape / orientation / actions
    policy = device = ca = None
    if args.checkpoint and args.checkpoint.exists():
        import torch
        from train_ppo_zebra import PPOGRUPolicy
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ca = ck["args"]
        args.orientation = ca.get("orientation", args.orientation)
        args.env_size = ca.get("env_size", args.env_size)
        args.env_width = ca.get("env_width") or args.env_size
        args.view_size = ca.get("view_size", args.view_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # optional fixed validation map set
    val = None
    if args.maps and args.maps.exists():
        with open(args.maps, "rb") as f:
            blob = pickle.load(f)
        val = blob["records"]
        args.orientation = blob.get("orientation", args.orientation)
        args.env_size, args.env_width = val[0].terrain.shape

    env = ZebraNavEnv(size=args.env_size, width=args.env_width, view_size=args.view_size,
                      orientation=args.orientation, max_steps=args.max_steps)

    if policy is None and args.checkpoint and args.checkpoint.exists():
        import torch
        n_tiles = int(ck["policy"]["tile_embed.weight"].shape[0])
        n_act = int(ck["policy"]["actor.weight"].shape[0])
        policy = PPOGRUPolicy(env.observation_space, num_actions=n_act,
                              gru_hidden=ca.get("gru_hidden", 128),
                              embed_dim=ca.get("embed_dim", 256),
                              num_tile_classes=n_tiles).to(device)
        policy.load_state_dict(ck["policy"]); policy.eval()

    pygame.init()
    tp = max(14, args.main_px // args.view_size)            # main-view tile px
    main_px = tp * args.view_size
    cell = max(3, min(6, (main_px - 40) // max(env._terrain.shape)))  # minimap cell px
    W = main_px + _PANEL_W
    Hpx = max(main_px, env._terrain.shape[0] * cell + 220)
    screen = pygame.display.set_mode((W, Hpx))
    pygame.display.set_caption("zebra_nav")
    sprites = _load_sprites(tp)
    font = pygame.font.SysFont("monospace", max(13, tp // 2))
    big = pygame.font.SysFont("monospace", max(20, tp), bold=True)
    clock = pygame.time.Clock()

    KEYMAP = {pygame.K_UP: A_UP, pygame.K_DOWN: A_DOWN, pygame.K_LEFT: A_LEFT,
              pygame.K_RIGHT: A_RIGHT, pygame.K_b: A_PLACE, pygame.K_m: A_MINE}

    # menu options
    n_maps = len(val) if val else 0
    map_choices = ([f"validation #{i}" for i in range(n_maps)] + ["random"]) if val else ["random"]
    menu = {"mode": 1 if policy is not None else 0,   # 0 Human, 1 AI
            "map": 0, "row": 0}
    MODE_LABELS = ["Human", "AI" + ("" if policy else " (no checkpoint)")]
    rows = ["Mode", "Map", "▶ Start"]
    state = {"screen": "menu", "obs": None, "hidden": None, "status": "", "effects": [],
             "ai_play": True, "period": 6}

    def start_episode():
        if val and menu["map"] < n_maps:
            env._fixed_record = val[menu["map"]]
        else:
            env._fixed_record = None
        state["obs"], _ = env.reset()
        if policy is not None:
            import torch
            state["hidden"] = torch.zeros(1, 1, policy.gru_hidden, device=device)
        state["effects"] = []
        state["status"] = "playing"
        state["screen"] = "play"

    def do_action(a):
        fr, fc = env._pos[0] + _FACE_DELTA[env._facing][0], env._pos[1] + _FACE_DELTA[env._facing][1]
        o, r, term, trunc, info = env.step(a)
        state["obs"] = o
        if info["mined"]:
            state["effects"].append(Effect((fr, fc), "mine"))
        if info["placed"]:
            state["effects"].append(Effect((fr, fc), "place"))
        state["status"] = f"step {info['step']}  return {info['episode_return']:+.2f}"
        if term or trunc:
            state["status"] = "REACHED! → menu" if term else "timeout → menu"
            state["screen"] = "menu"

    is_ai = lambda: policy is not None and menu["mode"] == 1
    frame = 0
    running = True
    while running:
        frame += 1
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    if state["screen"] == "play":
                        state["screen"] = "menu"
                    else:
                        running = False
                elif state["screen"] == "menu":
                    if ev.key in (pygame.K_UP, pygame.K_w):
                        menu["row"] = (menu["row"] - 1) % len(rows)
                    elif ev.key in (pygame.K_DOWN, pygame.K_s):
                        menu["row"] = (menu["row"] + 1) % len(rows)
                    elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        d = 1 if ev.key == pygame.K_RIGHT else -1
                        if menu["row"] == 0 and policy is not None:
                            menu["mode"] = (menu["mode"] + d) % 2
                        elif menu["row"] == 1:
                            menu["map"] = (menu["map"] + d) % len(map_choices)
                    elif ev.key == pygame.K_RETURN:
                        if menu["row"] == 2:
                            start_episode()
                        elif menu["row"] == 1:
                            menu["map"] = (menu["map"] + 1) % len(map_choices)
                        elif menu["row"] == 0 and policy is not None:
                            menu["mode"] = (menu["mode"] + 1) % 2
                else:  # play screen
                    if ev.key == pygame.K_r:
                        start_episode()
                    elif ev.key == pygame.K_a and policy is not None:
                        state["ai_play"] = not state["ai_play"]
                    elif ev.key == pygame.K_SPACE and is_ai():
                        do_action(_ai_action(policy, state["obs"], state, device))
                    elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        state["period"] = max(1, state["period"] - 1)
                    elif ev.key == pygame.K_MINUS:
                        state["period"] += 1
                    elif not is_ai() and ev.key in KEYMAP:
                        do_action(KEYMAP[ev.key])

        if state["screen"] == "play" and is_ai() and state["ai_play"] and frame % state["period"] == 0:
            do_action(_ai_action(policy, state["obs"], state, device))

        screen.fill((18, 18, 22))
        if state["screen"] == "menu":
            _text(screen, big, ["zebra_nav"], 30, 30, (250, 230, 90))
            opt = [f"Mode:  < {MODE_LABELS[menu['mode']]} >",
                   f"Map:   < {map_choices[menu['map']]} >",
                   "▶  Start  (Enter)"]
            for i, line in enumerate(opt):
                col = (255, 255, 120) if i == menu["row"] else (210, 210, 210)
                _text(screen, big, [line], 40, 110 + i * 56, col)
            _text(screen, font, [
                "↑/↓ select   ←/→ change   Enter confirm",
                "",
                "In game:  arrows move,  B build,  M mine",
                "          A toggle AI,  Space step,  +/- speed,  R new,  Esc menu",
                "",
                f"agent: {args.checkpoint.name if (policy is not None) else '(none — human only)'}",
                f"task : {args.orientation}",
            ], 40, 320, (180, 180, 190))
        else:
            _draw_main(screen, env, sprites, tp, state["effects"])
            mmx, mmy = main_px + 16, 16
            mmh = _draw_minimap(screen, env, mmx, mmy, cell)
            _text(screen, font, [
                f"{'AI' if is_ai() else 'HUMAN'}{' ▮▮' if (is_ai() and not state['ai_play']) else ''}",
                f"map: {map_choices[menu['map']]}",
                state["status"],
            ], mmx, mmy + mmh + 14, (235, 235, 235))
            _text(screen, font, [
                "B build  M mine",
                "arrows move",
                "A=AI Space=step",
                "+/- speed",
                "R new  Esc menu",
            ], mmx, mmy + mmh + 14 + 4 * (font.get_height() + 2) + 10, (170, 170, 180))
            state["effects"][:] = [e for e in state["effects"] if e.alive()]
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
