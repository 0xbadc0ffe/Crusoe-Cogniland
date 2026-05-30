#!/usr/bin/env python3
"""Playable pygame demo for the zebra_nav env — human or trained AI.

A small state-machine app (in the style of play_cogniland.py):

    MAIN_MENU ──┬─▶ PICK_MAP ─▶ PLAY            (Human)
                └─▶ PICK_AGENT ─▶ PICK_MAP ─▶ PLAY   (AI)

* MAIN_MENU  — Play as Human / Play as AI.
* PICK_AGENT — scrollable list of models/zebra_nav/*.pt (orientation shown).
* PICK_MAP   — thumbnail grid of maps for the chosen agent's orientation
               (the curated validation maps for natural), plus a Random tile.
* PLAY       — large egocentric Crafter-sprite view + small pixel minimap, with
               mining / bridge-building animations.

    python scripts/play_zebra.py
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.zebra_nav.env import (  # noqa: E402
    ZebraNavEnv, _FACE_DELTA, A_UP, A_DOWN, A_LEFT, A_RIGHT, A_PLACE, A_MINE,
    F_UP, F_DOWN, F_LEFT, F_RIGHT,
)
from cogniland.zebra_nav import generate_zebra_map, tiles as T  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_SPRITE_DIR = _REPO / "src/cogniland/assets/sprites"
_MODELS_DIR = _REPO / "models/zebra_nav"
_VAL_MAPS = _REPO / "data/zebra_nav/val_maps.pkl"

_BASE = {T.GRASS: "grass", T.WATER: "water", T.ROCK: "stone", T.WOOD: "path",
         T.OBSIDIAN: "lava", T.TREE: "tree", T.SAND: "sand", T.DIRT: "path",
         T.TARGET: "grass", T.CUE_WATER_THIN: "grass", T.CUE_ROCK_THIN: "grass"}
_OVERLAY = {T.TARGET: "flag", T.CUE_WATER_THIN: "diamond", T.CUE_ROCK_THIN: "diamond"}
_FACE_SPRITE = {F_UP: "player-up", F_DOWN: "player-down", F_LEFT: "player-left", F_RIGHT: "player-right"}
_BG = (18, 22, 30)
# generate_zebra_map kwargs we may pull from an agent's stored args
_GEN_KEYS = ("n_stripes", "thick_half", "thin_half", "obsidian_half", "window_h",
             "orientation", "water_frac", "rock_frac", "tree_frac", "goal_half")


def _load_sprites(tp):
    names = ["grass", "water", "stone", "sand", "tree", "lava", "path", "flag",
             "diamond", "player", "player-up", "player-down", "player-left", "player-right"]
    return {n: pygame.transform.scale(
        pygame.image.load(str(_SPRITE_DIR / f"{n}.png")).convert_alpha(), (tp, tp)) for n in names}


def _terrain_surface(terrain, cell):
    """Pixel-art surface of a full map from TILE_COLORS (for minimap / thumbnails)."""
    img = T.TILE_COLORS[terrain]
    s = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    H, W = terrain.shape
    return pygame.transform.scale(s, (W * cell, H * cell))


# ───────────────────────────── animations ─────────────────────────────────

class Effect:
    """Short cell animation (world coords). MINE = yellow rubble burst,
    BUILD = diffuse white sparkles. Fast (~8 frames)."""
    DUR = 8

    def __init__(self, cell, kind, rng):
        self.cell, self.kind, self.t = cell, kind, 0
        n = 14 if kind == "build" else 7
        ang = rng.uniform(0, 2 * math.pi, n)
        rad = rng.uniform(0.25, 1.0, n) ** 0.5      # spread outward, biased wide
        self.dx, self.dy = np.cos(ang) * rad, np.sin(ang) * rad
        self.delay = rng.uniform(0.0, 0.35, n)       # staggered twinkle (build)

    def alive(self):
        return self.t < self.DUR

    def draw(self, surf, cx, cy, tp):
        f = self.t / self.DUR
        if self.kind == "mine":
            for k in range(len(self.dx)):
                d = f * tp * 0.7
                s = max(1, int(tp * 0.18 * (1 - f)))
                pygame.draw.rect(surf, (235, 215, 70),
                                 (int(cx + self.dx[k] * d) - s, int(cy + self.dy[k] * d) - s, 2 * s, 2 * s))
        else:  # build: diffuse white sparkles
            for k in range(len(self.dx)):
                ff = min(1.0, max(0.0, f - self.delay[k]) / max(1e-3, 1 - self.delay[k]))
                if ff <= 0:
                    continue
                d = ff * tp * 0.85
                px, py = int(cx + self.dx[k] * d), int(cy + self.dy[k] * d)
                b = int(255 * (1 - ff))
                s = max(1, int(tp * 0.08 * (1 - ff) + 1))
                pygame.draw.rect(surf, (255, 255, 255), (px - s, py - s, 2 * s, 2 * s))
                if b > 60:                          # faint glow cross
                    pygame.draw.line(surf, (b, b, b), (px - s - 1, py), (px + s + 1, py))
                    pygame.draw.line(surf, (b, b, b), (px, py - s - 1), (px, py + s + 1))
        self.t += 1


# ─────────────────────────── menu / play drawing ──────────────────────────

def _txt(win, font, lines, x, y, color=(225, 225, 230), lh=None):
    lh = lh or font.get_height() + 4
    for i, ln in enumerate(lines):
        win.blit(font.render(ln, True, color), (x, y + i * lh))


def _button(win, rect, label, font, hover):
    pygame.draw.rect(win, (70, 110, 170) if hover else (44, 56, 80), rect, border_radius=8)
    pygame.draw.rect(win, (120, 160, 220) if hover else (80, 95, 130), rect, 2, border_radius=8)
    t = font.render(label, True, (240, 240, 250))
    win.blit(t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2))


def _draw_menu(win, big, font, w, h, mouse):
    win.fill(_BG)
    t = big.render("zebra_nav", True, (250, 230, 90))
    win.blit(t, ((w - t.get_width()) // 2, 70))
    _txt(win, font, ["partially-observed build/mine navigation"], (w - 360) // 2, 70 + t.get_height() + 6, (170, 170, 190))
    bw, bh = 320, 58
    cx = (w - bw) // 2
    rects = {"human": pygame.Rect(cx, h // 2 - 10, bw, bh),
             "ai": pygame.Rect(cx, h // 2 - 10 + bh + 18, bw, bh)}
    _button(win, rects["human"], "Play as Human   (H)", font, rects["human"].collidepoint(mouse))
    _button(win, rects["ai"], "Play as AI   (A)", font, rects["ai"].collidepoint(mouse))
    _txt(win, font, ["Q / Esc to quit"], (w - 140) // 2, h - 40, (140, 140, 160))
    return rects


def _draw_pick_agent(win, big, font, small, agents, sel, w, h):
    win.fill(_BG)
    win.blit(big.render("Pick an agent", True, (240, 240, 250)), (40, 24))
    _txt(win, small, ["↑/↓ navigate    Enter select    Esc back"], 40, 24 + big.get_height() + 6, (160, 160, 180))
    if not agents:
        _txt(win, font, ["No agents in models/zebra_nav/*.pt"], 40, 120, (240, 120, 120))
        return []
    top, row_h = 24 + big.get_height() + 46, 64
    rects = []
    for i, a in enumerate(agents):
        y = top + i * row_h
        rect = pygame.Rect(28, y - 4, w - 56, row_h - 8)
        rects.append(rect)
        if i == sel:
            pygame.draw.rect(win, (60, 90, 140), rect, border_radius=8)
        pygame.draw.rect(win, (90, 110, 150), rect, 1, border_radius=8)
        col = (255, 255, 255) if i == sel else (210, 210, 220)
        win.blit(font.render(a["name"], True, col), (44, y + 4))
        win.blit(small.render(f"{a['orientation']} · {a['descr']}", True, (170, 175, 190)), (44, y + 4 + font.get_height()))
    return rects


def _draw_pick_map(win, big, font, small, thumbs, labels, sel, w, h):
    win.fill(_BG)
    win.blit(big.render("Pick a map", True, (240, 240, 250)), (40, 24))
    _txt(win, small, ["arrows navigate    Enter start    Esc back"], 40, 24 + big.get_height() + 6, (160, 160, 180))
    cols = 3
    rows = max(1, (len(thumbs) + cols - 1) // cols)
    pad = 16
    top = 24 + big.get_height() + 50
    cw = (w - pad * (cols + 1)) // cols
    ch = (h - top - pad * (rows + 1) - 10) // rows
    rects = []
    for i, th in enumerate(thumbs):
        r, c = divmod(i, cols)
        x, y = pad + c * (cw + pad), top + r * (ch + pad)
        rect = pygame.Rect(x, y, cw, ch)
        rects.append(rect)
        if i == sel:
            pygame.draw.rect(win, (90, 130, 200), rect.inflate(8, 8), 3, border_radius=6)
        if th is not None:
            sc = pygame.transform.scale(th, (cw, ch - 22))
            win.blit(sc, (x, y))
        else:                                   # "Random" tile
            pygame.draw.rect(win, (40, 48, 66), (x, y, cw, ch - 22), border_radius=6)
            rt = font.render("🎲 Random", True, (235, 235, 120))
            win.blit(rt, (x + cw // 2 - rt.get_width() // 2, y + (ch - 22) // 2 - rt.get_height() // 2))
        win.blit(small.render(labels[i], True, (220, 220, 230)), (x + 4, y + ch - 20))
    return rects


def _draw_play(win, env, sprites, tp, main_px, panel_x, cell, effects, font, lines):
    win.fill(_BG)
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
    ar, ac = env._pos
    for e in effects:
        er, ec = e.cell
        vr, vc = er - ar + V // 2, ec - ac + V // 2
        if 0 <= vr < V and 0 <= vc < V:
            e.draw(win, vc * tp + tp // 2, vr * tp + tp // 2, tp)
    # minimap
    mm = _terrain_surface(env._terrain, cell)
    mx, my = panel_x + 14, 16
    pygame.draw.rect(win, (70, 70, 80), (mx - 2, my - 2, mm.get_width() + 4, mm.get_height() + 4), 1)
    win.blit(mm, (mx, my))
    pygame.draw.rect(win, (255, 255, 255), (mx + ac * cell - 1, my + ar * cell - 1, cell + 2, cell + 2))
    v = V // 2
    pygame.draw.rect(win, (255, 255, 0), (mx + (ac - v) * cell, my + (ar - v) * cell, V * cell, V * cell), 1)
    _txt(win, font, lines, mx, my + mm.get_height() + 16)


# ───────────────────────────── AI helper ──────────────────────────────────

def _ai_action(policy, obs, state, device):
    import torch
    with torch.no_grad():
        mm = torch.from_numpy(obs["minimap"])[None][None].to(device)
        sc = torch.from_numpy(obs["scalars"])[None][None].to(device)
        gout, h = policy._gru_forward({"minimap": mm, "scalars": sc},
                                      torch.zeros(1, 1, device=device), state["hidden"])
        state["hidden"] = h
        logits, _ = policy._heads(gout.squeeze(0))
        return int(torch.distributions.Categorical(logits=logits).sample())


# ───────────────────────────────── main ───────────────────────────────────

def _scan_agents():
    import yaml
    out = []
    for pt in sorted(_MODELS_DIR.glob("*.pt")):
        orient, descr = "?", ""
        y = pt.with_suffix(".yaml")
        if y.exists():
            try:
                cfg = yaml.safe_load(y.read_text())
                orient = cfg.get("orientation", "?")
            except Exception:
                pass
            for ln in y.read_text().splitlines():
                if ln.startswith("#") and "—" in ln:
                    descr = ln.split("—", 1)[1].strip()[:60]
                    break
        out.append({"path": pt, "name": pt.stem, "orientation": orient, "descr": descr})
    return out


def _map_set(cfg, n=9):
    """List of MapRecord for the active config (fixed seeds) + a None 'Random'."""
    kw = {k: cfg[k] for k in _GEN_KEYS if k in cfg and cfg[k] is not None}
    kw["size"] = cfg.get("env_size", 32)
    kw["width"] = cfg.get("env_width", kw["size"])
    recs = [generate_zebra_map(seed=10_000 + i, **kw) for i in range(n)]
    return recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main-px", type=int, default=600, help="target pixel size of the main play view")
    args = p.parse_args()

    import torch
    from train_ppo_zebra import PPOGRUPolicy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pygame.init()
    W0, H0, PANEL = 920, 740, 300
    win = pygame.display.set_mode((W0, H0))
    pygame.display.set_caption("zebra_nav")
    big = pygame.font.SysFont("monospace", 34, bold=True)
    font = pygame.font.SysFont("monospace", 20)
    small = pygame.font.SysFont("monospace", 15)
    clock = pygame.time.Clock()

    agents = _scan_agents()
    import yaml
    HUMAN_CFG = {"orientation": "natural", "env_size": 32, "env_width": 64, "view_size": 21,
                 "water_frac": 0.14, "rock_frac": 0.14, "tree_frac": 0.03, "goal_half": 4}

    S = {"state": "MENU", "mode": "ai", "agent": 0, "map": 0, "policy": None, "cfg": None,
         "thumbs": [], "labels": [], "recs": [], "env": None, "obs": None, "hidden": None,
         "status": "", "effects": [], "ai_play": True, "period": 6, "rng": np.random.default_rng(0),
         "rects": None, "tp": 28, "main_px": 600, "cell": 4}

    def cfg_of(i):
        y = agents[i]["path"].with_suffix(".yaml")
        return (yaml.safe_load(y.read_text()) if y.exists() else {})

    def make_env(cfg, max_steps=1500):
        m = [("env_size", "size"), ("env_width", "width"), ("view_size", "view_size"),
             ("orientation", "orientation"), ("n_stripes", "n_stripes"),
             ("thick_half", "thick_half"), ("thin_half", "thin_half"),
             ("obsidian_half", "obsidian_half"), ("window_h", "window_h"),
             ("water_frac", "water_frac"), ("rock_frac", "rock_frac"),
             ("tree_frac", "tree_frac"), ("goal_half", "goal_half")]
        ekw = {dst: cfg[src] for src, dst in m if cfg.get(src) is not None}
        ekw.setdefault("size", 32)
        return ZebraNavEnv(max_steps=max_steps, **ekw)

    def build_map_grid(cfg):
        S["recs"] = _map_set(cfg, 9)
        S["thumbs"] = [_terrain_surface(r.terrain, 6) for r in S["recs"]] + [None]
        S["labels"] = [f"map {i}" for i in range(len(S["recs"]))] + ["random"]
        S["map"] = 0

    def load_policy(i):
        ck = torch.load(agents[i]["path"], map_location="cpu", weights_only=False)
        ca = ck["args"]
        e = make_env(ca, 10)
        nt = int(ck["policy"]["tile_embed.weight"].shape[0])
        na = int(ck["policy"]["actor.weight"].shape[0])
        pol = PPOGRUPolicy(e.observation_space, num_actions=na, gru_hidden=ca.get("gru_hidden", 128),
                           embed_dim=ca.get("embed_dim", 256), num_tile_classes=nt).to(device)
        pol.load_state_dict(ck["policy"]); pol.eval()
        return pol

    def new_episode():
        env = S["env"]
        env._fixed_record = S["recs"][S["map"]] if S["map"] < len(S["recs"]) else None
        S["obs"], _ = env.reset()
        if S["policy"] is not None:
            S["hidden"] = torch.zeros(1, 1, S["policy"].gru_hidden, device=device)
        S["effects"] = []; S["status"] = "playing"

    def start_play():
        cfg = S["cfg"]; view = cfg.get("view_size", 11)
        S["env"] = make_env(cfg)
        S["tp"] = max(14, args.main_px // view)
        S["main_px"] = S["tp"] * view
        S["cell"] = max(2, min(7, (PANEL - 30) // max(2, S["env"].width)))
        S["sprites"] = _load_sprites(S["tp"])
        new_episode(); S["state"] = "PLAY"

    def do_action(a):
        env = S["env"]
        fr = (env._pos[0] + _FACE_DELTA[env._facing][0], env._pos[1] + _FACE_DELTA[env._facing][1])
        o, r, term, trunc, info = env.step(a)
        S["obs"] = o
        if info["mined"]:
            S["effects"].append(Effect(fr, "mine", S["rng"]))
        if info["placed"]:
            S["effects"].append(Effect(fr, "build", S["rng"]))
        S["status"] = f"step {info['step']}  ret {info['episode_return']:+.2f}"
        if term or trunc:
            S["status"] = "REACHED!" if term else "timeout"
            new_episode()

    def choose_human():
        S["mode"] = "human"; S["policy"] = None; S["cfg"] = HUMAN_CFG
        build_map_grid(S["cfg"]); S["state"] = "PICK_MAP"

    def choose_agent_enter():
        S["policy"] = load_policy(S["agent"]); S["cfg"] = cfg_of(S["agent"])
        build_map_grid(S["cfg"]); S["state"] = "PICK_MAP"

    is_ai = lambda: S["mode"] == "ai" and S["policy"] is not None
    KEYMAP = {pygame.K_UP: A_UP, pygame.K_DOWN: A_DOWN, pygame.K_LEFT: A_LEFT,
              pygame.K_RIGHT: A_RIGHT, pygame.K_b: A_PLACE, pygame.K_m: A_MINE}
    frame, running = 0, True
    while running:
        frame += 1
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            st = S["state"]
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    S["state"] = {"PLAY": "PICK_MAP",
                                  "PICK_MAP": ("PICK_AGENT" if S["mode"] == "ai" else "MENU"),
                                  "PICK_AGENT": "MENU", "MENU": "QUIT"}[st]
                    if S["state"] == "QUIT":
                        running = False
                elif st == "MENU":
                    if ev.key == pygame.K_q:
                        running = False
                    elif ev.key == pygame.K_h:
                        choose_human()
                    elif ev.key == pygame.K_a and agents:
                        S["mode"] = "ai"; S["state"] = "PICK_AGENT"
                elif st == "PICK_AGENT":
                    if ev.key in (pygame.K_UP, pygame.K_w):
                        S["agent"] = (S["agent"] - 1) % len(agents)
                    elif ev.key in (pygame.K_DOWN, pygame.K_s):
                        S["agent"] = (S["agent"] + 1) % len(agents)
                    elif ev.key == pygame.K_RETURN:
                        choose_agent_enter()
                elif st == "PICK_MAP":
                    n = len(S["thumbs"])
                    if ev.key in (pygame.K_LEFT, pygame.K_a):
                        S["map"] = (S["map"] - 1) % n
                    elif ev.key in (pygame.K_RIGHT, pygame.K_d):
                        S["map"] = (S["map"] + 1) % n
                    elif ev.key in (pygame.K_UP, pygame.K_w):
                        S["map"] = (S["map"] - 3) % n
                    elif ev.key in (pygame.K_DOWN, pygame.K_s):
                        S["map"] = (S["map"] + 3) % n
                    elif ev.key == pygame.K_RETURN:
                        start_play()
                elif st == "PLAY":
                    if ev.key == pygame.K_r:
                        new_episode()
                    elif ev.key == pygame.K_a and S["policy"] is not None:
                        S["ai_play"] = not S["ai_play"]
                    elif ev.key == pygame.K_SPACE and is_ai():
                        do_action(_ai_action(S["policy"], S["obs"], S, device))
                    elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        S["period"] = max(1, S["period"] - 1)
                    elif ev.key == pygame.K_MINUS:
                        S["period"] += 1
                    elif not is_ai() and ev.key in KEYMAP:
                        do_action(KEYMAP[ev.key])
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and S["rects"]:
                if st == "MENU":
                    if S["rects"]["human"].collidepoint(ev.pos):
                        choose_human()
                    elif S["rects"]["ai"].collidepoint(ev.pos) and agents:
                        S["mode"] = "ai"; S["state"] = "PICK_AGENT"
                elif st == "PICK_AGENT":
                    for i, rc in enumerate(S["rects"]):
                        if rc.collidepoint(ev.pos):
                            S["agent"] = i; choose_agent_enter(); break
                elif st == "PICK_MAP":
                    for i, rc in enumerate(S["rects"]):
                        if rc.collidepoint(ev.pos):
                            S["map"] = i; start_play(); break

        st = S["state"]
        if st == "PLAY" and is_ai() and S["ai_play"] and frame % S["period"] == 0:
            do_action(_ai_action(S["policy"], S["obs"], S, device))

        if st == "MENU":
            S["rects"] = _draw_menu(win, big, font, W0, H0, mouse)
        elif st == "PICK_AGENT":
            S["rects"] = _draw_pick_agent(win, big, font, small, agents, S["agent"], W0, H0)
        elif st == "PICK_MAP":
            S["rects"] = _draw_pick_map(win, big, font, small, S["thumbs"], S["labels"], S["map"], W0, H0)
        elif st == "PLAY":
            lines = [f"{'AI' if is_ai() else 'HUMAN'}{'  ||paused' if (is_ai() and not S['ai_play']) else ''}",
                     S["labels"][S["map"]], S["status"], "",
                     "arrows move  B build  M mine",
                     "A toggle-AI  Space step",
                     "+/- speed   R new   Esc back"]
            _draw_play(win, S["env"], S["sprites"], S["tp"], S["main_px"], S["main_px"], S["cell"],
                       S["effects"], small, lines)
            S["effects"][:] = [e for e in S["effects"] if e.alive()]
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
