#!/usr/bin/env python3
"""Playable pygame demo for the bridge_tunnel env — human or trained AI.

A small state-machine app (in the style of play_cogniland.py):

    MAIN_MENU ──┬─▶ PICK_MAP ─▶ PLAY            (Human)
                └─▶ PICK_AGENT ─▶ PICK_MAP ─▶ PLAY   (AI)

* MAIN_MENU  — Play as Human / Play as AI.
* PICK_AGENT — scrollable list of models/bridge_tunnel/*.pt (orientation shown).
* PICK_MAP   — thumbnail grid of maps for the chosen agent's orientation
               (the curated validation maps for natural), plus a Random tile.
* PLAY       — large egocentric Crafter-sprite view + small pixel minimap, with
               mining / bridge-building animations.

    python scripts/play_bridge_tunnel.py
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

# macOS: PyTorch (libomp/MKL) + SDL/OpenBLAS double-load segfaults. Pin threads
# and allow the duplicate libomp BEFORE importing any native lib. Importing torch
# here (in a controlled order, before pygame) makes libomp load once up front.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    torch = None
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.bridge_tunnel.env import (  # noqa: E402
    BridgeTunnelEnv, _FACE_DELTA, A_UP, A_DOWN, A_LEFT, A_RIGHT, A_PLACE, A_MINE,
    F_UP, F_DOWN, F_LEFT, F_RIGHT,
)
from cogniland.bridge_tunnel import generate_bridge_tunnel_map, tiles as T  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_SPRITE_DIR = _REPO / "src/cogniland/assets/sprites"
_MODELS_DIR = _REPO / "models/bridge_tunnel"
_VAL_MAPS = _REPO / "data/bridge_tunnel/val_maps.pkl"

_BASE = {T.GRASS: "grass", T.WATER: "water", T.ROCK: "stone", T.WOOD: "path",
         T.TREE: "tree", T.SAND: "sand", T.DIRT: "path", T.TARGET: "grass"}
_OVERLAY = {T.TARGET: "flag"}
_FACE_SPRITE = {F_UP: "player-up", F_DOWN: "player-down", F_LEFT: "player-left", F_RIGHT: "player-right"}
_BG = (18, 22, 30)
# generate_bridge_tunnel_map kwargs we may pull from an agent's stored args (natural-only)
_GEN_KEYS = ("orientation", "water_frac", "rock_frac", "tree_frac", "goal_half")


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
    t = big.render("bridge_tunnel", True, (250, 230, 90))
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
        _txt(win, font, ["No agents in models/bridge_tunnel/*.pt"], 40, 120, (240, 120, 120))
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


# ──────────────────────── action-distribution pad ─────────────────────────

_ACT_LABEL = {A_PLACE: "build", A_MINE: "mine"}
FLASH_DUR = 12                      # frames the chosen action stays lit before fading


def _prob_color(p):
    """Light gradient: dark slate (p=0) → bright cyan (p=1)."""
    if p is None:
        return (40, 46, 62)
    p = max(0.0, min(1.0, float(p)))
    lo, hi = (40, 46, 62), (95, 205, 255)
    return tuple(int(lo[i] + (hi[i] - lo[i]) * p) for i in range(3))


def _draw_arrow(win, rect, action, color):
    cx, cy = rect.center
    s = int(min(rect.w, rect.h) * 0.24)
    if action == A_UP:
        pts = [(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)]
    elif action == A_DOWN:
        pts = [(cx, cy + s), (cx - s, cy - s), (cx + s, cy - s)]
    elif action == A_LEFT:
        pts = [(cx - s, cy), (cx + s, cy - s), (cx + s, cy + s)]
    else:                            # A_RIGHT
        pts = [(cx + s, cy), (cx - s, cy - s), (cx - s, cy + s)]
    pygame.draw.polygon(win, color, pts)


def _shadow_text(win, font, text, cx, by):
    """White text with a 1px dark drop-shadow (readable on any button fill)."""
    fg = font.render(text, True, (245, 247, 250))
    sh = font.render(text, True, (18, 22, 30))
    x = cx - fg.get_width() // 2
    win.blit(sh, (x + 1, by + 1)); win.blit(fg, (x, by))


def _pad_button(win, rect, action, prob, flash_k, small, small_bold):
    """flash_k in [0, 1]: 1=just chosen (full highlight), 0=back to prob color."""
    base = _prob_color(prob)
    hi = (255, 232, 64)
    col = tuple(int(base[i] + (hi[i] - base[i]) * flash_k) for i in range(3))
    pygame.draw.rect(win, col, rect, border_radius=7)
    edge_lo, edge_hi = (120, 140, 175), (255, 255, 255)
    edge = tuple(int(edge_lo[i] + (edge_hi[i] - edge_lo[i]) * flash_k) for i in range(3))
    pygame.draw.rect(win, edge, rect, 2 if flash_k > 0.05 else 1, border_radius=7)
    if action in (A_UP, A_DOWN, A_LEFT, A_RIGHT):
        _draw_arrow(win, rect, action, (245, 247, 250))
    else:                                            # build / mine — bold label
        lbl = small_bold.render(_ACT_LABEL[action], True, (245, 247, 250))
        _shadow_text(win, small_bold, _ACT_LABEL[action], rect.centerx,
                     rect.centery - lbl.get_height())
    if prob is not None:
        _shadow_text(win, small, f"{prob * 100:.0f}%", rect.centerx,
                     rect.bottom - small.get_height() - 3)


def _draw_action_pad(win, x0, Wp, bottom_y, probs, flash, font, small, small_bold):
    """A 4-way d-pad + build/mine buttons, each shaded by its policy probability;
    the most-recently-taken action lights up bright yellow once and fades out."""
    bs = max(40, min(64, Wp // 3 - 6))
    cx0 = x0 + (Wp - 3 * bs) // 2
    act_h = int(bs * 0.8)
    total_h = 22 + 3 * bs + 10 + act_h
    y0 = bottom_y - total_h
    win.blit(font.render("policy  pi(a|s)", True, (175, 182, 200)), (x0, y0))
    gy = y0 + 22
    P = (lambda a: None) if probs is None else (lambda a: float(probs[a]))

    def k(a):                          # smooth fade-out (ease-out cubic)
        if not flash or flash["a"] != a:
            return 0.0
        f = max(0.0, min(1.0, flash["t"] / FLASH_DUR))
        return f * f * f
    cells = {A_UP:    pygame.Rect(cx0 + bs, gy, bs, bs),
             A_LEFT:  pygame.Rect(cx0, gy + bs, bs, bs),
             A_RIGHT: pygame.Rect(cx0 + 2 * bs, gy + bs, bs, bs),
             A_DOWN:  pygame.Rect(cx0 + bs, gy + 2 * bs, bs, bs)}
    for a, rc in cells.items():
        _pad_button(win, rc, a, P(a), k(a), small, small_bold)
    ay, gap = gy + 3 * bs + 10, 8
    aw = (Wp - gap) // 2
    for i, a in enumerate((A_PLACE, A_MINE)):
        _pad_button(win, pygame.Rect(x0 + i * (aw + gap), ay, aw, act_h), a, P(a), k(a), small, small_bold)


# ───────────────────────────── AI helper ──────────────────────────────────

def _ai_action(policy, obs, state, device):
    """Returns (sampled action, action-probability vector over the 6 actions)."""
    import torch
    with torch.no_grad():
        mm = torch.from_numpy(obs["minimap"])[None][None].to(device)
        sc = torch.from_numpy(obs["scalars"])[None][None].to(device)
        gout, h = policy._gru_forward({"minimap": mm, "scalars": sc},
                                      torch.zeros(1, 1, device=device), state["hidden"])
        state["hidden"] = h
        logits, _ = policy._heads(gout.squeeze(0))
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        a = int(torch.distributions.Categorical(logits=logits).sample())
        return a, probs


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
    recs = [generate_bridge_tunnel_map(seed=10_000 + i, **kw) for i in range(n)]
    return recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main-px", type=int, default=600, help="target pixel size of the main play view")
    args = p.parse_args()

    from train_ppo_bridge_tunnel import PPOGRUPolicy
    device = torch.device("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")

    pygame.init()
    W0, H0, PANEL = 920, 740, 300
    win = pygame.display.set_mode((W0, H0))
    pygame.display.set_caption("bridge_tunnel")
    big = pygame.font.SysFont("monospace", 34, bold=True)
    font = pygame.font.SysFont("monospace", 20)
    small = pygame.font.SysFont("monospace", 15)
    small_bold = pygame.font.SysFont("monospace", 16, bold=True)
    clock = pygame.time.Clock()

    agents = _scan_agents()
    import yaml
    HUMAN_CFG = {"orientation": "natural", "env_size": 32, "env_width": 64, "view_size": 21,
                 "water_frac": 0.14, "rock_frac": 0.14, "tree_frac": 0.03, "goal_half": 4}

    S = {"state": "MENU", "mode": "ai", "agent": 0, "map": 0, "policy": None, "cfg": None,
         "thumbs": [], "labels": [], "recs": [], "env": None, "obs": None, "hidden": None,
         "status": "", "effects": [], "ai_play": True, "period": 6, "rng": np.random.default_rng(0),
         "rects": None, "tp": 28, "main_px": 600, "cell": 4, "probs": None, "flash": None}

    def cfg_of(i):
        y = agents[i]["path"].with_suffix(".yaml")
        return (yaml.safe_load(y.read_text()) if y.exists() else {})

    def make_env(cfg, max_steps=1500):
        m = [("env_size", "size"), ("env_width", "width"), ("view_size", "view_size"),
             ("orientation", "orientation"),
             ("water_frac", "water_frac"), ("rock_frac", "rock_frac"),
             ("tree_frac", "tree_frac"), ("goal_half", "goal_half")]
        ekw = {dst: cfg[src] for src, dst in m if cfg.get(src) is not None}
        ekw.setdefault("size", 32)
        return BridgeTunnelEnv(max_steps=max_steps, **ekw)

    def build_map_grid(cfg):
        # NATURAL maps are generated with opensimplex, which segfaults on some
        # machines (macOS/arm) — so the demo NEVER generates them: it always
        # uses the curated, pre-pickled validation set (== the eval maps).
        if not _VAL_MAPS.exists():
            raise SystemExit(
                "natural demo needs data/bridge_tunnel/val_maps.pkl — "
                "generate it once on a machine where opensimplex works: "
                "python scripts/make_bridge_tunnel_val_maps.py")
        S["val_pool"] = pickle.load(open(_VAL_MAPS, "rb"))["records"]
        S["recs"] = S["val_pool"][:9]
        S["thumbs"] = [_terrain_surface(r.terrain, 6) for r in S["recs"]] + [None]
        S["labels"] = [f"map {i}" for i in range(len(S["recs"]))] + ["random"]
        S["map"] = 0

    def load_policy(i):
        ck = torch.load(agents[i]["path"], map_location="cpu", weights_only=False)
        ca = ck["args"]
        e = make_env(ca, 10)
        na = int(ck["policy"]["actor.weight"].shape[0])
        oe = ca.get("obs_encoding", "embed")
        if "tile_embed.weight" in ck["policy"]:
            nt = int(ck["policy"]["tile_embed.weight"].shape[0])
        else:                                   # onehot: K = conv in-channels − 2 (CoordConv)
            nt = int(ck["policy"]["cnn.0.weight"].shape[1]) - 2; oe = "onehot"
        pol = PPOGRUPolicy(e.observation_space, num_actions=na, gru_hidden=ca.get("gru_hidden", 128),
                           embed_dim=ca.get("embed_dim", 256), num_tile_classes=nt,
                           obs_encoding=oe).to(device)
        pol.load_state_dict(ck["policy"]); pol.eval()
        return pol

    def new_episode():
        env = S["env"]
        if S["map"] < len(S["recs"]):
            env._fixed_record = S["recs"][S["map"]]
        else:                             # "random" → a random val map (never generate)
            pool = S.get("val_pool") or S["recs"]
            env._fixed_record = pool[int(S["rng"].integers(len(pool)))]
        S["obs"], _ = env.reset()
        if S["policy"] is not None:
            S["hidden"] = torch.zeros(1, 1, S["policy"].gru_hidden, device=device)
        S["effects"] = []; S["status"] = "playing"; S["probs"] = None; S["flash"] = None

    def start_play():
        cfg = S["cfg"]; view = cfg.get("view_size", 11)
        S["env"] = make_env(cfg)
        S["tp"] = max(14, args.main_px // view)
        S["main_px"] = S["tp"] * view
        S["cell"] = max(2, min(7, (PANEL - 30) // max(2, S["env"].width)))
        S["sprites"] = _load_sprites(S["tp"])
        new_episode(); S["state"] = "PLAY"

    def do_action(a, probs=None):
        env = S["env"]
        S["flash"] = {"a": int(a), "t": FLASH_DUR}      # blink the chosen action
        if probs is not None:
            S["probs"] = probs
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
                        _a, _p = _ai_action(S["policy"], S["obs"], S, device)
                        do_action(_a, _p)
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
            _a, _p = _ai_action(S["policy"], S["obs"], S, device)
            do_action(_a, _p)

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
            # live policy action-distribution pad in the bottom of the right panel
            _padx = S["main_px"] + 14
            _draw_action_pad(win, _padx, W0 - _padx - 14, H0 - 64, S["probs"], S["flash"],
                             font, small, small_bold)
            if S["flash"]:
                S["flash"]["t"] -= 1
                if S["flash"]["t"] <= 0:
                    S["flash"] = None
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
