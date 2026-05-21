#!/usr/bin/env python3
"""Playable pygame demo for the Cogniland navigation env.

Now a small state-machine app with a main menu:

    MAIN_MENU  ──┬──▶  PLAYING_HUMAN  ──┐
                 │                      │
                 └──▶  PICK_WEIGHTS  ──▶ PICK_MAP  ──▶ PLAYING_AI
                              ▲                              │
                              └──── B / Esc to step back ────┘

* **MAIN_MENU** — "Play as Human" (H) or "Play as AI" (A); Q/Esc to quit.
* **PICK_WEIGHTS** — scrollable list of every ``.pt`` found under
  ``runs/``; arrows + Enter to pick, B/Esc back.
* **PICK_MAP** — 4×3 grid of the 12 ``data/demo_maps/`` thumbnails;
  arrows + Enter to pick, B/Esc back to PICK_WEIGHTS.
* **PLAYING_HUMAN** — the original demo, keys unchanged.
* **PLAYING_AI** — same window layout but the policy drives the env at a
  configurable framerate (``+`` / ``-`` to adjust, default 8 fps).
  Episode auto-resets on the *same* selected map. ``B`` returns to
  MAIN_MENU.

Human controls
--------------
  arrow keys / WASD     move (up / down / left / right)
  B                     build with build_scalar=+1  → Raft
  V                     build with build_scalar=-1  → Harness
  N                     new map (same size + map_type)
  1 / 2 / 3 / 4         set size to 32 / 64 / 96 / 128, then new map
  M                     cycle map_type: random → lake → rocky → random
  D                     toggle debug overlay (shows map_type + correct_object)
  Q / Esc               quit

Usage
-----
    python scripts/play_cogniland.py --size 64
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

# macOS conda envs with both PyTorch's libomp and Python's libomp loaded will
# segfault inside pygame's display init. Set this BEFORE importing pygame.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pygame

# allow `python scripts/play_cogniland.py` from the repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogniland.nav import CognilandNavEnv, MapRecord  # noqa: E402
from cogniland.nav.renderer import SpriteSheet, render_color_grid  # noqa: E402

SIZES = (32, 64, 96, 128)
MAP_TYPES = ("random", "lake", "rocky", "balanced")
FACING_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}

# Demo-maps dataset produced by ``scripts/generate_demo_maps.py``.
DEMO_MAP_DIR = Path("data/demo_maps")
CKPT_SEARCH_DIRS = (Path("runs"),)


# ============================================================ helpers (shared)

def _hud_lines(env: CognilandNavEnv, info: dict, last_reward: float, debug: bool) -> list[str]:
    has_item = int(info["skill_active"])
    lines = [
        f"size={env.size}  map_type={info['map_type']}",
        f"pos={info['position']}",
        f"target={info['target']}",
        f"step={info['step']}/{env.max_steps}",
        f"object={info['active_object']}  has_item={has_item}",
        f"return={info['episode_return']:+.2f}",
        f"last_r={last_reward:+.2f}",
    ]
    if debug:
        lines.append("---")
        lines.append(f"[debug] correct={info['correct_object']}")
        lines.append(
            f"oracle no/r/h="
            f"{info['no_skill_oracle_cost']:.1f}/"
            f"{info['raft_oracle_cost']:.1f}/"
            f"{info['harness_oracle_cost']:.1f}"
        )
    return lines


def _make_env(size: int, map_type: str, view_size: int, tile_px: int, seed: int,
              obs_mode: str = "symbolic",
              map_record: MapRecord | None = None) -> CognilandNavEnv:
    return CognilandNavEnv(
        size=size,
        map_type=map_type,
        view_size=view_size,
        tile_px=tile_px,
        seed=seed,
        obs_mode=obs_mode,
        render_mode=None,
        map_record=map_record,
    )


def _blit_numpy(win, arr: np.ndarray, dest: tuple[int, int]) -> None:
    arr_c = np.ascontiguousarray(arr)
    surface = pygame.image.frombuffer(
        arr_c.tobytes(), (arr_c.shape[1], arr_c.shape[0]), "RGB"
    )
    win.blit(surface, dest)


def _render_minimap(
    terrain: np.ndarray,
    agent_pos: tuple[int, int],
    target_pos: tuple[int, int],
    main_view_tiles: int,
    cell_px: int,
) -> np.ndarray:
    """Full-map minimap with overlay markers."""
    base = render_color_grid(terrain, cell_px=cell_px)
    H, W, _ = base.shape

    ar, ac = agent_pos
    ay, ax = ar * cell_px, ac * cell_px
    if 0 <= ay < H and 0 <= ax < W:
        base[ay : ay + cell_px, ax : ax + cell_px] = (50, 200, 255)
    tr, tc = target_pos
    ty, tx = tr * cell_px, tc * cell_px
    if 0 <= ty < H and 0 <= tx < W:
        base[ty : ty + cell_px, tx : tx + cell_px] = (250, 220, 60)

    half = main_view_tiles // 2
    r0 = max(0, ar - half)
    c0 = max(0, ac - half)
    r1 = min(terrain.shape[0], ar + half + 1)
    c1 = min(terrain.shape[1], ac + half + 1)
    y0 = r0 * cell_px
    x0 = c0 * cell_px
    y1 = min(H - 1, r1 * cell_px - 1)
    x1 = min(W - 1, c1 * cell_px - 1)
    box = np.array((255, 255, 0), dtype=np.uint8)
    base[y0, x0 : x1 + 1] = box
    base[y1, x0 : x1 + 1] = box
    base[y0 : y1 + 1, x0] = box
    base[y0 : y1 + 1, x1] = box
    return base


def _render_main_view(
    sprites: SpriteSheet,
    terrain: np.ndarray,
    agent_pos: tuple[int, int],
    main_view_tiles: int,
    agent_facing: str,
) -> np.ndarray:
    chw = sprites.render_observation(
        terrain,
        agent_pos,
        view_size=main_view_tiles,
        agent_facing=agent_facing,
    )
    return chw.transpose(1, 2, 0)


def _overlay_obs_rect(canvas: np.ndarray, main_view_tiles: int, obs_view_size: int, tile_px: int) -> None:
    half_diff = (main_view_tiles - obs_view_size) // 2
    if half_diff <= 0:
        return
    y0 = half_diff * tile_px
    x0 = half_diff * tile_px
    y1 = (half_diff + obs_view_size) * tile_px - 1
    x1 = (half_diff + obs_view_size) * tile_px - 1
    color = np.array((255, 255, 0), dtype=np.uint8)
    canvas[y0, x0 : x1 + 1] = color
    canvas[y1, x0 : x1 + 1] = color
    canvas[y0 : y1 + 1, x0] = color
    canvas[y0 : y1 + 1, x1] = color


# ---- sweat-droplet animation -----------------------------------------------

SWEAT_LIFETIME_FRAMES = 28  # ~0.5s @ 60fps


def _draw_sweat_overlay(surface, sweat_anims: list[dict], frame: int,
                        main_view_tiles: int, tile_px: int) -> None:
    import pygame as _pg

    agent_view_cx = main_view_tiles // 2
    agent_view_cy = main_view_tiles // 2
    base_x = agent_view_cx * tile_px + tile_px // 2
    base_y = agent_view_cy * tile_px - 2

    for anim in sweat_anims:
        age = frame - anim["start_frame"]
        if age < 0 or age >= SWEAT_LIFETIME_FRAMES:
            continue
        t = age / SWEAT_LIFETIME_FRAMES
        alpha = int(220 * (1.0 - t))
        rise = int(t * 14)
        sway = anim["sway"] + int(2 * np.sin(t * 4))
        radius = max(3, tile_px // 4)
        droplet = _pg.Surface((radius * 2 + 2, radius * 2 + 4), _pg.SRCALPHA)
        _pg.draw.circle(droplet, (130, 200, 255, alpha), (radius + 1, radius + 2), radius)
        _pg.draw.circle(droplet, (220, 240, 255, alpha), (radius, radius + 1), max(1, radius // 3))
        surface.blit(droplet, (base_x - radius - 1 + sway, base_y - radius - 1 - rise))


def _prune_sweat(sweat_anims: list[dict], frame: int) -> list[dict]:
    return [a for a in sweat_anims if frame - a["start_frame"] < SWEAT_LIFETIME_FRAMES]


# =================================================== gameplay-frame renderer

def _draw_gameplay_frame(
    win: pygame.Surface,
    env: CognilandNavEnv,
    info: dict,
    sprites: SpriteSheet,
    main_view_tiles: int,
    main_tile_px: int,
    main_px: int,
    side_px: int,
    win_h: int,
    last_reward: float,
    debug: bool,
    sweat_anims: list[dict],
    frame_counter: int,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    controls: list[str],
    extra_status: str | None = None,
) -> None:
    """Render one gameplay frame. Used by both human and AI play states."""
    win.fill((20, 20, 24))

    # --- LEFT: egocentric main view ---------------------------------------
    main_img = _render_main_view(
        sprites,
        env._record.terrain,  # type: ignore[union-attr]
        info["position"],
        main_view_tiles,
        FACING_NAMES.get(env._facing, "down"),
    )
    _overlay_obs_rect(main_img, main_view_tiles, env.view_size, main_tile_px)
    main_img_c = np.ascontiguousarray(main_img)
    main_surface = pygame.image.frombuffer(
        main_img_c.tobytes(), (main_img_c.shape[1], main_img_c.shape[0]), "RGB"
    )
    _draw_sweat_overlay(main_surface, sweat_anims, frame_counter, main_view_tiles, main_tile_px)
    win.blit(main_surface, (0, 0))

    # --- RIGHT: minimap + HUD + controls ----------------------------------
    rx0 = main_px + 16
    mm_cell = max(1, (side_px - 8) // env.size)
    minimap = _render_minimap(
        env._record.terrain,  # type: ignore[union-attr]
        info["position"],
        info["target"],
        main_view_tiles,
        mm_cell,
    )
    mm_h, mm_w, _ = minimap.shape
    mm_x = rx0 + ((side_px - mm_w) // 2)
    mm_y = 8
    pygame.draw.rect(win, (60, 60, 70), (mm_x - 2, mm_y - 2, mm_w + 4, mm_h + 4), 1)
    _blit_numpy(win, minimap, (mm_x, mm_y))

    text_y = mm_y + mm_h + 12
    for line in _hud_lines(env, info, last_reward, debug):
        text = font.render(line, True, (240, 240, 240))
        win.blit(text, (rx0, text_y))
        text_y += 22
    if extra_status:
        text = font.render(extra_status, True, (255, 220, 120))
        win.blit(text, (rx0, text_y))
        text_y += 22

    line_h = 18
    panel_h = line_h * len(controls) + 10
    panel_w = side_px
    panel_x = rx0
    panel_y = win_h - panel_h - 6
    panel_y = max(text_y + 6, panel_y)
    pygame.draw.rect(win, (40, 40, 48), (panel_x, panel_y, panel_w, panel_h))
    pygame.draw.rect(win, (90, 90, 100), (panel_x, panel_y, panel_w, panel_h), 1)
    for i, line in enumerate(controls):
        text = small_font.render(line, True, (210, 210, 220))
        win.blit(text, (panel_x + 8, panel_y + 5 + i * line_h))

    pygame.display.flip()


# ====================================================== filesystem discovery

def _list_checkpoints() -> list[Path]:
    """Find every ``.pt`` under ``runs/`` (recursive)."""
    out: list[Path] = []
    for d in CKPT_SEARCH_DIRS:
        if not d.exists():
            continue
        out.extend(sorted(d.rglob("*.pt")))
    return out


def _list_demo_maps() -> list[Path]:
    if not DEMO_MAP_DIR.exists():
        return []
    return sorted(DEMO_MAP_DIR.glob("*.pkl"))


def _fmt_size(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n //= 1024
    return f"{n}T"


# ============================================================ menu rendering

def _draw_button(win, rect: pygame.Rect, label: str, font, hovered: bool) -> None:
    bg = (70, 110, 170) if hovered else (50, 70, 110)
    pygame.draw.rect(win, bg, rect)
    pygame.draw.rect(win, (180, 200, 240), rect, 2)
    text = font.render(label, True, (240, 240, 240))
    win.blit(
        text,
        (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2),
    )


def _draw_main_menu(win, font, big_font, win_w: int, win_h: int,
                    mouse_pos: tuple[int, int]) -> dict[str, pygame.Rect]:
    win.fill((18, 22, 30))
    title = big_font.render("Cogniland Nav", True, (240, 240, 250))
    win.blit(title, ((win_w - title.get_width()) // 2, 80))
    sub = font.render("Choose a play mode", True, (180, 180, 200))
    win.blit(sub, ((win_w - sub.get_width()) // 2, 80 + title.get_height() + 8))

    btn_w, btn_h = 320, 56
    cx = (win_w - btn_w) // 2
    cy = win_h // 2 - btn_h - 12
    rects = {
        "human": pygame.Rect(cx, cy, btn_w, btn_h),
        "ai": pygame.Rect(cx, cy + btn_h + 18, btn_w, btn_h),
    }
    _draw_button(win, rects["human"], "Play as Human  (H)", font, rects["human"].collidepoint(mouse_pos))
    _draw_button(win, rects["ai"], "Play as AI  (A)", font, rects["ai"].collidepoint(mouse_pos))

    hint = font.render("Q / Esc to quit", True, (140, 140, 160))
    win.blit(hint, ((win_w - hint.get_width()) // 2, win_h - hint.get_height() - 16))
    pygame.display.flip()
    return rects


def _draw_pick_weights(win, font, small_font, big_font,
                       ckpts: list[Path], sel: int,
                       win_w: int, win_h: int) -> None:
    win.fill((18, 22, 30))
    title = big_font.render("Pick PPO checkpoint", True, (240, 240, 250))
    win.blit(title, (40, 24))
    sub = small_font.render("up/down  navigate     enter  select     B / Esc  back", True, (160, 160, 180))
    win.blit(sub, (40, 24 + title.get_height() + 6))

    if not ckpts:
        msg = font.render("No .pt files found under runs/", True, (240, 120, 120))
        win.blit(msg, (40, 120))
        pygame.display.flip()
        return

    list_top = 24 + title.get_height() + 36
    row_h = 26
    visible_rows = max(1, (win_h - list_top - 20) // row_h)
    # simple windowed view so the selected row is always visible.
    start = max(0, min(sel - visible_rows // 2, len(ckpts) - visible_rows))
    start = max(0, start)
    end = min(len(ckpts), start + visible_rows)
    for i in range(start, end):
        y = list_top + (i - start) * row_h
        is_sel = i == sel
        if is_sel:
            pygame.draw.rect(win, (60, 90, 140), (32, y - 2, win_w - 64, row_h))
        path = ckpts[i]
        try:
            stat = path.stat()
            size_str = _fmt_size(stat.st_size)
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
        except OSError:
            size_str = "?"
            mtime = "?"
        label = f"{i+1:>3}. {path}  ({size_str}, {mtime})"
        color = (250, 250, 250) if is_sel else (200, 200, 210)
        win.blit(small_font.render(label, True, color), (40, y))
    pygame.display.flip()


def _load_thumbnail(png_path: Path, target_w: int, target_h: int) -> pygame.Surface:
    img = pygame.image.load(str(png_path)).convert()
    return pygame.transform.smoothscale(img, (target_w, target_h))


def _draw_pick_map(win, font, small_font, big_font,
                   maps: list[Path], thumbs: list[pygame.Surface],
                   sel: int, win_w: int, win_h: int) -> list[pygame.Rect]:
    win.fill((18, 22, 30))
    title = big_font.render("Pick a map", True, (240, 240, 250))
    win.blit(title, (40, 24))
    sub = small_font.render("arrows  navigate     enter  start     B / Esc  back", True, (160, 160, 180))
    win.blit(sub, (40, 24 + title.get_height() + 6))

    rects: list[pygame.Rect] = []
    if not maps:
        msg = font.render("No maps in data/demo_maps/. Run scripts/generate_demo_maps.py.", True, (240, 120, 120))
        win.blit(msg, (40, 120))
        pygame.display.flip()
        return rects

    # 4 columns × 3 rows grid (12 maps).
    cols, rows = 4, 3
    pad = 16
    grid_top = 24 + title.get_height() + 56
    cell_w = (win_w - pad * (cols + 1)) // cols
    cell_h = (win_h - grid_top - pad * (rows + 1) - 30) // rows
    for i, path in enumerate(maps):
        r, c = divmod(i, cols)
        if r >= rows:
            break
        x = pad + c * (cell_w + pad)
        y = grid_top + r * (cell_h + pad)
        rect = pygame.Rect(x, y, cell_w, cell_h)
        rects.append(rect)
        if i == sel:
            pygame.draw.rect(win, (90, 130, 200), rect.inflate(8, 8), 3)
        # thumbnail
        thumb = thumbs[i]
        tx = x + (cell_w - thumb.get_width()) // 2
        ty = y + (cell_h - thumb.get_height() - 20) // 2
        win.blit(thumb, (tx, ty))
        # label below
        win.blit(
            small_font.render(path.stem, True, (220, 220, 230)),
            (x + 6, y + cell_h - 22),
        )
    pygame.display.flip()
    return rects


# ============================================================ play loops

def _action_from_keydown(key: int) -> dict | None:
    if key in (pygame.K_UP, pygame.K_w):
        return {"move": 0, "build_scalar": np.array([0.0], np.float32)}
    if key in (pygame.K_DOWN, pygame.K_s):
        return {"move": 1, "build_scalar": np.array([0.0], np.float32)}
    if key in (pygame.K_LEFT, pygame.K_a):
        return {"move": 2, "build_scalar": np.array([0.0], np.float32)}
    if key in (pygame.K_RIGHT, pygame.K_d):
        return {"move": 3, "build_scalar": np.array([0.0], np.float32)}
    if key == pygame.K_b:
        return {"move": 4, "build_scalar": np.array([+1.0], np.float32)}
    if key == pygame.K_v:
        return {"move": 4, "build_scalar": np.array([-1.0], np.float32)}
    return None


class GameplayCtx:
    """Bag of mutable state shared by the gameplay rendering helper."""

    def __init__(self, args, win, sprites, font, small_font,
                 main_view_tiles: int, main_tile_px: int, main_px: int,
                 side_px: int, win_h: int, controls: list[str]):
        self.args = args
        self.win = win
        self.sprites = sprites
        self.font = font
        self.small_font = small_font
        self.main_view_tiles = main_view_tiles
        self.main_tile_px = main_tile_px
        self.main_px = main_px
        self.side_px = side_px
        self.win_h = win_h
        self.controls = controls
        self.sweat_anims: list[dict] = []
        self.frame_counter = 0
        self.last_reward = 0.0
        self.debug = False

    def render(self, env, info, extra_status: str | None = None) -> None:
        self.sweat_anims = _prune_sweat(self.sweat_anims, self.frame_counter)
        _draw_gameplay_frame(
            self.win, env, info, self.sprites,
            self.main_view_tiles, self.main_tile_px, self.main_px,
            self.side_px, self.win_h,
            self.last_reward, self.debug, self.sweat_anims, self.frame_counter,
            self.font, self.small_font, self.controls, extra_status=extra_status,
        )

    def on_step(self, info: dict) -> None:
        if info.get("slipped"):
            self.sweat_anims.append({
                "start_frame": self.frame_counter,
                "sway": int(np.random.randint(-2, 3)),
            })


def _play_human(args, ctx: GameplayCtx) -> str:
    """Run the human play loop. Returns the next state name."""
    obs_view_size = ctx.main_view_tiles if ctx.main_view_tiles < args.view_size else args.view_size
    obs_view_size = min(args.view_size, ctx.main_view_tiles)
    env = _make_env(args.size, args.map_type, obs_view_size, ctx.main_tile_px, args.seed)
    obs, info = env.reset()
    map_type_idx = MAP_TYPES.index(args.map_type)
    clock = pygame.time.Clock()
    ctx.controls = [
        "WASD / arrows  move",
        "B  build raft",
        "V  build harness",
        "N  new map",
        "1-4  size 32/64/96/128",
        "M  cycle map type",
        "D  toggle debug",
        "B-menu  ([) back to menu",
        "Q / Esc  quit",
    ]

    def rebuild(new_size: int, new_map_type: str) -> None:
        nonlocal env, obs, info
        env.close()
        env = _make_env(new_size, new_map_type, obs_view_size, ctx.main_tile_px, args.seed + 1)
        args.seed += 1
        obs, info = env.reset()
        ctx.last_reward = 0.0
        ctx.sweat_anims.clear()

    ctx.render(env, info)
    while True:
        action: dict | None = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return "QUIT"
            if event.type != pygame.KEYDOWN:
                continue
            k = event.key
            if k in (pygame.K_q, pygame.K_ESCAPE):
                env.close()
                return "QUIT"
            if k == pygame.K_LEFTBRACKET:
                env.close()
                return "MAIN_MENU"
            if k in (pygame.K_n,):
                rebuild(env.size, env.map_type)
            elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                idx = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3}[k]
                rebuild(SIZES[idx], env.map_type)
            elif k == pygame.K_m:
                map_type_idx = (map_type_idx + 1) % len(MAP_TYPES)
                rebuild(env.size, MAP_TYPES[map_type_idx])
            elif k == pygame.K_d:
                ctx.debug = not ctx.debug
            else:
                action = _action_from_keydown(k)
                if action is None:
                    continue

        if action is not None:
            obs, ctx.last_reward, terminated, truncated, info = env.step(action)
            ctx.on_step(info)
            if terminated or truncated:
                ctx.render(env, info)
                pygame.time.wait(500)
                obs, info = env.reset()
                ctx.last_reward = 0.0
                ctx.sweat_anims.clear()

        ctx.render(env, info)
        ctx.frame_counter += 1
        clock.tick(60)


def _play_ai(args, ctx: GameplayCtx, ckpt_path: Path, map_record: MapRecord) -> str:
    """Run the AI play loop. Returns the next state name."""
    # Lazy import torch so the menus stay snappy when no checkpoint is picked.
    from cogniland.inference import PPOAgent

    try:
        agent = PPOAgent.load(ckpt_path, device="cpu")
    except Exception as exc:  # noqa: BLE001 — surface any load error to the user
        print(f"[error] failed to load checkpoint {ckpt_path}: {exc}")
        return "MAIN_MENU"

    # Match the env the policy was trained on (obs_mode in particular).
    ck = agent.ckpt_args
    obs_mode = ck.get("obs_mode", "symbolic")
    view_size = int(ck.get("view_size", args.view_size))
    if view_size % 2 == 0:
        view_size += 1
    env = _make_env(
        size=int(getattr(map_record, "terrain").shape[0]),
        map_type=getattr(map_record, "map_type", "random"),
        view_size=view_size,
        tile_px=ctx.main_tile_px,
        seed=args.seed,
        obs_mode=obs_mode,
        map_record=map_record,
    )
    obs, info = env.reset()
    hidden = agent.initial_hidden(1)
    done = False
    fps = 8.0
    clock = pygame.time.Clock()

    ctx.controls = [
        "AI play",
        f"checkpoint: {ckpt_path.name}",
        "+ / -  speed",
        "R  reset same map",
        "D  toggle debug",
        "[ back to menu",
        "Q / Esc  quit",
    ]

    last_step_time = 0.0
    ctx.render(env, info, extra_status=f"AI  fps={fps:.1f}")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return "QUIT"
            if event.type != pygame.KEYDOWN:
                continue
            k = event.key
            if k in (pygame.K_q, pygame.K_ESCAPE):
                env.close()
                return "QUIT"
            if k == pygame.K_LEFTBRACKET:
                env.close()
                return "MAIN_MENU"
            if k == pygame.K_d:
                ctx.debug = not ctx.debug
            elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                fps = min(60.0, fps * 1.5)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                fps = max(1.0, fps / 1.5)
            elif k == pygame.K_r:
                obs, info = env.reset()
                hidden = agent.initial_hidden(1)
                done = False
                ctx.last_reward = 0.0
                ctx.sweat_anims.clear()

        # Step the agent at the requested framerate (independent of the
        # 60 fps render loop so the window stays responsive).
        now = time.time()
        if (now - last_step_time) >= (1.0 / fps):
            action, hidden = agent.act(obs, hidden, done=done, greedy=True)
            obs, r, term, trunc, info = env.step(action)
            ctx.last_reward = float(r)
            ctx.on_step(info)
            done = bool(term or trunc)
            last_step_time = now
            if done:
                ctx.render(env, info, extra_status=f"AI  fps={fps:.1f}  episode done")
                pygame.time.wait(700)
                obs, info = env.reset()
                hidden = agent.initial_hidden(1)
                done = False
                ctx.last_reward = 0.0
                ctx.sweat_anims.clear()

        ctx.render(env, info, extra_status=f"AI  fps={fps:.1f}")
        ctx.frame_counter += 1
        clock.tick(60)


# ============================================================ main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64, choices=SIZES)
    parser.add_argument("--map-type", default="random", choices=MAP_TYPES)
    parser.add_argument("--view-size", type=int, default=21, help="agent partial-obs side")
    parser.add_argument("--main-view-tiles", type=int, default=None,
                        help="egocentric main view side (defaults to view-size so the "
                             "demo shows exactly what the agent sees)")
    parser.add_argument("--side-px", type=int, default=320, help="right panel width in px")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.main_view_tiles is None:
        args.main_view_tiles = args.view_size
    if args.main_view_tiles % 2 == 0:
        args.main_view_tiles += 1

    pygame.init()
    pygame.key.set_repeat(120, 80)
    screen_info = pygame.display.Info()

    max_w = max(600, screen_info.current_w - 60)
    max_h = max(400, screen_info.current_h - 80)
    avail = min(max_h, max_w - args.side_px - 16)
    main_tile_px = max(6, avail // args.main_view_tiles)
    main_px = main_tile_px * args.main_view_tiles
    side_px = args.side_px
    win_w = main_px + side_px + 16
    win_h = main_px

    win = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Cogniland Nav")
    font = pygame.font.SysFont(None, 22)
    small_font = pygame.font.SysFont(None, 18)
    big_font = pygame.font.SysFont(None, 44)
    sprites = SpriteSheet(tile_px=main_tile_px)

    ctx = GameplayCtx(args, win, sprites, font, small_font,
                      args.main_view_tiles, main_tile_px, main_px, side_px, win_h,
                      controls=[])

    state = "MAIN_MENU"
    selected_ckpt: Path | None = None
    selected_map: Path | None = None
    selected_record: MapRecord | None = None

    # Cached lists/thumbnails — only rebuild when entering the relevant state.
    ckpts: list[Path] = []
    ckpt_sel = 0
    maps: list[Path] = []
    map_sel = 0
    thumbs: list[pygame.Surface] = []

    running = True
    while running:
        if state == "MAIN_MENU":
            rects = _draw_main_menu(win, font, big_font, win_w, win_h, pygame.mouse.get_pos())
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    k = event.key
                    if k in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif k == pygame.K_h:
                        state = "PLAYING_HUMAN"
                    elif k == pygame.K_a:
                        state = "PICK_WEIGHTS"
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if rects["human"].collidepoint(event.pos):
                        state = "PLAYING_HUMAN"
                    elif rects["ai"].collidepoint(event.pos):
                        state = "PICK_WEIGHTS"
            pygame.time.wait(16)

        elif state == "PICK_WEIGHTS":
            if not ckpts:
                ckpts = _list_checkpoints()
                ckpt_sel = 0
            _draw_pick_weights(win, font, small_font, big_font, ckpts, ckpt_sel, win_w, win_h)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    k = event.key
                    if k in (pygame.K_q, pygame.K_ESCAPE) or k == pygame.K_LEFTBRACKET or k == pygame.K_b:
                        # B/Esc/[: go back to main menu
                        state = "MAIN_MENU"
                        ckpts = []  # force rescan next time
                    elif k == pygame.K_UP and ckpts:
                        ckpt_sel = (ckpt_sel - 1) % len(ckpts)
                    elif k == pygame.K_DOWN and ckpts:
                        ckpt_sel = (ckpt_sel + 1) % len(ckpts)
                    elif k in (pygame.K_RETURN, pygame.K_KP_ENTER) and ckpts:
                        selected_ckpt = ckpts[ckpt_sel]
                        state = "PICK_MAP"
            pygame.time.wait(16)

        elif state == "PICK_MAP":
            if not maps:
                maps = _list_demo_maps()
                map_sel = 0
                thumbs = []
                # Pre-load and scale thumbnails once. Sized to fit the 4x3 grid.
                cols, rows = 4, 3
                pad = 16
                grid_top_est = 24 + 44 + 56  # rough — see _draw_pick_map
                cell_w = (win_w - pad * (cols + 1)) // cols
                cell_h = (win_h - grid_top_est - pad * (rows + 1) - 30) // rows
                tw, th = max(32, cell_w - 16), max(32, cell_h - 32)
                for p in maps:
                    png = p.with_suffix(".png")
                    try:
                        thumbs.append(_load_thumbnail(png, tw, th))
                    except Exception as exc:  # noqa: BLE001
                        print(f"[warn] failed to load thumbnail {png}: {exc}")
                        thumbs.append(pygame.Surface((tw, th)))
            rects = _draw_pick_map(win, font, small_font, big_font, maps, thumbs, map_sel, win_w, win_h)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    k = event.key
                    if k in (pygame.K_q, pygame.K_ESCAPE) or k == pygame.K_LEFTBRACKET or k == pygame.K_b:
                        state = "PICK_WEIGHTS"
                        maps = []
                    elif maps and k in (pygame.K_LEFT, pygame.K_a):
                        map_sel = (map_sel - 1) % len(maps)
                    elif maps and k in (pygame.K_RIGHT, pygame.K_d):
                        map_sel = (map_sel + 1) % len(maps)
                    elif maps and k in (pygame.K_UP, pygame.K_w):
                        map_sel = (map_sel - 4) % len(maps)
                    elif maps and k in (pygame.K_DOWN, pygame.K_s):
                        map_sel = (map_sel + 4) % len(maps)
                    elif maps and k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        selected_map = maps[map_sel]
                        try:
                            with selected_map.open("rb") as f:
                                selected_record = pickle.load(f)
                        except Exception as exc:  # noqa: BLE001
                            print(f"[error] failed to load {selected_map}: {exc}")
                            continue
                        state = "PLAYING_AI"
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for i, rect in enumerate(rects):
                        if rect.collidepoint(event.pos):
                            map_sel = i
                            break
            pygame.time.wait(16)

        elif state == "PLAYING_HUMAN":
            ctx.frame_counter = 0
            ctx.last_reward = 0.0
            ctx.sweat_anims = []
            next_state = _play_human(args, ctx)
            state = "MAIN_MENU" if next_state == "MAIN_MENU" else "QUIT"
            if state == "QUIT":
                running = False

        elif state == "PLAYING_AI":
            ctx.frame_counter = 0
            ctx.last_reward = 0.0
            ctx.sweat_anims = []
            assert selected_ckpt is not None and selected_record is not None
            next_state = _play_ai(args, ctx, selected_ckpt, selected_record)
            state = "MAIN_MENU" if next_state == "MAIN_MENU" else "QUIT"
            if state == "QUIT":
                running = False

        else:
            print(f"[bug] unknown state {state!r}")
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
