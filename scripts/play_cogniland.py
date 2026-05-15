#!/usr/bin/env python3
"""Playable pygame demo for the Cogniland navigation env.

The window is split in two:

* **Left** — an *egocentric* main view that follows the agent. It shows a
  ``--main-view-tiles`` × ``--main-view-tiles`` patch around the agent
  (default 21), rendered with the Crafter sprites. Inside it a yellow
  rectangle outlines the agent's actual partial observation (``view_size``
  tiles), which is what a trained policy would see.
* **Right** — a small **minimap** of the full map (flat-colour cells) with
  the agent (cyan), target (yellow), and the egocentric viewport
  (yellow box) marked. Below the minimap sit the HUD stats and the
  controls panel.

Controls
--------
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
import sys
from pathlib import Path

# macOS conda envs with both PyTorch's libomp and Python's libomp loaded will
# segfault inside pygame's display init. Set this BEFORE importing pygame.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pygame

# allow `python scripts/play_cogniland.py` from the repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogniland.nav import CognilandNavEnv  # noqa: E402
from cogniland.nav.renderer import SpriteSheet, render_color_grid  # noqa: E402

SIZES = (32, 64, 96, 128)
MAP_TYPES = ("random", "lake", "rocky", "balanced")
FACING_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}


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


def _make_env(size: int, map_type: str, view_size: int, tile_px: int, seed: int) -> CognilandNavEnv:
    # The demo renders sprites directly; the env only needs a cheap obs.
    return CognilandNavEnv(
        size=size,
        map_type=map_type,
        view_size=view_size,
        tile_px=tile_px,
        seed=seed,
        obs_mode="symbolic",
        render_mode=None,
    )


def _blit_numpy(win, arr: np.ndarray, dest: tuple[int, int]) -> None:
    """Blit an HxWx3 uint8 numpy array to the pygame window."""
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
    """Full-map minimap with overlay markers: agent (cyan), target (gold), and
    a yellow box for the egocentric viewport.
    """
    base = render_color_grid(terrain, cell_px=cell_px)
    H, W, _ = base.shape

    # agent dot
    ar, ac = agent_pos
    ay, ax = ar * cell_px, ac * cell_px
    if 0 <= ay < H and 0 <= ax < W:
        base[ay : ay + cell_px, ax : ax + cell_px] = (50, 200, 255)
    # target dot
    tr, tc = target_pos
    ty, tx = tr * cell_px, tc * cell_px
    if 0 <= ty < H and 0 <= tx < W:
        base[ty : ty + cell_px, tx : tx + cell_px] = (250, 220, 60)

    # egocentric view rect (clipped to map bounds)
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
    """Egocentric crop centred on the agent, HxWx3 uint8."""
    chw = sprites.render_observation(
        terrain,
        agent_pos,
        view_size=main_view_tiles,
        agent_facing=agent_facing,
    )
    return chw.transpose(1, 2, 0)


def _overlay_obs_rect(canvas: np.ndarray, main_view_tiles: int, obs_view_size: int, tile_px: int) -> None:
    """Outline the inner ``obs_view_size`` square inside the egocentric view."""
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


def _draw_sweat_overlay(surface, sweat_anims: list[dict], frame: int, main_view_tiles: int, tile_px: int) -> None:
    """Draw all active sweat droplets on top of the main-view surface.

    Each droplet starts above the agent (centre tile of the egocentric
    view), rises a few pixels, and fades to transparent.
    """
    import pygame as _pg

    agent_view_cx = main_view_tiles // 2
    agent_view_cy = main_view_tiles // 2
    base_x = agent_view_cx * tile_px + tile_px // 2
    base_y = agent_view_cy * tile_px - 2  # just above the agent's head

    for anim in sweat_anims:
        age = frame - anim["start_frame"]
        if age < 0 or age >= SWEAT_LIFETIME_FRAMES:
            continue
        t = age / SWEAT_LIFETIME_FRAMES  # 0 → 1
        alpha = int(220 * (1.0 - t))
        rise = int(t * 14)  # px upward
        sway = anim["sway"] + int(2 * np.sin(t * 4))
        radius = max(3, tile_px // 4)
        droplet = _pg.Surface((radius * 2 + 2, radius * 2 + 4), _pg.SRCALPHA)
        # body: light-blue circle
        _pg.draw.circle(droplet, (130, 200, 255, alpha), (radius + 1, radius + 2), radius)
        # specular highlight
        _pg.draw.circle(droplet, (220, 240, 255, alpha), (radius, radius + 1), max(1, radius // 3))
        surface.blit(droplet, (base_x - radius - 1 + sway, base_y - radius - 1 - rise))


def _prune_sweat(sweat_anims: list[dict], frame: int) -> list[dict]:
    return [a for a in sweat_anims if frame - a["start_frame"] < SWEAT_LIFETIME_FRAMES]


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
        args.main_view_tiles += 1  # must be odd to centre the agent

    pygame.init()
    pygame.key.set_repeat(120, 80)
    screen_info = pygame.display.Info()

    # Fit the egocentric main view to the largest tile_px the screen allows.
    # We want the main view to fill nearly the whole window vertically, with
    # `side_px` reserved on the right for the minimap + HUD + controls.
    max_w = max(600, screen_info.current_w - 60)
    max_h = max(400, screen_info.current_h - 80)
    # main view is square; pick tile_px so square fits in min(max_h, max_w - side_px).
    avail = min(max_h, max_w - args.side_px - 16)
    main_tile_px = max(6, avail // args.main_view_tiles)
    main_px = main_tile_px * args.main_view_tiles
    side_px = args.side_px

    obs_view_size = min(args.view_size, args.main_view_tiles)
    env = _make_env(args.size, args.map_type, obs_view_size, main_tile_px, args.seed)
    sprites = SpriteSheet(tile_px=main_tile_px)
    obs, info = env.reset()
    last_reward = 0.0
    debug = False
    map_type_idx = MAP_TYPES.index(args.map_type)

    win_w = main_px + side_px + 16
    win_h = main_px
    win = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Cogniland Nav — egocentric demo")
    font = pygame.font.SysFont(None, 22)
    small_font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()

    controls = [
        "WASD / arrows  move",
        "B  build raft",
        "V  build harness",
        "N  new map",
        "1-4  size 32/64/96/128",
        "M  cycle map type",
        "D  toggle debug",
        "Q / Esc  quit",
    ]

    sweat_anims: list[dict] = []
    frame_counter = 0

    def _redraw() -> None:
        nonlocal win, sweat_anims
        win.fill((20, 20, 24))

        # ---- LEFT: egocentric main view ----------------------------------
        main_img = _render_main_view(
            sprites,
            env._record.terrain,  # type: ignore[union-attr]
            info["position"],
            args.main_view_tiles,
            FACING_NAMES.get(env._facing, "down"),
        )
        _overlay_obs_rect(main_img, args.main_view_tiles, env.view_size, main_tile_px)
        # convert numpy → Surface so we can blit sweat droplets on top
        main_img_c = np.ascontiguousarray(main_img)
        main_surface = pygame.image.frombuffer(
            main_img_c.tobytes(), (main_img_c.shape[1], main_img_c.shape[0]), "RGB"
        )
        sweat_anims = _prune_sweat(sweat_anims, frame_counter)
        _draw_sweat_overlay(main_surface, sweat_anims, frame_counter, args.main_view_tiles, main_tile_px)
        win.blit(main_surface, (0, 0))

        # ---- RIGHT: minimap + HUD + controls -----------------------------
        rx0 = main_px + 16

        # minimap sits in (rx0, 8)..(rx0 + side_px, 8 + side_px) at most
        mm_cell = max(1, (side_px - 8) // env.size)
        minimap = _render_minimap(
            env._record.terrain,  # type: ignore[union-attr]
            info["position"],
            info["target"],
            args.main_view_tiles,
            mm_cell,
        )
        mm_h, mm_w, _ = minimap.shape
        mm_x = rx0 + ((side_px - mm_w) // 2)
        mm_y = 8
        # border behind the minimap
        pygame.draw.rect(
            win,
            (60, 60, 70),
            (mm_x - 2, mm_y - 2, mm_w + 4, mm_h + 4),
            1,
        )
        _blit_numpy(win, minimap, (mm_x, mm_y))

        # HUD lines below the minimap
        text_y = mm_y + mm_h + 12
        for line in _hud_lines(env, info, last_reward, debug):
            text = font.render(line, True, (240, 240, 240))
            win.blit(text, (rx0, text_y))
            text_y += 22

        # Controls panel at the bottom of the right column
        line_h = 18
        panel_h = line_h * len(controls) + 10
        panel_w = side_px
        panel_x = rx0
        panel_y = win_h - panel_h - 6
        # don't let controls overlap with HUD if window is short
        panel_y = max(text_y + 6, panel_y)
        pygame.draw.rect(win, (40, 40, 48), (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(win, (90, 90, 100), (panel_x, panel_y, panel_w, panel_h), 1)
        for i, line in enumerate(controls):
            text = small_font.render(line, True, (210, 210, 220))
            win.blit(text, (panel_x + 8, panel_y + 5 + i * line_h))

        pygame.display.flip()

    _redraw()

    def _rebuild_env(new_size: int, new_map_type: str) -> None:
        nonlocal env, sprites, obs, info, last_reward
        env = _make_env(new_size, new_map_type, obs_view_size, main_tile_px, args.seed + 1)
        args.seed += 1
        obs, info = env.reset()
        last_reward = 0.0

    running = True
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k in (pygame.K_UP, pygame.K_w):
                    action = {"move": 0, "build_scalar": np.array([0.0], np.float32)}
                elif k in (pygame.K_DOWN, pygame.K_s):
                    action = {"move": 1, "build_scalar": np.array([0.0], np.float32)}
                elif k in (pygame.K_LEFT, pygame.K_a):
                    action = {"move": 2, "build_scalar": np.array([0.0], np.float32)}
                elif k in (pygame.K_RIGHT, pygame.K_d):
                    action = {"move": 3, "build_scalar": np.array([0.0], np.float32)}
                elif k == pygame.K_b:
                    action = {"move": 4, "build_scalar": np.array([+1.0], np.float32)}
                elif k == pygame.K_v:
                    action = {"move": 4, "build_scalar": np.array([-1.0], np.float32)}
                elif k == pygame.K_n:
                    _rebuild_env(env.size, env.map_type)
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    idx = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3}[k]
                    _rebuild_env(SIZES[idx], env.map_type)
                elif k == pygame.K_m:
                    map_type_idx = (map_type_idx + 1) % len(MAP_TYPES)
                    _rebuild_env(env.size, MAP_TYPES[map_type_idx])
                elif k == pygame.K_d:
                    debug = not debug

        if action is not None:
            obs, last_reward, terminated, truncated, info = env.step(action)
            if info.get("slipped"):
                sweat_anims.append({
                    "start_frame": frame_counter,
                    "sway": int(np.random.randint(-2, 3)),
                })
            if terminated or truncated:
                _redraw()
                pygame.time.wait(500)
                obs, info = env.reset()
                last_reward = 0.0
                sweat_anims.clear()

        _redraw()
        frame_counter += 1
        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
