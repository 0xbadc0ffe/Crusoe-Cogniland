"""Pygame demo for the pure-JAX Cogniland env.

Default view: the 45×45 minimap takes the main area (scaled up). Press
``M`` to toggle a full-map panel on the **right** side of the window;
the minimap stays as the primary view on the left.

Controls:
    WASD / arrows  — move (4 cardinal moves)
    F              — forage (berry → +100 HP; forest → +10 wood)
    1 / 2 / 3      — craft raft / rope / shoes (one tool only; 100 wood)
    M              — toggle the full-map overlay
    R              — reset to a fresh episode
    ESC            — quit

Usage:
    python -m cogniland_jax.demo_pygame --maps data/maps/val.pt --difficulty hard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover
    print("pygame not installed. `pip install 'cogniland-jax[demo]'`")
    sys.exit(1)

from cogniland_jax import CognilandEnv, EnvParams
from cogniland_jax import constants as C
from cogniland_jax.maps import load_map_arrays


# ── Tile-class colour palette (matches the generator's RGB scheme) ─────
_TILE_COLORS = {
    C.TILE_UNSEEN:      (22, 22, 30),
    1:  (10, 35, 225),     # ocean
    2:  (25, 65, 225),     # deep_water
    3:  (65, 105, 225),    # water
    4:  (238, 214, 175),   # beach
    5:  (210, 180, 140),   # sandy
    6:  (34, 139, 34),     # grassland
    7:  (10, 110, 10),     # forest
    8:  (149, 147, 147),   # rocky
    9:  (245, 240, 240),   # mountains
    C.TILE_BERRY:       (155, 35, 60),
    C.TILE_TARGET_YES:  (60, 255, 80),
    C.TILE_TARGET_NO:   (255, 60, 60),
    C.TILE_DEADLY:      (0, 0, 0),
}

# Key → action id (cardinal moves). Action 4 (forage) / 5-7 (craft)
# come from their own keys so the layout is readable.
_KEY_MOVE = {
    pygame.K_UP: 0, pygame.K_w: 0,
    pygame.K_DOWN: 1, pygame.K_s: 1,
    pygame.K_LEFT: 2, pygame.K_a: 2,
    pygame.K_RIGHT: 3, pygame.K_d: 3,
}


def _tile_array_to_surface(tile_arr: np.ndarray, pixel: int) -> "pygame.Surface":
    """Render an integer tile-class array as an RGB pygame surface."""
    H, W = tile_arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, col in _TILE_COLORS.items():
        rgb[tile_arr == cls] = col
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    return pygame.transform.scale(surf, (W * pixel, H * pixel))


def _full_tile_array(state, params):
    """Build an H×W int tile-class array of the current map (no occlusion).

    Used by the full-map toggle. Highlights YES/NO targets and the agent.
    """
    terrain = np.asarray(params.terrain_idx[int(state.map_idx)])
    berry = np.asarray(params.berry_mask[int(state.map_idx)])
    tile = np.where(terrain < 0, C.TILE_DEADLY, (terrain.astype(np.int16) + 1))
    tile = np.where(berry & (terrain >= 0), C.TILE_BERRY, tile)
    tile[int(state.yes_r), int(state.yes_c)] = C.TILE_TARGET_YES
    tile[int(state.no_r), int(state.no_c)] = C.TILE_TARGET_NO
    return tile


def _draw_agent(surf, px, cy, cx, color=(255, 60, 60), radius_frac=0.35) -> None:
    pygame.draw.circle(surf, color,
                       (int((cx + 0.5) * px), int((cy + 0.5) * px)),
                       int(px * radius_frac))


def _draw_hud(screen, font_small, font_big, state, last_msg, params_difficulty):
    w = screen.get_width()
    h = screen.get_height()
    # Stats bar at top
    hp = float(state.hp)
    wood = int(state.wood)
    tool = int(state.tool)
    steps = int(state.steps)
    tool_name = ["none", "raft", "rope", "shoes"][tool]
    diff_label = ["easy", "medium", "hard"][int(params_difficulty) % 3]

    hp_color = (70, 210, 110) if hp > 60 else (240, 190, 60) if hp > 25 else (240, 80, 70)
    t1 = font_big.render(f"HP {hp:5.1f}", True, hp_color)
    t2 = font_small.render(f"wood {wood}/100   tool {tool_name}   steps {steps}   "
                           f"difficulty {diff_label}", True, (215, 215, 220))
    screen.blit(t1, (20, 14))
    screen.blit(t2, (20, 50))
    if last_msg:
        msg = font_small.render(last_msg, True, (255, 200, 60))
        screen.blit(msg, (20, h - 28))


def _terminal_banner(done, state, info_reached):
    if not done:
        return None
    if info_reached:
        return "reached target — press R to restart"
    if int(state.hp) <= 0:
        return "died — press R to restart"
    return "episode over — press R to restart"


def run_demo(
    maps_path: str,
    difficulty: str = "hard",
    biome_filter=None,
    seed: int = 0,
    minimap_scale: int = 12,
    full_map_scale: int = 3,
) -> None:
    arrays = load_map_arrays(maps_path, biome_filter=biome_filter)
    diff_map = {"easy": 0, "medium": 1, "hard": 2}
    params = EnvParams.from_map_arrays(
        **arrays,
        difficulty=jnp.int32(diff_map[difficulty]),
    )
    env = CognilandEnv(default_params=params)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # JIT-compile on startup so the first key press isn't laggy.
    key = jax.random.PRNGKey(seed)
    obs, state = reset_fn(key, params)
    for a in range(C.NUM_ACTIONS):
        k, key = jax.random.split(key)
        _ = step_fn(k, state, jnp.int32(a), params)[0]["minimap"].block_until_ready()

    # Pygame setup
    pygame.init()
    pygame.display.set_caption("cogniland-jax demo")
    mm_px = minimap_scale
    mm_side = C.MINIMAP_DIAMETER * mm_px
    full_side = C.MAP_SIZE * full_map_scale
    pad = 32
    hud_h = 100

    show_full = False

    def _window_size():
        w = pad + mm_side + pad + (full_side + pad if show_full else 0)
        h = max(mm_side, full_side) + hud_h + pad
        return max(w, 480), h

    screen = pygame.display.set_mode(_window_size())
    font_small = pygame.font.SysFont("DejaVu Sans Mono", 16)
    font_big = pygame.font.SysFont("DejaVu Sans Mono", 28, bold=True)
    clock = pygame.time.Clock()

    last_msg = ""
    last_info_reached = False
    running = True
    while running:
        action = None
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key in _KEY_MOVE:
                    action = _KEY_MOVE[ev.key]
                elif ev.key == pygame.K_f:
                    action = 4
                elif ev.key == pygame.K_1:
                    action = 5
                elif ev.key == pygame.K_2:
                    action = 6
                elif ev.key == pygame.K_3:
                    action = 7
                elif ev.key == pygame.K_m:
                    show_full = not show_full
                    screen = pygame.display.set_mode(_window_size())
                elif ev.key == pygame.K_r:
                    key, rk = jax.random.split(key)
                    obs, state = reset_fn(rk, params)
                    last_msg = "new episode"
                    last_info_reached = False

        if action is not None and not bool(state.terminated):
            key, sk = jax.random.split(key)
            obs, state, reward, done, info = step_fn(
                sk, state, jnp.int32(action), params,
            )
            last_info_reached = bool(info["reached"])
            if action == 4:
                prev_hp = float(info["hp_prev"])
                cur_hp = float(info["hp_curr"])
                if cur_hp > prev_hp:
                    last_msg = f"berry! +{int(cur_hp - prev_hp)} HP"
                else:
                    last_msg = "foraged" if cur_hp != prev_hp else "nothing to forage"
            elif action >= 5:
                tool_names = {5: "raft", 6: "rope", 7: "shoes"}
                if int(info["crafted"]) > 0:
                    last_msg = f"crafted {tool_names[action]}"
                else:
                    last_msg = "can't craft (need 100 wood, no tool)"
            banner = _terminal_banner(bool(done), state, last_info_reached)
            if banner:
                last_msg = banner

        screen.fill((22, 22, 30))

        # Minimap on the left (always visible).
        minimap_np = np.asarray(obs["minimap"])
        mm_surf = _tile_array_to_surface(minimap_np, mm_px)
        screen.blit(mm_surf, (pad, hud_h))
        # Agent marker in the centre of the minimap.
        _draw_agent(screen, mm_px,
                    hud_h // mm_px + C.MINIMAP_RADIUS,
                    pad // mm_px + C.MINIMAP_RADIUS)
        # Actually: compute pixel coords directly (pad + centre).
        cx = pad + C.MINIMAP_RADIUS * mm_px + mm_px // 2
        cy = hud_h + C.MINIMAP_RADIUS * mm_px + mm_px // 2
        pygame.draw.circle(screen, (255, 60, 60), (cx, cy), int(mm_px * 0.4))

        if show_full:
            full = _full_tile_array(state, params)
            fm_surf = _tile_array_to_surface(full, full_map_scale)
            fx = pad + mm_side + pad
            fy = hud_h
            screen.blit(fm_surf, (fx, fy))
            # Agent marker on the full map.
            ax = fx + int(state.pos_c) * full_map_scale + full_map_scale // 2
            ay = fy + int(state.pos_r) * full_map_scale + full_map_scale // 2
            pygame.draw.circle(screen, (255, 60, 60), (ax, ay), max(3, full_map_scale - 1))

        _draw_hud(screen, font_small, font_big, state, last_msg,
                  int(params.difficulty))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="data/maps/val.pt",
                    help="Path to pre-generated map .pt file")
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")
    ap.add_argument("--biome", action="append", default=None,
                    help="Biome filter; pass repeatedly for multiple (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minimap-scale", type=int, default=12)
    ap.add_argument("--full-map-scale", type=int, default=3)
    args = ap.parse_args()

    if not Path(args.maps).exists():
        print(f"Maps file not found: {args.maps}")
        sys.exit(1)

    run_demo(
        maps_path=args.maps,
        difficulty=args.difficulty,
        biome_filter=args.biome,
        seed=args.seed,
        minimap_scale=args.minimap_scale,
        full_map_scale=args.full_map_scale,
    )


if __name__ == "__main__":
    main()
