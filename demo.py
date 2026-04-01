#!/usr/bin/env python3
"""Cogniland Demo — Play as human or watch a trained AI agent.

Usage:
    # Generate demo maps first (once):
    python scripts/generate_demo_maps.py

    # Then launch:
    python demo.py

Main menu:
    H — Play as Human
    A — Watch AI Agent
    ESC — Quit
"""

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pygame
import torch
from omegaconf import OmegaConf

from cogniland.env.constants import ACTIONS, NUM_ACTIONS
from cogniland.env.islands import Islands
from cogniland.env.types import CustomMapConfig, EnvConfig, EnvState, MapGenConfig, MinimapConfig
from cogniland.models.ppo import ActorCritic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
_model_cfg = OmegaConf.load(_CONFIGS_DIR / "models" / "ppo.yaml")
_env_cfg   = OmegaConf.load(_CONFIGS_DIR / "env"    / "default.yaml")

MODEL_SCALAR_DIM       = _model_cfg.get("scalar_dim", 7)
MODEL_MINIMAP_CHANNELS = _model_cfg.get("minimap_channels", 2)
MODEL_HIDDEN_DIM       = _model_cfg.get("hidden_dim", 128)
MODEL_ACTION_DIM       = _model_cfg.get("action_dim", NUM_ACTIONS)
MODEL_CNN_CHANNELS     = _model_cfg.get("cnn_channels", 32)
MODEL_CNN_OUT_SPATIAL  = _model_cfg.get("cnn_out_spatial", 4)
MODEL_SCALAR_HIDDEN    = _model_cfg.get("scalar_hidden", 64)

_env_mm = _env_cfg.get("minimap", _env_cfg)
ENV_MM_MAX_RAY  = _env_mm.get("max_ray",          _env_cfg.get("minimap_max_ray",          22))
ENV_MM_OCCLUDE  = _env_mm.get("occlude",           _env_cfg.get("minimap_occlude",          True))
ENV_MM_CLR_TOL  = _env_mm.get("clear_tolerance",   _env_cfg.get("minimap_clear_tolerance",  0.1))
_env_mg = _env_cfg.get("map_generation", _env_cfg)
ENV_MAP_SIZE    = _env_mg.get("size", _env_cfg.get("size", 250))

VAL_MAPS_PATH        = Path("data/val_seed42_n16.pt")
BEHAVIORAL_MAPS_PATH = Path("data/test_behavior.pt")

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------
WINDOW_W, WINDOW_H   = 1200, 800
MAP_DISPLAY_SIZE     = 550
MINIMAP_DISPLAY_SIZE = 220
ACTION_NAMES         = {0: "↑", 1: "↓", 2: "→", 3: "←", 4: "•"}

COLORS = {
    "player":    (255,  50,  50),
    "target":    ( 50, 255,  50),
    "black":     (  0,   0,   0),
    "white":     (255, 255, 255),
    "gray":      (128, 128, 128),
    "red":       (255,   0,   0),
    "green_ui":  (  0, 220,   0),
    "blue_ui":   ( 80, 140, 255),
    "panel_bg":  ( 25,  25,  35),
    "panel_fg":  (200, 200, 210),
    "highlight": (255, 200,  50),
}

# ---------------------------------------------------------------------------
# Demo-map helpers
# ---------------------------------------------------------------------------


def _fast_colorize(world_map: np.ndarray, compiled) -> np.ndarray:
    """Vectorised heightmap → [H, W, 3] uint8.  Much faster than colorize()."""
    thresholds = compiled.thresholds.cpu().numpy()
    color_lut  = compiled.color_lut.cpu().numpy()          # [T, 3] uint8
    indices    = np.searchsorted(thresholds, world_map).clip(0, compiled.num_terrains - 1)
    return color_lut[indices].astype(np.uint8)             # [H, W, 3]


def maps_to_thumbnails(maps: torch.Tensor, compiled, size: int) -> list[pygame.Surface]:
    """Render each map as a scaled pygame Surface thumbnail."""
    thumbs = []
    for wm in maps:
        rgb  = _fast_colorize(wm.numpy(), compiled)        # [H, W, 3]
        surf = pygame.Surface((wm.shape[1], wm.shape[0]))
        pygame.surfarray.blit_array(surf, rgb.transpose(1, 0, 2))
        thumbs.append(pygame.transform.scale(surf, (size, size)))
    return thumbs


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def draw_star(surface, cx, cy, r_outer, r_inner=None, color=(255, 215, 0),
              outline=(0, 0, 0), n_points=5):
    """Draw a matplotlib-style 5-pointed star (inner radius ≈ 0.38 × outer)."""
    if r_inner is None:
        r_inner = r_outer * 0.38
    pts = []
    for i in range(2 * n_points):
        r = r_outer if i % 2 == 0 else r_inner
        angle = math.pi * i / n_points - math.pi / 2
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, outline, pts, 1)


def terrain_color(level, compiled):
    idx = int(level)
    if 0 <= idx < compiled.num_terrains:
        c = compiled.color_lut[idx].cpu().tolist()
        return (c[0], c[1], c[2])
    return COLORS["white"]


def heightmap_to_surface(world_map, display_size, compiled, vis_mask=None):
    """Render a 2-D heightmap tensor → scaled pygame Surface (slow path for small surfaces)."""
    H, W = world_map.shape[:2]
    wm_np   = world_map.numpy() if world_map.dim() == 2 else world_map[..., 0].numpy()
    rgb     = _fast_colorize(wm_np, compiled)
    if vis_mask is not None:
        mask_np = vis_mask.numpy() if hasattr(vis_mask, 'numpy') else vis_mask
        rgb[mask_np < 0.5] = 0
    surf    = pygame.Surface((W, H))
    pygame.surfarray.blit_array(surf, rgb.transpose(1, 0, 2))
    return pygame.transform.scale(surf, (display_size, display_size))


def make_map_surface_with_fog(base_rgb, seen_mask, map_size, display_size):
    fog = np.where(seen_mask[:, :, None], 1.0, 0.55).astype(np.float32)
    rgb = (base_rgb * fog).astype(np.uint8)
    surf = pygame.Surface((map_size, map_size))
    pygame.surfarray.blit_array(surf, rgb.transpose(1, 0, 2))
    return pygame.transform.scale(surf, (display_size, display_size))


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def discover_checkpoints(artifacts_dir="artifacts"):
    results = []
    if not os.path.isdir(artifacts_dir):
        return results
    for entry in sorted(os.listdir(artifacts_dir)):
        run_dir = os.path.join(artifacts_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        pts = [f for f in os.listdir(run_dir) if f.endswith(".pt")]
        if not pts:
            continue
        def _step(name):
            m = re.search(r"ckpt_(\d+)\.pt", name)
            return int(m.group(1)) if m else 0
        pts.sort(key=_step)
        results.append((entry, os.path.join(run_dir, pts[-1])))
    return results


def load_actor_critic(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    # Infer architecture dims from checkpoint weights
    cnn_channels    = sd["cnn.3.weight"].shape[0]          # e.g. 64 or 32
    minimap_channels = sd["cnn.0.weight"].shape[1]         # e.g. 3
    scalar_dim      = sd["scalar_net.0.weight"].shape[1]   # e.g. 5
    scalar_hidden   = sd["scalar_net.0.weight"].shape[0]   # e.g. 128 or 64
    hidden_dim      = sd["trunk.0.weight"].shape[0]        # e.g. 448 or 256
    action_dim      = sd["actor.weight"].shape[0]          # e.g. 5
    cnn_out_flat    = sd["trunk.0.weight"].shape[1] - scalar_hidden
    cnn_out_spatial = int((cnn_out_flat // cnn_channels) ** 0.5)
    model = ActorCritic(
        scalar_dim=scalar_dim,
        minimap_channels=minimap_channels,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        cnn_channels=cnn_channels,
        cnn_out_spatial=cnn_out_spatial,
        scalar_hidden=scalar_hidden,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def build_obs(state: EnvState, env_config: EnvConfig, compiled):
    s = state
    num_terrains = compiled.num_terrains
    scalars = torch.stack([
        s.compass[:, 0],
        s.compass[:, 1],
        s.terrain_idx / max(num_terrains - 1, 1),
        s.resources / env_config.max_resources,
        s.hp / env_config.max_hp,
    ], dim=1)
    return {"scalars": scalars, "minimap": s.minimap}


# ---------------------------------------------------------------------------
# Screen: main menu
# ---------------------------------------------------------------------------

def screen_main_menu(screen, clock):
    """Returns 'human', 'agent', or None (quit)."""
    font_title = pygame.font.Font(None, 64)
    font_med   = pygame.font.Font(None, 34)
    font_small = pygame.font.Font(None, 24)

    has_maps = VAL_MAPS_PATH.exists() or BEHAVIORAL_MAPS_PATH.exists()

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:    return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return None
                if ev.key == pygame.K_h:      return "human"
                if ev.key == pygame.K_a:      return "agent"

        screen.fill(COLORS["panel_bg"])

        title = font_title.render("Cogniland", True, COLORS["blue_ui"])
        screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 130))

        sub = font_med.render("Choose a mode", True, COLORS["panel_fg"])
        screen.blit(sub, (WINDOW_W // 2 - sub.get_width() // 2, 210))

        for key, label, desc, y in [
            ("H", "Human",    "Play the game yourself",           320),
            ("A", "AI Agent", "Watch a trained agent navigate",   400),
        ]:
            ks  = font_med.render(f"[{key}]",    True, COLORS["blue_ui"])
            ls  = font_med.render(f"  {label}",  True, COLORS["white"])
            ds  = font_small.render(desc,         True, COLORS["gray"])
            x   = WINDOW_W // 2 - 160
            screen.blit(ks, (x, y))
            screen.blit(ls, (x + ks.get_width(), y))
            screen.blit(ds, (x + 10, y + 32))

        if not has_maps:
            warn = font_small.render(
                "No demo maps found — a random map will be generated each session.",
                True, (255, 180, 50),
            )
            screen.blit(warn, (WINDOW_W // 2 - warn.get_width() // 2, 510))
            gen_hint = font_small.render(
                "Generate maps: python scripts/generate_demo_maps.py",
                True, COLORS["gray"],
            )
            screen.blit(gen_hint, (WINDOW_W // 2 - gen_hint.get_width() // 2, 534))

        hint = font_small.render("ESC — Quit", True, COLORS["gray"])
        screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 50))

        pygame.display.flip()
        clock.tick(30)


# ---------------------------------------------------------------------------
# Screen: map selection
# ---------------------------------------------------------------------------

def _screen_pick_map_grid(screen, clock, maps, compiled, labels=None):
    """Thumbnail grid for a set of maps. Returns selected index or None (ESC)."""
    N = maps.shape[0]
    print("Building map thumbnails …", flush=True)
    thumbs = maps_to_thumbnails(maps, compiled, 150)
    print("Done.")
    if labels is None:
        labels = [f"Map {i + 1}" for i in range(N)]

    COLS, THUMB, PAD, GRID_TOP = 4, 150, 14, 90
    GRID_W = COLS * THUMB + (COLS - 1) * PAD
    GRID_X = (WINDOW_W - GRID_W) // 2

    def tile_rect(idx):
        col = idx % COLS
        row = idx // COLS
        return pygame.Rect(GRID_X + col * (THUMB + PAD),
                           GRID_TOP + row * (THUMB + PAD + 18), THUMB, THUMB)

    font_large = pygame.font.Font(None, 42)
    font_small = pygame.font.Font(None, 20)
    selected   = 0

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:    return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return None
                if ev.key == pygame.K_RETURN: return selected
                if ev.key == pygame.K_RIGHT:  selected = (selected + 1) % N
                if ev.key == pygame.K_LEFT:   selected = (selected - 1) % N
                if ev.key == pygame.K_DOWN:   selected = min(selected + COLS, N - 1)
                if ev.key == pygame.K_UP:     selected = max(selected - COLS, 0)
            if ev.type == pygame.MOUSEMOTION:
                for i in range(N):
                    if tile_rect(i).collidepoint(ev.pos):
                        selected = i
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for i in range(N):
                    if tile_rect(i).collidepoint(ev.pos):
                        return i

        screen.fill(COLORS["panel_bg"])
        title = font_large.render("Select Map", True, COLORS["blue_ui"])
        screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 28))

        for i, surf in enumerate(thumbs):
            r = tile_rect(i)
            screen.blit(surf, r.topleft)
            lbl = font_small.render(labels[i], True, COLORS["panel_fg"])
            screen.blit(lbl, (r.x + THUMB // 2 - lbl.get_width() // 2, r.y + THUMB + 2))
            border_color = COLORS["highlight"] if i == selected else COLORS["gray"]
            pygame.draw.rect(screen, border_color, r, 3 if i == selected else 1)

        hint = font_small.render(
            "Arrows / mouse to browse   •   Enter / click to select   •   ESC = back",
            True, COLORS["gray"],
        )
        screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 36))
        pygame.display.flip()
        clock.tick(30)


def screen_select_map(screen, clock):
    """3-option map source menu.

    Returns (world_map, spawn_rc, target_rc):
      - Random / ESC  →  (None, None, None)
      - Val map       →  (tensor [H,W], None, None)
      - Behavioral    →  (tensor [H,W], (r,c), (r,c))   ← positions pre-set
    """
    compiled    = EnvConfig().compile_terrain("cpu")
    val_exists  = VAL_MAPS_PATH.exists()
    beh_exists  = BEHAVIORAL_MAPS_PATH.exists()

    font_large = pygame.font.Font(None, 52)
    font_med   = pygame.font.Font(None, 34)
    font_small = pygame.font.Font(None, 22)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:    return None, None, None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return None, None, None
                if ev.key == pygame.K_r:      return None, None, None

                if ev.key == pygame.K_v and val_exists:
                    data = torch.load(str(VAL_MAPS_PATH), map_location="cpu", weights_only=True)
                    maps = data["maps"]
                    idx  = _screen_pick_map_grid(screen, clock, maps, compiled)
                    if idx is None:
                        continue   # back to source menu
                    return maps[idx], None, None

                if ev.key == pygame.K_b and beh_exists:
                    data = torch.load(str(BEHAVIORAL_MAPS_PATH), map_location="cpu",
                                      weights_only=False)
                    maps, spawns, targets, names = (
                        data["maps"], data["spawns"], data["targets"], data["names"]
                    )
                    idx = _screen_pick_map_grid(screen, clock, maps, compiled, labels=names)
                    if idx is None:
                        continue
                    return maps[idx], spawns[idx], targets[idx]

        screen.fill(COLORS["panel_bg"])
        title = font_large.render("Select Map", True, COLORS["blue_ui"])
        screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 130))

        y = 250
        for key, label, desc, avail in [
            ("R", "Random Map",   "Procedurally generated",          True),
            ("V", "Val Maps",     f"{VAL_MAPS_PATH.name}  (16 maps)", val_exists),
            ("B", "Behavioral",   f"{BEHAVIORAL_MAPS_PATH.name}  (16 maps)", beh_exists),
        ]:
            key_col = COLORS["blue_ui"] if avail else COLORS["gray"]
            lbl_col = COLORS["white"]   if avail else COLORS["gray"]
            screen.blit(font_med.render(f"[{key}]",       True, key_col), (WINDOW_W//2 - 200, y))
            screen.blit(font_med.render(f"  {label}",     True, lbl_col), (WINDOW_W//2 - 160, y))
            screen.blit(font_small.render(desc,           True, COLORS["gray"]), (WINDOW_W//2 - 150, y + 30))
            y += 90

        hint = font_small.render("ESC = back to main menu", True, COLORS["gray"])
        screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 50))
        pygame.display.flip()
        clock.tick(30)


# ---------------------------------------------------------------------------
# Screen: checkpoint selection (agent only)
# ---------------------------------------------------------------------------

def screen_select_checkpoint(screen, clock):
    """Returns ckpt_path or None."""
    checkpoints = discover_checkpoints()
    if not checkpoints:
        print("No checkpoints found in artifacts/. Train a model first.")
        return None

    font_large = pygame.font.Font(None, 40)
    font_med   = pygame.font.Font(None, 28)
    font_small = pygame.font.Font(None, 22)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:    return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return None
                idx = ev.key - pygame.K_1
                if 0 <= idx < len(checkpoints):
                    return checkpoints[idx][1]

        screen.fill(COLORS["panel_bg"])
        title = font_large.render("Select Checkpoint", True, COLORS["blue_ui"])
        screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 40))

        y = 120
        for i, (run_id, ckpt) in enumerate(checkpoints):
            if i >= 9: break
            line = font_med.render(
                f"  {i+1}  —  {run_id}  /  {os.path.basename(ckpt)}",
                True, COLORS["panel_fg"],
            )
            screen.blit(line, (80, y)); y += 36

        hint = font_small.render("Press 1-9 to select  •  ESC = back", True, COLORS["gray"])
        screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 60))
        pygame.display.flip()
        clock.tick(30)


# ---------------------------------------------------------------------------
# Screen: position picker (agent only)
# ---------------------------------------------------------------------------

def screen_pick_positions(screen, clock, env_config, world_map=None,
                          default_spawn=None, default_target=None):
    """Returns (spawn_rc, target_rc) or None."""
    env        = Islands(env_config,
                         world_maps=world_map.unsqueeze(0) if world_map is not None else None)
    _compiled  = env.compiled
    world_map  = env.world_map
    map_surf   = heightmap_to_surface(world_map, MAP_DISPLAY_SIZE, _compiled)
    map_size   = world_map.shape[0]

    font_large = pygame.font.Font(None, 36)
    font_med   = pygame.font.Font(None, 26)
    font_small = pygame.font.Font(None, 22)
    MAP_X, MAP_Y = 20, 60

    spawn  = default_spawn
    target = default_target

    def w2s(r, c):
        s = MAP_DISPLAY_SIZE / map_size
        return int(c * s) + MAP_X, int(r * s) + MAP_Y

    def s2w(sx, sy):
        s = MAP_DISPLAY_SIZE / map_size
        return (max(0, min(int((sy - MAP_Y) / s), map_size - 1)),
                max(0, min(int((sx - MAP_X) / s), map_size - 1)))

    def is_land(r, c):
        return world_map[r, c].item() > _compiled.land_threshold

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:    return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:  return None
                if ev.key == pygame.K_r:       spawn = target = None
                if ev.key == pygame.K_RETURN and spawn and target:
                    return spawn, target
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if MAP_X <= mx < MAP_X + MAP_DISPLAY_SIZE and MAP_Y <= my < MAP_Y + MAP_DISPLAY_SIZE:
                    r, c = s2w(mx, my)
                    if is_land(r, c):
                        if   spawn  is None: spawn  = (r, c)
                        elif target is None: target = (r, c)
                        else:                spawn = (r, c); target = None

        screen.fill(COLORS["panel_bg"])

        msg = ("Click to place SPAWN (red)" if spawn is None
               else "Click to place TARGET (green)" if target is None
               else "Press ENTER to start  •  R to reset")
        screen.blit(font_large.render(msg, True, COLORS["blue_ui"]), (MAP_X, 15))
        screen.blit(map_surf, (MAP_X, MAP_Y))
        pygame.draw.rect(screen, COLORS["white"], (MAP_X, MAP_Y, MAP_DISPLAY_SIZE, MAP_DISPLAY_SIZE), 1)

        if spawn:
            sx, sy = w2s(*spawn)
            pygame.draw.circle(screen, COLORS["player"], (sx, sy), 7)
            pygame.draw.circle(screen, COLORS["white"],  (sx, sy), 7, 1)
        if target:
            draw_star(screen, *w2s(*target), r_outer=9, r_inner=4)

        panel_x = MAP_X + MAP_DISPLAY_SIZE + 30
        py = MAP_Y
        screen.blit(font_med.render("TERRAIN LEGEND", True, COLORS["panel_fg"]), (panel_x, py)); py += 30
        for lev in range(_compiled.num_terrains):
            col  = terrain_color(lev, _compiled)
            cost = _compiled.move_costs[lev].item()
            pygame.draw.rect(screen, col,          (panel_x, py, 14, 14))
            pygame.draw.rect(screen, COLORS["white"],(panel_x, py, 14, 14), 1)
            screen.blit(font_small.render(
                f"  {_compiled.terrain_names[lev].capitalize()} (cost {cost:.1f})",
                True, COLORS["panel_fg"]), (panel_x + 18, py))
            py += 20

        py += 20
        if spawn:
            screen.blit(font_small.render(f"Spawn:  ({spawn[0]}, {spawn[1]})",
                        True, COLORS["player"]),  (panel_x, py)); py += 22
        if target:
            screen.blit(font_small.render(f"Target: ({target[0]}, {target[1]})",
                        True, COLORS["target"]),  (panel_x, py))

        screen.blit(font_small.render("R = reset  •  ESC = back  •  Click only on land",
                    True, COLORS["gray"]), (MAP_X, WINDOW_H - 30))
        pygame.display.flip()
        clock.tick(30)


# ---------------------------------------------------------------------------
# Screen: AI playback
# ---------------------------------------------------------------------------

def screen_ai_playback(screen, clock, ckpt_path, spawn_rc, target_rc, world_map=None):
    """Returns 'quit', 'reset', or 'menu'."""
    device = "cpu"
    model  = load_actor_critic(ckpt_path, device)
    print(f"Loaded model from {ckpt_path}")

    env_config = EnvConfig(
        map_generation=MapGenConfig(seed=42),
        minimap=MinimapConfig(
            max_ray=ENV_MM_MAX_RAY,
            occlude=ENV_MM_OCCLUDE,
            clear_tolerance=ENV_MM_CLR_TOL,
        ),
        custom_map=CustomMapConfig(
            spawn_r=spawn_rc[0], spawn_c=spawn_rc[1],
            target_r=target_rc[0], target_c=target_rc[1],
        ),
    )
    env = Islands(env_config, world_maps=world_map.unsqueeze(0) if world_map is not None else None)
    _compiled = env.compiled
    state, target_pos = env.reset(batch_size=1, seed=42)

    map_size = env.world_map.shape[0]
    base_rgb = _fast_colorize(env.world_map.numpy(), _compiled)

    seen_mask = np.zeros((map_size, map_size), dtype=bool)
    vis_counts = np.zeros((map_size, map_size), dtype=np.int32)
    _D = 2 * env_config.minimap_max_ray + 1
    _dy, _dx = np.meshgrid(np.arange(_D) - env_config.minimap_max_ray,
                           np.arange(_D) - env_config.minimap_max_ray, indexing="ij")

    def update_seen(st):
        vis = st.minimap[0, 2].numpy()
        cy, cx = int(st.position[0, 0].item()), int(st.position[0, 1].item())
        rows = np.clip(cy + _dy, 0, map_size - 1)
        cols = np.clip(cx + _dx, 0, map_size - 1)
        visible = vis > 0.5
        seen_mask[rows[visible], cols[visible]] = True
        vis_counts[rows[visible], cols[visible]] += 1

    update_seen(state)

    u0 = env_config.init_hp + env_config.init_resources
    drawdown_sq_sum = 0.0
    risk_count = 0
    font_large = pygame.font.Font(None, 36)
    font_med   = pygame.font.Font(None, 26)
    font_small = pygame.font.Font(None, 22)
    MAP_X, MAP_Y = 20, 60

    trajectory     = [tuple(state.position[0].cpu().tolist())]
    hp_history     = [state.hp[0].item()]
    res_history    = [state.resources[0].item()]
    step_count     = 0
    game_over      = won = paused = False
    frames_per_step = 12
    frame_counter  = 0
    last_action    = None

    def w2s(r, c):
        s = MAP_DISPLAY_SIZE / map_size
        return int(c * s) + MAP_X, int(r * s) + MAP_Y

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:   return "quit"
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:                              return "menu"
                if ev.key == pygame.K_r:                                   return "reset"
                if ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    frames_per_step = max(1, frames_per_step - 2)
                if ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    frames_per_step = min(120, frames_per_step + 2)
                if ev.key == pygame.K_p:
                    paused = not paused

        if not game_over and not paused:
            frame_counter += 1
            if frame_counter >= frames_per_step:
                frame_counter = 0
                obs = build_obs(state, env_config, _compiled)
                with torch.no_grad():
                    action = model.get_deterministic_action(obs)
                last_action = action.item()

                res   = state.resources[0].item()
                hp    = state.hp[0].item()
                u_t = res + hp
                drawdown_sq_sum += (max(0.0, (u0 - u_t) / u0)) ** 2
                risk_count += 1

                result = env.step(state, action, target_pos)
                state  = result.state
                step_count += 1
                update_seen(state)
                trajectory.append(tuple(state.position[0].cpu().tolist()))
                hp_history.append(state.hp[0].item())
                res_history.append(state.resources[0].item())

                alive   = result.info["alive"][0]
                reached = result.info["reached"][0]
                if not alive:     game_over, won = True, False
                elif reached:     game_over, won = True, True
                elif step_count >= env_config.max_steps:
                    game_over, won = True, False

        # ── Draw ────────────────────────────────────────────────────────────
        screen.fill(COLORS["panel_bg"])

        speed  = f"Speed: {60 // max(frames_per_step, 1)} steps/s"
        status = "PAUSED" if paused else ("GAME OVER" if game_over else "PLAYING")
        screen.blit(font_large.render(f"AI Agent  —  {status}  —  {speed}",
                    True, COLORS["blue_ui"]), (MAP_X, 12))

        screen.blit(make_map_surface_with_fog(base_rgb, seen_mask, map_size, MAP_DISPLAY_SIZE),
                    (MAP_X, MAP_Y))

        n = len(trajectory)
        for i in range(1, n):
            t  = i / max(n - 1, 1)
            p1 = w2s(*trajectory[i - 1])
            p2 = w2s(*trajectory[i])
            pygame.draw.line(screen, (255, int(200*(1-t)), int(50*(1-t))), p1, p2, 2)

        draw_star(screen, *w2s(*target_rc), r_outer=9, r_inner=4)

        pr, pc = state.position[0].cpu().tolist()
        px, py_ = w2s(int(pr), int(pc))
        pygame.draw.circle(screen, COLORS["player"], (px, py_), 6)
        pygame.draw.circle(screen, COLORS["white"],  (px, py_), 6, 1)
        if last_action is not None:
            screen.blit(font_med.render(ACTION_NAMES.get(last_action, "?"),
                        True, COLORS["white"]), (px + 10, py_ - 10))

        pygame.draw.rect(screen, COLORS["white"],
                         (MAP_X, MAP_Y, MAP_DISPLAY_SIZE, MAP_DISPLAY_SIZE), 1)

        # Minimap
        mm_x, mm_y = MAP_X + MAP_DISPLAY_SIZE + 30, MAP_Y
        screen.blit(font_small.render("Agent view  (▲ = target)", True, COLORS["panel_fg"]),
                    (mm_x, mm_y - 16))
        mm_surf = heightmap_to_surface(state.minimap[0, 0], MINIMAP_DISPLAY_SIZE, _compiled,
                                       vis_mask=state.minimap[0, 2])

        screen.blit(mm_surf, (mm_x, mm_y))
        cx_mm, cy_mm = mm_x + MINIMAP_DISPLAY_SIZE // 2, mm_y + MINIMAP_DISPLAY_SIZE // 2
        pygame.draw.circle(screen, COLORS["player"], (cx_mm, cy_mm), 3)

        # Show target as a star on the minimap when within visibility
        target_ch = state.minimap[0, 1]  # target indicator channel
        if target_ch.any():
            ty, tx = (target_ch > 0).nonzero(as_tuple=False)[0].tolist()
            mm_diameter = target_ch.shape[0]
            mm_scale = MINIMAP_DISPLAY_SIZE / mm_diameter
            px = int(tx * mm_scale + mm_scale / 2) + mm_x
            py = int(ty * mm_scale + mm_scale / 2) + mm_y
            draw_star(screen, px, py, r_outer=5, r_inner=2)

        compass = state.compass[0]
        dyd, dxd = float(compass[0]), float(compass[1])
        mag = (dyd**2 + dxd**2)**0.5
        if mag > 1e-6:
            dyd /= mag; dxd /= mag
            ex, ey = int(cx_mm + dxd*35), int(cy_mm + dyd*35)
            pygame.draw.line(screen, (255, 220, 50), (cx_mm, cy_mm), (ex, ey), 2)
            a = math.atan2(dyd, dxd)
            for s in (math.pi/5, -math.pi/5):
                hx = int(ex + 9*math.cos(a+math.pi+s))
                hy = int(ey + 9*math.sin(a+math.pi+s))
                pygame.draw.line(screen, (255, 220, 50), (ex, ey), (hx, hy), 2)
        pygame.draw.rect(screen, COLORS["white"],
                         (mm_x, mm_y, MINIMAP_DISPLAY_SIZE, MINIMAP_DISPLAY_SIZE), 1)

        # Stats
        sx, sy_ = mm_x, mm_y + MINIMAP_DISPLAY_SIZE + 30
        s        = state
        hp_val   = s.hp[0].item(); hp_ratio = hp_val / env_config.max_hp
        res_val  = s.resources[0].item()
        risk_m   = (drawdown_sq_sum / max(risk_count, 1)) ** 0.5
        # Normalised entropy of visibility distribution
        _vc_flat = vis_counts.ravel().astype(np.float64)
        _total = _vc_flat.sum()
        if _total > 0:
            _p = _vc_flat[_vc_flat > 0] / _total
            expl_ent = -(_p * np.log(_p)).sum() / np.log(vis_counts.size)
        else:
            expl_ent = 0.0
        dist_val = (s.position[0].float() - target_pos[0].float()).abs().sum().item()

        for lbl, val, col in [
            ("HP",        f"{hp_val:.1f} / {env_config.max_hp:.0f}",
                          COLORS["green_ui"] if hp_ratio > 0.5 else (COLORS["red"] if hp_ratio < 0.3 else (255,165,0))),
            ("Resources", f"{res_val:.1f}",
                          COLORS["red"] if res_val/env_config.max_resources < 0.2 else COLORS["panel_fg"]),
            ("Time Cost", f"{s.cost[0].item():.2f}",   COLORS["panel_fg"]),
            ("Moves",     f"{step_count}",              COLORS["panel_fg"]),
            ("Risk Exp.", f"{risk_m:.3f}",
                          COLORS["red"] if risk_m > 0.5 else COLORS["panel_fg"]),
            ("Exploration", f"{expl_ent:.3f}",            COLORS["panel_fg"]),
            ("Distance",  f"{dist_val:.1f}",            COLORS["panel_fg"]),
            ("Terrain",   _compiled.terrain_names[int(s.terrain_idx[0].item())].capitalize(),
                          COLORS["panel_fg"]),
            ("Position",  f"({int(pr)}, {int(pc)})",    COLORS["panel_fg"]),
        ]:
            screen.blit(font_small.render(f"{lbl}:", True, COLORS["gray"]),  (sx, sy_))
            screen.blit(font_med.render(val, True, col), (sx + 90, sy_ - 2))
            sy_ += 28

        # ── Right panel: plot → terrain legend → controls ────────────────────
        rx  = mm_x + MINIMAP_DISPLAY_SIZE + 20   # x ≈ 840
        rw  = WINDOW_W - rx - 8                  # available width ≈ 352
        ry  = MAP_Y
        font_tiny = pygame.font.Font(None, 18)

        # HP + Resources combined real-time plot
        combined_history = [h + r for h, r in zip(hp_history, res_history)]
        PLOT_W, PLOT_H = rw, 150
        plot_surf = pygame.Surface((PLOT_W, PLOT_H))
        plot_surf.fill((15, 15, 25))

        pygame.draw.line(plot_surf, COLORS["gray"], (0, PLOT_H - 1), (PLOT_W, PLOT_H - 1), 1)
        pygame.draw.line(plot_surf, COLORS["gray"], (0, 0), (0, PLOT_H - 1), 1)

        # Dotted line at y = 100
        y100 = PLOT_H - 1 - int(100 / 200 * (PLOT_H - 2))
        for dx in range(0, PLOT_W, 8):
            pygame.draw.line(plot_surf, (200, 190, 60),
                             (dx, y100), (min(dx + 4, PLOT_W - 1), y100), 1)

        n_steps = max(len(combined_history) - 1, 1)
        def _px(step): return int(step / n_steps * (PLOT_W - 1))
        def _py(val):  return PLOT_H - 1 - int(min(max(val, 0), 200) / 200 * (PLOT_H - 2))

        pts = [(_px(i), _py(v)) for i, v in enumerate(combined_history)]
        if len(pts) >= 2:
            pygame.draw.lines(plot_surf, (70, 200, 120), False, pts, 2)

        for yval, label in [(200, "200"), (100, "100"), (0, "0")]:
            plot_surf.blit(font_tiny.render(label, True, COLORS["gray"]),
                           (2, _py(yval) - 7))

        screen.blit(font_small.render("HP + Resources", True, COLORS["panel_fg"]), (rx, ry))
        ry += 18
        screen.blit(plot_surf, (rx, ry))
        ry += PLOT_H + 12

        # Terrain legend — pixel-aligned columns
        screen.blit(font_med.render("TERRAIN", True, COLORS["blue_ui"]), (rx, ry)); ry += 21
        COL_NAME  = rx + 14          # name starts here
        COL_COST  = rx + 114         # cost column (fixed x)
        COL_DRAIN = rx + 158         # drain column (fixed x)
        for lev in range(_compiled.num_terrains):
            tc      = terrain_color(lev, _compiled)
            name    = _compiled.terrain_names[lev].capitalize()
            cost    = _compiled.move_costs[lev].item()
            drain   = int(round(_compiled.res_rate[lev].item()))
            drain_s = f"+{drain}" if drain >= 0 else str(drain)
            pygame.draw.rect(screen, tc,              (rx, ry + 1, 11, 11))
            pygame.draw.rect(screen, COLORS["white"], (rx, ry + 1, 11, 11), 1)
            screen.blit(font_tiny.render(name,              True, COLORS["panel_fg"]), (COL_NAME,  ry))
            screen.blit(font_tiny.render(f"{cost:.2f}",     True, COLORS["gray"]),     (COL_COST,  ry))
            screen.blit(font_tiny.render(f"{drain_s}/step", True, COLORS["gray"]),     (COL_DRAIN, ry))
            ry += 19
        ry += 10

        # Controls (bottom)
        screen.blit(font_med.render("CONTROLS", True, COLORS["blue_ui"]), (rx, ry)); ry += 22
        for ctrl in ["+/-  Speed", "P    Pause", "R    Reset", "ESC  Menu"]:
            screen.blit(font_small.render(ctrl, True, COLORS["gray"]), (rx, ry)); ry += 19

        if game_over:
            overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (0, 0))
            msg   = ("TARGET REACHED!" if won
                     else "AGENT DIED" if s.hp[0].item() <= 0 else "MAX STEPS REACHED")
            color = COLORS["green_ui"] if won else COLORS["red"]
            cy_ov = WINDOW_H // 2
            ms = font_large.render(msg, True, color)
            screen.blit(ms, ms.get_rect(center=(WINDOW_W//2, cy_ov-50)))
            for k, line in enumerate([
                f"Moves: {step_count}  •  Time cost: {s.cost[0].item():.2f}",
                f"Risk exposure: {risk_m:.3f}  •  Exploration: {expl_ent:.3f}",
            ]):
                surf = font_med.render(line, True, COLORS["white"])
                screen.blit(surf, surf.get_rect(center=(WINDOW_W//2, cy_ov+k*32)))
            hs = font_small.render("R = try again  •  ESC = menu", True, COLORS["gray"])
            screen.blit(hs, hs.get_rect(center=(WINDOW_W//2, cy_ov+80)))

        pygame.display.flip()
        clock.tick(60)


# ---------------------------------------------------------------------------
# Human play
# ---------------------------------------------------------------------------

class HumanDemo:
    """Interactive human play session on a single map."""

    MAP_DISPLAY  = 400
    MINIMAP_SIZE = 240

    def __init__(self, screen, world_map: torch.Tensor | None = None):
        self.screen    = screen
        self._world_map = world_map      # None → generate random each reset
        self.font_small  = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large  = pygame.font.Font(None, 36)
        self._reset()

    def _reset(self):
        seed = torch.randint(1, 10000, (1,)).item()
        config = EnvConfig(
            map_generation=MapGenConfig(seed=seed),
            minimap=MinimapConfig(max_ray=15, occlude=True, clear_tolerance=0.1),
        )
        world_maps = (self._world_map.unsqueeze(0)
                      if self._world_map is not None else None)
        self.env = Islands(config, world_maps=world_maps)
        self._compiled = self.env.compiled

        # Ensure spawn and target are far enough apart
        while True:
            self.state, self.target_pos = self.env.reset(batch_size=1, seed=seed)
            dist = (self.state.position[0].float() - self.target_pos[0].float()).abs().sum().item()
            if dist >= config.size * 0.3:
                break
            seed += 1

        H = W = config.size
        self.seen_mask = np.zeros((H, W), dtype=bool)
        self._vis_counts = np.zeros((H, W), dtype=np.int32)
        self._update_seen()

        wm = self.env.world_map.numpy()
        self._base_rgb = _fast_colorize(wm, self._compiled)   # [H, W, 3]

        self.game_over   = False
        self.won         = False
        self.moves_count = 0
        self._u0 = self.env.config.init_hp + self.env.config.init_resources
        self._drawdown_sq_sum = 0.0
        self._risk_count = 0
        self._hp_history  = [self.state.hp[0].item()]
        self._res_history = [self.state.resources[0].item()]

    def _update_seen(self):
        vis = self.state.minimap[0, 2].numpy()
        mr  = self.env.config.minimap_max_ray
        D   = 2 * mr + 1
        pos = self.state.position[0].numpy()
        cy, cx = int(pos[0]), int(pos[1])
        H, W   = self.seen_mask.shape
        dy_g, dx_g = np.meshgrid(np.arange(D)-mr, np.arange(D)-mr, indexing="ij")
        rows = np.clip(cy + dy_g, 0, H-1)
        cols = np.clip(cx + dx_g, 0, W-1)
        visible = vis > 0.5
        self.seen_mask[rows[visible], cols[visible]] = True
        self._vis_counts[rows[visible], cols[visible]] += 1

    def _move(self, action):
        if self.game_over:
            return
        res   = self.state.resources[0].item()
        hp    = self.state.hp[0].item()
        u_t = res + hp
        self._drawdown_sq_sum += (max(0.0, (self._u0 - u_t) / self._u0)) ** 2
        self._risk_count += 1

        result = self.env.step(self.state, action.to(self.env._device), self.target_pos)
        self.state = result.state
        self.moves_count += 1
        self._hp_history.append(self.state.hp[0].item())
        self._res_history.append(self.state.resources[0].item())
        self._update_seen()

        if not result.info["alive"][0]:
            self.game_over, self.won = True, False
        elif result.info["reached"][0]:
            self.game_over, self.won = True, True

    # ── Drawing ─────────────────────────────────────────────────────────────

    def _draw_map(self):
        MAP_X, MAP_Y = 10, 10
        surf = make_map_surface_with_fog(
            self._base_rgb, self.seen_mask,
            self.env.world_map.shape[0], self.MAP_DISPLAY,
        )
        self.screen.blit(surf, (MAP_X, MAP_Y))
        scale = self.MAP_DISPLAY / self.env.world_map.shape[0]

        pp = self.state.position[0]
        pygame.draw.circle(self.screen, COLORS["player"],
                           (int(pp[1]*scale)+MAP_X, int(pp[0]*scale)+MAP_Y), 5)

        tp = self.target_pos[0]
        draw_star(self.screen, int(tp[1]*scale)+MAP_X, int(tp[0]*scale)+MAP_Y,
                  r_outer=8, r_inner=3)

        pygame.draw.rect(self.screen, COLORS["black"],
                         (MAP_X, MAP_Y, self.MAP_DISPLAY, self.MAP_DISPLAY), 2)

    def _draw_minimap(self):
        mm_x, mm_y = self.MAP_DISPLAY + 30, 30
        surf = heightmap_to_surface(self.state.minimap[0, 0],
                                     self.MINIMAP_SIZE, self._compiled,
                                     vis_mask=self.state.minimap[0, 2])

        mm_rect = pygame.Rect(mm_x, mm_y, self.MINIMAP_SIZE, self.MINIMAP_SIZE)
        self.screen.blit(surf, mm_rect.topleft)

        compass = self.state.compass[0]
        cx_mm, cy_mm = mm_rect.center
        dyd, dxd = float(compass[0]), float(compass[1])
        mag = (dyd**2 + dxd**2)**0.5
        if mag > 1e-6:
            dyd /= mag; dxd /= mag
            ex, ey = int(cx_mm + dxd*32), int(cy_mm + dyd*32)
            pygame.draw.line(self.screen, (255, 220, 50), (cx_mm, cy_mm), (ex, ey), 2)
            a = math.atan2(dyd, dxd)
            for s in (math.pi/5, -math.pi/5):
                hx = int(ex + 8*math.cos(a+math.pi+s))
                hy = int(ey + 8*math.sin(a+math.pi+s))
                pygame.draw.line(self.screen, (255, 220, 50), (ex, ey), (hx, hy), 2)
        pygame.draw.circle(self.screen, COLORS["player"], mm_rect.center, 3)

        # Show target as a star on the minimap when within visibility
        target_ch = self.state.minimap[0, 1]
        if target_ch.any():
            ty, tx = (target_ch > 0).nonzero(as_tuple=False)[0].tolist()
            mm_diameter = target_ch.shape[0]
            mm_scale = self.MINIMAP_SIZE / mm_diameter
            # blit_array transposes (row,col) → (x,y), so screen x=col, y=row
            px = int(tx * mm_scale + mm_scale / 2) + mm_x
            py = int(ty * mm_scale + mm_scale / 2) + mm_y
            draw_star(self.screen, px, py, r_outer=5, r_inner=2)

        pygame.draw.rect(self.screen, COLORS["black"], mm_rect, 2)
        self.screen.blit(self.font_small.render("Agent view  (▲ = target)", True, COLORS["black"]),
                         (mm_x, mm_y - 16))

    def _draw_ui(self):
        ui_x = self.MAP_DISPLAY + 30
        ui_y = self.MINIMAP_SIZE + 60
        s, ec = self.state, self.env.config
        hp_r   = s.hp[0] / ec.max_hp
        res_r  = s.resources[0] / ec.max_resources
        risk_m = (self._drawdown_sq_sum / max(self._risk_count, 1)) ** 0.5
        expl_ent = self._compute_exploration()

        for text, font, color in [
            ("Live Stats",                                       self.font_medium, COLORS["black"]),
            (f"Moves:     {self.moves_count}",                   self.font_small,  COLORS["black"]),
            (f"Time Cost: {s.cost[0]:.2f}",                      self.font_small,  COLORS["black"]),
            (f"HP:        {s.hp[0]:.1f} / {ec.max_hp:.0f}",      self.font_small,
                COLORS["red"] if hp_r < 0.3 else ((255,165,0) if hp_r < 0.6 else COLORS["green_ui"])),
            (f"Resources: {s.resources[0]:.1f} / {ec.max_resources:.0f}", self.font_small,
                COLORS["red"] if res_r < 0.2 else COLORS["black"]),
            (f"Risk Exp:  {risk_m:.3f}",                         self.font_small,
                COLORS["red"] if risk_m > 0.5 else COLORS["black"]),
            (f"Exploration: {expl_ent:.3f}",                       self.font_small,  COLORS["black"]),
            ("",                                                 self.font_small,  COLORS["black"]),
            ("Position",                                         self.font_medium, COLORS["black"]),
            (f"Pos:    ({s.position[0][0]}, {s.position[0][1]})", self.font_small, COLORS["black"]),
            (f"Target: ({self.target_pos[0][0]}, {self.target_pos[0][1]})", self.font_small, COLORS["black"]),
            (f"Dist:   {(s.position[0].float()-self.target_pos[0].float()).abs().sum():.1f}",
                self.font_small, COLORS["black"]),
            (f"Terrain: {self._compiled.terrain_names[int(s.terrain_idx[0])]}", self.font_small, COLORS["black"]),
        ]:
            if text:
                self.screen.blit(font.render(text, True, color), (ui_x, ui_y))
            ui_y += 26 if font == self.font_medium else 20

    def _draw_hp_res_plot(self, x, y, w, h):
        """Draw HP + Resources combined real-time plot (matches AI agent panel)."""
        font_tiny = pygame.font.Font(None, 18)
        combined = [hp + res for hp, res in zip(self._hp_history, self._res_history)]

        plot_surf = pygame.Surface((w, h))
        plot_surf.fill((15, 15, 25))
        pygame.draw.line(plot_surf, COLORS["gray"], (0, h - 1), (w, h - 1), 1)
        pygame.draw.line(plot_surf, COLORS["gray"], (0, 0), (0, h - 1), 1)

        # Dotted line at y = 100
        y100 = h - 1 - int(100 / 200 * (h - 2))
        for dx in range(0, w, 8):
            pygame.draw.line(plot_surf, (200, 190, 60),
                             (dx, y100), (min(dx + 4, w - 1), y100), 1)

        n_steps = max(len(combined) - 1, 1)
        def _px(step): return int(step / n_steps * (w - 1))
        def _py(val):  return h - 1 - int(min(max(val, 0), 200) / 200 * (h - 2))

        pts = [(_px(i), _py(v)) for i, v in enumerate(combined)]
        if len(pts) >= 2:
            pygame.draw.lines(plot_surf, (70, 200, 120), False, pts, 2)

        for yval, label in [(200, "200"), (100, "100"), (0, "0")]:
            plot_surf.blit(font_tiny.render(label, True, COLORS["gray"]),
                           (2, _py(yval) - 7))

        self.screen.blit(self.font_small.render("HP + Resources", True, COLORS["black"]),
                         (x, y))
        self.screen.blit(plot_surf, (x, y + 18))

    def _draw_right_panel(self):
        right_x = self.MAP_DISPLAY + self.MINIMAP_SIZE + 55
        y = 10

        # HP + Resources plot
        plot_w = WINDOW_W - right_x - 8
        plot_h = 150
        self._draw_hp_res_plot(right_x, y, plot_w, plot_h)
        y += plot_h + 30

        self.screen.blit(self.font_medium.render("Controls", True, COLORS["black"]),
                         (right_x, y)); y += 26
        for line in ["WASD / Arrows: Move", "Space: Stay", "R: New game", "ESC: Menu"]:
            self.screen.blit(self.font_small.render(line, True, COLORS["gray"]),
                             (right_x, y)); y += 20
        y += 20
        self.screen.blit(self.font_medium.render("Terrain Legend", True, COLORS["black"]),
                         (right_x, y)); y += 26
        for i, name in enumerate(self._compiled.terrain_names):
            col  = terrain_color(i, self._compiled)
            cost = self._compiled.move_costs[i].item()
            pygame.draw.rect(self.screen, col,             (right_x, y, 14, 14))
            pygame.draw.rect(self.screen, COLORS["black"], (right_x, y, 14, 14), 1)
            self.screen.blit(
                self.font_small.render(f"{name.capitalize()} (cost {cost:.1f})",
                                       True, COLORS["black"]),
                (right_x + 18, y)); y += 20

    def _compute_exploration(self):
        _vc = self._vis_counts.ravel().astype(np.float64)
        _tot = _vc.sum()
        if _tot > 0:
            _p = _vc[_vc > 0] / _tot
            return -(_p * np.log(_p)).sum() / np.log(self._vis_counts.size)
        return 0.0

    def _draw_game_over(self):
        overlay = pygame.Surface((WINDOW_W, WINDOW_H))
        overlay.set_alpha(180); overlay.fill(COLORS["black"])
        self.screen.blit(overlay, (0, 0))
        if self.won:
            title, color = "VICTORY!", COLORS["green_ui"]
            lines = [
                f"Reached target in {self.moves_count} moves",
                f"Final time cost: {self.state.cost[0]:.2f}",
                f"Risk exposure (ρ): {(self._drawdown_sq_sum / max(self._risk_count, 1)) ** 0.5:.3f}",
                f"Exploration (E): {self._compute_exploration():.3f}",
            ]
        else:
            title, color = "GAME OVER", COLORS["red"]
            lines = [
                f"HP reached zero after {self.moves_count} moves",
                f"Time cost: {self.state.cost[0]:.2f}",
                f"Risk exposure (ρ): {(self._drawdown_sq_sum / max(self._risk_count, 1)) ** 0.5:.3f}",
            ]
        cy = WINDOW_H // 2
        ts = self.font_large.render(title, True, color)
        self.screen.blit(ts, ts.get_rect(center=(WINDOW_W//2, cy-60)))
        for k, line in enumerate(lines):
            surf = self.font_medium.render(line, True, COLORS["white"])
            self.screen.blit(surf, surf.get_rect(center=(WINDOW_W//2, cy-10+k*32)))
        rs = self.font_small.render("R = new game  •  ESC = menu", True, COLORS["white"])
        self.screen.blit(rs, rs.get_rect(center=(WINDOW_W//2, cy+130)))

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self, clock):
        """Run the human play loop. Returns 'quit' or 'menu'."""
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return "quit"
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:  return "menu"
                    if ev.key == pygame.K_r:       self._reset()
                    elif not self.game_over:
                        action = None
                        if   ev.key in (pygame.K_UP,    pygame.K_w): action = torch.tensor([ACTIONS["up"]])
                        elif ev.key in (pygame.K_DOWN,  pygame.K_s): action = torch.tensor([ACTIONS["down"]])
                        elif ev.key in (pygame.K_LEFT,  pygame.K_a): action = torch.tensor([ACTIONS["left"]])
                        elif ev.key in (pygame.K_RIGHT, pygame.K_d): action = torch.tensor([ACTIONS["right"]])
                        elif ev.key == pygame.K_SPACE:               action = torch.tensor([ACTIONS["stay"]])
                        if action is not None:
                            self._move(action)

            self.screen.fill(COLORS["white"])
            self._draw_map()
            self._draw_minimap()
            self._draw_ui()
            self._draw_right_panel()
            if self.game_over:
                self._draw_game_over()
            pygame.display.flip()
            clock.tick(60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Cogniland")
    clock = pygame.time.Clock()

    while True:
        mode = screen_main_menu(screen, clock)
        if mode is None:
            break

        # Map selection (shared between both modes).
        # Returns (world_map, fixed_spawn, fixed_target):
        #   random / ESC → (None, None, None)
        #   val map      → (tensor, None, None)
        #   behavioral   → (tensor, (r,c), (r,c))
        world_map, fixed_spawn, fixed_target = screen_select_map(screen, clock)

        if mode == "human":
            demo   = HumanDemo(screen, world_map=world_map)
            result = demo.run(clock)
            if result == "quit":
                break

        elif mode == "agent":
            ckpt_path = screen_select_checkpoint(screen, clock)
            if ckpt_path is None:
                continue   # back to main menu

            env_config = EnvConfig(map_generation=MapGenConfig(seed=42))

            while True:
                if fixed_spawn is not None and fixed_target is not None:
                    # Behavioral map: positions are pre-set, skip the picker
                    spawn_rc, target_rc = fixed_spawn, fixed_target
                else:
                    result = screen_pick_positions(screen, clock, env_config,
                                                   world_map=world_map)
                    if result is None:
                        break   # back to main menu
                    spawn_rc, target_rc = result

                outcome = screen_ai_playback(
                    screen, clock, ckpt_path, spawn_rc, target_rc,
                    world_map=world_map,
                )
                if outcome == "quit":
                    pygame.quit()
                    sys.exit()
                elif outcome == "menu":
                    break
                # "reset" → back to position picking on same map

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
