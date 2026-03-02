#!/usr/bin/env python3
"""AI Demo — Watch a trained agent navigate the island step-by-step.

Usage:
    python ai_demo.py

Flow:
    1. Select a checkpoint from artifacts/
    2. Click on the map to place spawn (red) then target (green)
    3. Press Enter to start — watch the AI play in real time

Controls during playback:
    +/=     Speed up
    -       Slow down
    P       Pause / resume
    R       Reset (back to position selection)
    ESC     Quit
"""

import os
import re
import sys

import pygame
import torch

from cogniland.env.constants import (
    ACTIONS,
    NUM_ACTIONS,
    TERRAIN_LEVELS,
    TERRAIN_VISIBILITY,
    VISIBILITY_RANGES,
    palette,
)
from cogniland.env.core import compute_minimap_batch, compute_terrain_levels
from cogniland.env.islands import Islands
from cogniland.env.types import EnvConfig, EnvState

# ---------------------------------------------------------------------------
# Load model (lightweight — only needs ActorCritic, no Hydra / PPOAgent)
# ---------------------------------------------------------------------------
from cogniland.models.ppo import ActorCritic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_W, WINDOW_H = 1200, 800
MAP_DISPLAY_SIZE = 550          # pixels for the big map
MINIMAP_DISPLAY_SIZE = 180
ACTION_NAMES = {0: "↑", 1: "↓", 2: "→", 3: "←", 4: "•"}

# Default architecture (matches ppo.yaml)
DEFAULT_SCALAR_DIM = 7
DEFAULT_MINIMAP_CHANNELS = 2
DEFAULT_HIDDEN_DIM = 128
DEFAULT_ACTION_DIM = NUM_ACTIONS
DEFAULT_MINIMAP_MAX_RAY = 10

# Colors
COLORS = {k: tuple(v) for k, v in palette.items()}
COLORS.update({
    "player":   (255,  50,  50),
    "target":   ( 50, 255,  50),
    "trail":    (255, 200,  50),
    "black":    (  0,   0,   0),
    "white":    (255, 255, 255),
    "gray":     (128, 128, 128),
    "dark_gray":( 40,  40,  40),
    "red":      (255,   0,   0),
    "green_ui": (  0, 255,   0),
    "blue_ui":  ( 80, 140, 255),
    "panel_bg": ( 25,  25,  35),
    "panel_fg": (200, 200, 210),
})


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def discover_checkpoints(artifacts_dir="artifacts"):
    """Return list of (run_id, ckpt_path) sorted by run_id."""
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
    """Load ActorCritic weights from a .pt checkpoint."""
    minimap_size = 2 * DEFAULT_MINIMAP_MAX_RAY + 1
    model = ActorCritic(
        scalar_dim=DEFAULT_SCALAR_DIM,
        minimap_channels=DEFAULT_MINIMAP_CHANNELS,
        minimap_size=minimap_size,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        action_dim=DEFAULT_ACTION_DIM,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_obs(state: EnvState, minimap_max_ray: int):
    """Replicate BatchedIslandEnv.get_obs() for a single-batch state."""
    s = state
    vis_range = TERRAIN_VISIBILITY.to(s.terrain_lev.device)[s.terrain_lev.long()].float()
    vis_norm = vis_range / minimap_max_ray
    scalars = torch.stack([
        s.compass[:, 0],
        s.compass[:, 1],
        s.terrain_lev,
        s.terrain_clock,
        s.resources,
        s.hp,
        vis_norm,
    ], dim=1)
    return {"scalars": scalars, "minimap": s.minimap}


def terrain_level_for_height(height_val):
    for level in range(9):
        if height_val <= TERRAIN_LEVELS[level]["threshold"]:
            return level
    return 8


def terrain_color(level):
    info = TERRAIN_LEVELS.get(int(level))
    if info is None:
        return COLORS["white"]
    return COLORS.get(info["color"], COLORS["white"])


def heightmap_to_surface(world_map, display_size):
    """Render a 2D heightmap tensor → pygame.Surface."""
    H, W = world_map.shape[:2]
    surf = pygame.Surface((W, H))
    for y in range(H):
        for x in range(W):
            h = world_map[y, x].item() if world_map.dim() == 2 else world_map[y, x, 0].item()
            lev = terrain_level_for_height(h)
            surf.set_at((x, y), terrain_color(lev))
    return pygame.transform.scale(surf, (display_size, display_size))


# ───────────────────────────────────────────────────────────────────────────
# Screens
# ───────────────────────────────────────────────────────────────────────────

def screen_select_checkpoint(screen, clock):
    """Phase 1: let user pick a checkpoint. Returns ckpt_path or None."""
    checkpoints = discover_checkpoints()
    if not checkpoints:
        print("No checkpoints found in artifacts/. Train a model first.")
        return None

    font_large = pygame.font.Font(None, 40)
    font_med   = pygame.font.Font(None, 28)
    font_small = pygame.font.Font(None, 22)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return None
                idx = ev.key - pygame.K_1  # 0-based
                if 0 <= idx < len(checkpoints):
                    return checkpoints[idx][1]

        screen.fill(COLORS["panel_bg"])

        title = font_large.render("Select Checkpoint", True, COLORS["blue_ui"])
        screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 40))

        y = 120
        for i, (run_id, ckpt) in enumerate(checkpoints):
            if i >= 9:
                break
            ckpt_name = os.path.basename(ckpt)
            color = COLORS["panel_fg"]
            line = font_med.render(f"  {i+1}  —  {run_id}  /  {ckpt_name}", True, color)
            screen.blit(line, (80, y))
            y += 36

        hint = font_small.render("Press 1-9 to select  •  ESC to quit", True, COLORS["gray"])
        screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, WINDOW_H - 60))

        pygame.display.flip()
        clock.tick(30)


def screen_pick_positions(screen, clock, env_config):
    """Phase 2: render map, let user click spawn then target.

    Returns (spawn_rc, target_rc) as integer (row, col) tuples, or None on quit.
    """
    env = Islands(env_config)
    world_map = env.world_map
    map_surface = heightmap_to_surface(world_map, MAP_DISPLAY_SIZE)
    map_size = world_map.shape[0]

    font_large = pygame.font.Font(None, 36)
    font_med   = pygame.font.Font(None, 26)
    font_small = pygame.font.Font(None, 22)

    MAP_X, MAP_Y = 20, 60

    spawn = None   # (row, col) in world coords
    target = None

    def world_to_screen(r, c):
        scale = MAP_DISPLAY_SIZE / map_size
        return int(c * scale) + MAP_X, int(r * scale) + MAP_Y

    def screen_to_world(sx, sy):
        scale = MAP_DISPLAY_SIZE / map_size
        c = int((sx - MAP_X) / scale)
        r = int((sy - MAP_Y) / scale)
        return max(0, min(r, map_size - 1)), max(0, min(c, map_size - 1))

    def is_land(r, c):
        from cogniland.env.constants import TERRAIN_THRESHOLDS
        return world_map[r, c].item() > TERRAIN_THRESHOLDS[2].item()

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return None
                if ev.key == pygame.K_r:
                    spawn, target = None, None
                if ev.key == pygame.K_RETURN and spawn is not None and target is not None:
                    return spawn, target
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if MAP_X <= mx < MAP_X + MAP_DISPLAY_SIZE and MAP_Y <= my < MAP_Y + MAP_DISPLAY_SIZE:
                    r, c = screen_to_world(mx, my)
                    if is_land(r, c):
                        if spawn is None:
                            spawn = (r, c)
                        elif target is None:
                            target = (r, c)
                        else:
                            # allow re-picking
                            spawn = (r, c)
                            target = None

        screen.fill(COLORS["panel_bg"])

        # Title
        if spawn is None:
            msg = "Click to place SPAWN (red)"
        elif target is None:
            msg = "Click to place TARGET (green)"
        else:
            msg = "Press ENTER to start  •  R to reset"
        title = font_large.render(msg, True, COLORS["blue_ui"])
        screen.blit(title, (MAP_X, 15))

        # Map
        screen.blit(map_surface, (MAP_X, MAP_Y))
        pygame.draw.rect(screen, COLORS["white"], (MAP_X, MAP_Y, MAP_DISPLAY_SIZE, MAP_DISPLAY_SIZE), 1)

        # Markers
        if spawn is not None:
            sx, sy = world_to_screen(*spawn)
            pygame.draw.circle(screen, COLORS["player"], (sx, sy), 7)
            pygame.draw.circle(screen, COLORS["white"], (sx, sy), 7, 1)
        if target is not None:
            tx, ty = world_to_screen(*target)
            pygame.draw.circle(screen, COLORS["target"], (tx, ty), 7)
            pygame.draw.circle(screen, COLORS["white"], (tx, ty), 7, 1)

        # Side panel — terrain legend
        panel_x = MAP_X + MAP_DISPLAY_SIZE + 30
        py = MAP_Y
        legend_title = font_med.render("TERRAIN LEGEND", True, COLORS["panel_fg"])
        screen.blit(legend_title, (panel_x, py)); py += 30
        for lev, info in TERRAIN_LEVELS.items():
            col = terrain_color(lev)
            pygame.draw.rect(screen, col, (panel_x, py, 14, 14))
            pygame.draw.rect(screen, COLORS["white"], (panel_x, py, 14, 14), 1)
            txt = font_small.render(f"  {info['name'].capitalize()} (cost {info['cost']})", True, COLORS["panel_fg"])
            screen.blit(txt, (panel_x + 18, py))
            py += 20

        # Coordinates
        py += 20
        if spawn:
            st = font_small.render(f"Spawn: ({spawn[0]}, {spawn[1]})", True, COLORS["player"])
            screen.blit(st, (panel_x, py)); py += 22
        if target:
            tt = font_small.render(f"Target: ({target[0]}, {target[1]})", True, COLORS["target"])
            screen.blit(tt, (panel_x, py)); py += 22

        # Hints
        hints = font_small.render("R = reset  •  ESC = quit  •  Click only on land", True, COLORS["gray"])
        screen.blit(hints, (MAP_X, WINDOW_H - 30))

        pygame.display.flip()
        clock.tick(30)


def screen_ai_playback(screen, clock, ckpt_path, spawn_rc, target_rc):
    """Phase 3: load model, run step-by-step, visualise."""

    device = "cpu"
    model = load_actor_critic(ckpt_path, device)
    print(f"Loaded model from {ckpt_path}")

    # Build env with fixed spawn/target
    env_config = EnvConfig(
        seed=42,
        minimap_ray=25,
        minimap_max_ray=DEFAULT_MINIMAP_MAX_RAY,
        minimap_occlude=True,
        minimap_min_clear_lv=0.25,
        spawn_r=spawn_rc[0], spawn_c=spawn_rc[1],
        target_r=target_rc[0], target_c=target_rc[1],
    )
    env = Islands(env_config)
    state, target_pos = env.reset(batch_size=1, seed=42)

    # Pre-render map surface (only once)
    map_surface = heightmap_to_surface(env.world_map, MAP_DISPLAY_SIZE)
    map_size = env.world_map.shape[0]

    font_large = pygame.font.Font(None, 36)
    font_med   = pygame.font.Font(None, 26)
    font_small = pygame.font.Font(None, 22)

    MAP_X, MAP_Y = 20, 60

    # Playback state
    trajectory = [tuple(state.position[0].cpu().tolist())]
    step_count = 0
    game_over = False
    won = False
    paused = False
    frames_per_step = 12   # lower = faster
    frame_counter = 0
    last_action = None

    ACTION_INDEX_TO_NAME = {v: k for k, v in ACTIONS.items()}

    def world_to_screen(r, c):
        scale = MAP_DISPLAY_SIZE / map_size
        return int(c * scale) + MAP_X, int(r * scale) + MAP_Y

    while True:
        # Events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "quit"
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return "quit"
                if ev.key == pygame.K_r:
                    return "reset"
                if ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    frames_per_step = max(1, frames_per_step - 2)
                if ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    frames_per_step = min(120, frames_per_step + 2)
                if ev.key == pygame.K_p:
                    paused = not paused

        # AI step
        if not game_over and not paused:
            frame_counter += 1
            if frame_counter >= frames_per_step:
                frame_counter = 0

                obs = build_obs(state, DEFAULT_MINIMAP_MAX_RAY)
                with torch.no_grad():
                    action = model.get_deterministic_action(obs)

                last_action = action.item()
                result = env.step(state, action, target_pos)
                state = result.state
                step_count += 1

                pos = tuple(state.position[0].cpu().tolist())
                trajectory.append(pos)

                alive = result.info["alive"][0]
                reached = result.info["reached"][0]
                if not alive:
                    game_over = True
                    won = False
                elif reached:
                    game_over = True
                    won = True
                elif step_count >= env_config.max_steps:
                    game_over = True
                    won = False

        # ── Draw ──
        screen.fill(COLORS["panel_bg"])

        # Title bar
        speed_label = f"Speed: {60 // max(frames_per_step, 1)} steps/s"
        status = "PAUSED" if paused else ("GAME OVER" if game_over else "PLAYING")
        title = font_large.render(f"AI Demo  —  {status}  —  {speed_label}", True, COLORS["blue_ui"])
        screen.blit(title, (MAP_X, 12))

        # Map
        screen.blit(map_surface, (MAP_X, MAP_Y))

        # Trajectory trail (gradient from yellow → red)
        n = len(trajectory)
        for i in range(1, n):
            t = i / max(n - 1, 1)
            r_c = int(255)
            g_c = int(200 * (1 - t))
            b_c = int(50 * (1 - t))
            p1 = world_to_screen(*trajectory[i - 1])
            p2 = world_to_screen(*trajectory[i])
            pygame.draw.line(screen, (r_c, g_c, b_c), p1, p2, 2)

        # Target marker
        tr, tc = target_rc
        tx, ty = world_to_screen(tr, tc)
        pygame.draw.circle(screen, COLORS["target"], (tx, ty), 7)
        pygame.draw.circle(screen, COLORS["white"], (tx, ty), 7, 1)

        # Player marker
        pr, pc = state.position[0].cpu().tolist()
        px, py_ = world_to_screen(int(pr), int(pc))
        pygame.draw.circle(screen, COLORS["player"], (px, py_), 6)
        pygame.draw.circle(screen, COLORS["white"], (px, py_), 6, 1)

        # Action label next to player
        if last_action is not None:
            act_sym = ACTION_NAMES.get(last_action, "?")
            act_label = font_med.render(act_sym, True, COLORS["white"])
            screen.blit(act_label, (px + 10, py_ - 10))

        # Map border
        pygame.draw.rect(screen, COLORS["white"], (MAP_X, MAP_Y, MAP_DISPLAY_SIZE, MAP_DISPLAY_SIZE), 1)

        # ── Minimap ──
        mm_x = MAP_X + MAP_DISPLAY_SIZE + 30
        mm_y = MAP_Y
        mm_label = font_med.render("Agent Minimap", True, COLORS["panel_fg"])
        screen.blit(mm_label, (mm_x, mm_y - 22))

        minimap_data = state.minimap[0, 0]  # [H, W]
        mm_surf = heightmap_to_surface(minimap_data, MINIMAP_DISPLAY_SIZE)
        screen.blit(mm_surf, (mm_x, mm_y))
        # Center dot
        cx = mm_x + MINIMAP_DISPLAY_SIZE // 2
        cy = mm_y + MINIMAP_DISPLAY_SIZE // 2
        pygame.draw.circle(screen, COLORS["player"], (cx, cy), 3)
        pygame.draw.rect(screen, COLORS["white"], (mm_x, mm_y, MINIMAP_DISPLAY_SIZE, MINIMAP_DISPLAY_SIZE), 1)

        # ── Stats panel ──
        stats_x = mm_x
        stats_y = mm_y + MINIMAP_DISPLAY_SIZE + 30

        s = state
        hp_val = s.hp[0].item()
        hp_max = env_config.max_hp
        hp_ratio = hp_val / hp_max

        stats = [
            ("HP", f"{hp_val:.1f} / {hp_max}", COLORS["green_ui"] if hp_ratio > 0.5 else (COLORS["red"] if hp_ratio < 0.3 else (255, 165, 0))),
            ("Resources", f"{s.resources[0].item():.1f}", COLORS["panel_fg"]),
            ("Time Cost", f"{s.cost[0].item():.2f}", COLORS["panel_fg"]),
            ("Moves", f"{step_count}", COLORS["panel_fg"]),
            ("Position", f"({int(pr)}, {int(pc)})", COLORS["panel_fg"]),
            ("Terrain", TERRAIN_LEVELS[int(s.terrain_lev[0].item())]["name"].capitalize(), COLORS["panel_fg"]),
            ("Distance", f"{torch.norm(s.compass[0].float()).item():.1f}", COLORS["panel_fg"]),
        ]

        for label, value, color in stats:
            lbl = font_small.render(f"{label}:", True, COLORS["gray"])
            val = font_med.render(value, True, color)
            screen.blit(lbl, (stats_x, stats_y))
            screen.blit(val, (stats_x + 90, stats_y - 2))
            stats_y += 28

        # Controls hint
        stats_y += 15
        controls = [
            "+/- : Speed",
            "P   : Pause",
            "R   : Reset",
            "ESC : Quit",
        ]
        ctrl_title = font_med.render("CONTROLS", True, COLORS["blue_ui"])
        screen.blit(ctrl_title, (stats_x, stats_y)); stats_y += 26
        for ctrl in controls:
            ct = font_small.render(ctrl, True, COLORS["gray"])
            screen.blit(ct, (stats_x, stats_y))
            stats_y += 20

        # ── Game over overlay ──
        if game_over:
            overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (0, 0))

            if won:
                msg = "TARGET REACHED!"
                color = COLORS["green_ui"]
                detail = f"Arrived in {step_count} moves  •  Cost: {s.cost[0].item():.2f}"
            else:
                if s.hp[0].item() <= 0:
                    msg = "AGENT DIED"
                else:
                    msg = "MAX STEPS REACHED"
                color = COLORS["red"]
                detail = f"Survived {step_count} moves  •  HP: {hp_val:.1f}"

            msg_surf = font_large.render(msg, True, color)
            det_surf = font_med.render(detail, True, COLORS["white"])
            hint_surf = font_small.render("R = try again  •  ESC = quit", True, COLORS["gray"])

            screen.blit(msg_surf, (WINDOW_W // 2 - msg_surf.get_width() // 2, WINDOW_H // 2 - 50))
            screen.blit(det_surf, (WINDOW_W // 2 - det_surf.get_width() // 2, WINDOW_H // 2))
            screen.blit(hint_surf, (WINDOW_W // 2 - hint_surf.get_width() // 2, WINDOW_H // 2 + 40))

        pygame.display.flip()
        clock.tick(60)


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Cogniland — AI Demo")
    clock = pygame.time.Clock()

    while True:
        # Phase 1: checkpoint
        ckpt_path = screen_select_checkpoint(screen, clock)
        if ckpt_path is None:
            break

        while True:
            # Phase 2: position picking
            env_config = EnvConfig(seed=42)
            result = screen_pick_positions(screen, clock, env_config)
            if result is None:
                pygame.quit()
                sys.exit()
            spawn_rc, target_rc = result

            # Phase 3: playback
            outcome = screen_ai_playback(screen, clock, ckpt_path, spawn_rc, target_rc)
            if outcome == "quit":
                pygame.quit()
                sys.exit()
            elif outcome == "reset":
                continue  # back to position picking
            else:
                break  # back to checkpoint selection

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
