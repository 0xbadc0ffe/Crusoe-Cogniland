#!/usr/bin/env python3
"""Interactive Pygame Demo for Island Navigation Game

Usage:
    python human_demo.py

Controls:
    Arrow keys or WASD: Move
    Space: Stay (useful for resource gathering)
    R: Reset game
    ESC: Quit
"""

import numpy as np
import pygame
import torch
import math
import sys

from cogniland.env.constants import ACTIONS
from cogniland.env.types import EnvConfig, MapGenConfig, MinimapConfig
from cogniland.env.islands import Islands

# Initialize Pygame
pygame.init()

# UI-only colors
COLORS = {
    'player': (255, 0, 0),
    'target': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'gray': (128, 128, 128),
    'red': (255, 0, 0),
    'green_ui': (0, 255, 0),
    'blue_ui': (0, 0, 255),
}


def draw_star(surface, cx, cy, r_outer, r_inner, color=(255, 215, 0), n_points=5):
    """Draw a filled n-pointed star centred at (cx, cy)."""
    pts = []
    for i in range(2 * n_points):
        r = r_outer if i % 2 == 0 else r_inner
        angle = math.pi * i / n_points - math.pi / 2
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (0, 0, 0), pts, 1)


class IslandGameDemo:
    def __init__(self, window_width=1200, window_height=800):
        self.window_width = window_width
        self.window_height = window_height
        self.map_size = 400
        self.minimap_size = 240
        self.ui_width = 300

        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Island Navigation Game")

        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)

        self.reset_environment()

        self.running = True
        self.clock = pygame.time.Clock()

    def reset_environment(self):
        seed = torch.randint(1, 1000, (1,)).item()
        config = EnvConfig(
            map_generation=MapGenConfig(seed=seed),
            minimap=MinimapConfig(max_ray=15, occlude=True, clear_tolerance=0.1),
        )
        self.env = Islands(config)
        self._compiled = self.env.compiled

        while True:
            self.state, self.target_pos = self.env.reset(batch_size=1, seed=seed)
            spawn = self.state.position[0]
            target = self.target_pos[0]
            dist = (spawn - target).float().abs().sum().item()
            if dist >= config.size * 0.3:
                break
            seed += 1

        H = W = config.size
        self.seen_mask = np.zeros((H, W), dtype=bool)
        self._update_seen_mask()

        # Precompute base map RGB image [H, W, 3] uint8
        wm = self.env.world_map.numpy()
        compiled = self._compiled
        thresholds = compiled.thresholds.cpu().numpy()
        terrain_map = np.searchsorted(thresholds, wm).clip(0, compiled.num_terrains - 1)
        color_lut = compiled.color_lut.cpu().numpy()
        self._base_map_rgb = color_lut[terrain_map]  # [H, W, 3]

        self.game_over = False
        self.won = False
        self.moves_count = 0
        self._risk_sum = 0.0
        self._risk_count = 0

    def _update_seen_mask(self):
        """Mark all currently-visible cells from the minimap visibility channel."""
        vis_mask = self.state.minimap[0, 2].numpy()  # [D, D] float in [0, 1]
        max_ray = self.env.config.minimap_max_ray
        D = 2 * max_ray + 1
        pos = self.state.position[0].numpy()
        cy, cx = int(pos[0]), int(pos[1])
        H, W = self.seen_mask.shape

        dy = np.arange(D) - max_ray
        dx = np.arange(D) - max_ray
        dy_grid, dx_grid = np.meshgrid(dy, dx, indexing="ij")
        world_rows = np.clip(cy + dy_grid, 0, H - 1)
        world_cols = np.clip(cx + dx_grid, 0, W - 1)
        visible = vis_mask > 0.5
        self.seen_mask[world_rows[visible], world_cols[visible]] = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset_environment()
                elif not self.game_over:
                    action = None
                    if event.key in [pygame.K_UP, pygame.K_w]:
                        action = torch.tensor([ACTIONS["up"]])
                    elif event.key in [pygame.K_DOWN, pygame.K_s]:
                        action = torch.tensor([ACTIONS["down"]])
                    elif event.key in [pygame.K_LEFT, pygame.K_a]:
                        action = torch.tensor([ACTIONS["left"]])
                    elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                        action = torch.tensor([ACTIONS["right"]])
                    elif event.key == pygame.K_SPACE:
                        action = torch.tensor([ACTIONS["stay"]])

                    if action is not None:
                        self.make_move(action)

    def make_move(self, action):
        if self.game_over:
            return

        # Compute per-terrain drain for risk tracking before step
        compiled = self._compiled
        terrain_idx = int(self.state.terrain_idx[0].item())
        drain = compiled.res_drain[terrain_idx].item()
        res = self.state.resources[0].item()
        hp = self.state.hp[0].item()
        risk = drain / (res + hp / 2.0 + 1e-6)
        self._risk_sum += risk
        self._risk_count += 1

        result = self.env.step(self.state, action.to(self.env._device), self.target_pos)
        self.state = result.state
        self.moves_count += 1

        self._update_seen_mask()

        alive = result.info["alive"][0]
        reached = result.info["reached"][0]

        if not alive:
            self.game_over = True
            self.won = False
        elif reached:
            self.game_over = True
            self.won = True

    def terrain_level_to_color(self, level):
        compiled = self._compiled
        idx = int(level)
        if 0 <= idx < compiled.num_terrains:
            c = compiled.color_lut[idx].cpu().tolist()
            return (c[0], c[1], c[2])
        return COLORS['white']

    def draw_map(self):
        """Draw world map with fog-of-war darkening for unseen cells."""
        # Apply fog: darken unseen cells to 55% brightness
        fog = np.where(self.seen_mask[:, :, None], 1.0, 0.55).astype(np.float32)
        rgb = (self._base_map_rgb * fog).astype(np.uint8)  # [H, W, 3]

        # Convert to pygame surface via surfarray (fast path)
        surf = pygame.Surface((rgb.shape[1], rgb.shape[0]))
        pygame.surfarray.blit_array(surf, rgb.transpose(1, 0, 2))
        map_surface = pygame.transform.scale(surf, (self.map_size, self.map_size))

        map_rect = pygame.Rect(10, 10, self.map_size, self.map_size)
        self.screen.blit(map_surface, map_rect)

        map_scale = self.map_size / self.env.world_map.shape[0]

        player_pos = self.state.position[0]
        player_x = int(player_pos[1] * map_scale) + 10
        player_y = int(player_pos[0] * map_scale) + 10
        pygame.draw.circle(self.screen, COLORS['player'], (player_x, player_y), 5)

        target = self.target_pos[0]
        target_x = int(target[1] * map_scale) + 10
        target_y = int(target[0] * map_scale) + 10
        draw_star(self.screen, target_x, target_y, r_outer=8, r_inner=3)

        pygame.draw.rect(self.screen, COLORS['black'], map_rect, 2)

    def draw_minimap(self):
        minimap_data = self.state.minimap[0, 0]  # heightmap channel
        height, width = minimap_data.shape
        surf = pygame.Surface((width, height))
        for y in range(height):
            for x in range(width):
                h_val = minimap_data[y, x].item()
                compiled = self._compiled
                thresholds = compiled.thresholds.cpu()
                t_level = compiled.num_terrains - 1
                for level in range(compiled.num_terrains):
                    if h_val <= thresholds[level].item():
                        t_level = level
                        break
                surf.set_at((x, y), self.terrain_level_to_color(t_level))
        minimap_surface = pygame.transform.scale(surf, (self.minimap_size, self.minimap_size))

        minimap_rect = pygame.Rect(self.map_size + 30, 30, self.minimap_size, self.minimap_size)
        self.screen.blit(minimap_surface, minimap_rect)

        # Compass arrow pointing toward target
        compass = self.state.compass[0]
        cx_mm, cy_mm = minimap_rect.center
        dy_dir = -float(compass[0])
        dx_dir = -float(compass[1])
        mag = (dx_dir ** 2 + dy_dir ** 2) ** 0.5
        if mag > 1e-6:
            dx_dir /= mag
            dy_dir /= mag
            arrow_len = 32
            ex = int(cx_mm + dx_dir * arrow_len)
            ey = int(cy_mm + dy_dir * arrow_len)
            arrow_color = (255, 220, 50)
            pygame.draw.line(self.screen, arrow_color, (cx_mm, cy_mm), (ex, ey), 2)
            head_len = 8
            angle = math.atan2(dy_dir, dx_dir)
            back = angle + math.pi
            for side in (math.pi / 5, -math.pi / 5):
                hx = int(ex + head_len * math.cos(back + side))
                hy = int(ey + head_len * math.sin(back + side))
                pygame.draw.line(self.screen, arrow_color, (ex, ey), (hx, hy), 2)

        pygame.draw.circle(self.screen, COLORS['player'], minimap_rect.center, 3)
        pygame.draw.rect(self.screen, COLORS['black'], minimap_rect, 2)

        label = self.font_small.render("Agent view  (\u25b2 = target)", True, COLORS['black'])
        self.screen.blit(label, (minimap_rect.x, minimap_rect.y - 16))

    def draw_ui(self):
        ui_x = self.map_size + 30
        ui_y = self.minimap_size + 60  # below enlarged minimap

        s = self.state
        ec = self.env.config

        hp_ratio = s.hp[0] / ec.max_hp
        res_ratio = s.resources[0] / ec.max_resources
        risk_mean = self._risk_sum / max(self._risk_count, 1)
        explored_pct = 100.0 * self.seen_mask.sum() / self.seen_mask.size

        stats = [
            ("Live Stats", self.font_medium, COLORS['black']),
            (f"Moves:     {self.moves_count}", self.font_small, COLORS['black']),
            (f"Time Cost: {s.cost[0]:.2f}", self.font_small, COLORS['black']),
            (f"HP:        {s.hp[0]:.1f} / {ec.max_hp:.0f}", self.font_small,
                COLORS['red'] if hp_ratio < 0.3 else (255, 165, 0) if hp_ratio < 0.6 else COLORS['green_ui']),
            (f"Resources: {s.resources[0]:.1f} / {ec.max_resources:.0f}", self.font_small,
                COLORS['red'] if res_ratio < 0.2 else COLORS['black']),
            (f"Risk Exp:  {risk_mean:.3f}", self.font_small,
                COLORS['red'] if risk_mean > 0.5 else COLORS['black']),
            (f"Explored:  {explored_pct:.1f}%", self.font_small, COLORS['black']),
            ("", self.font_small, COLORS['black']),
            ("Position", self.font_medium, COLORS['black']),
            (f"Pos:    ({s.position[0][0]}, {s.position[0][1]})", self.font_small, COLORS['black']),
            (f"Target: ({self.target_pos[0][0]}, {self.target_pos[0][1]})", self.font_small, COLORS['black']),
            (f"Dist:   {(s.position[0].float() - self.target_pos[0].float()).abs().sum():.1f}", self.font_small, COLORS['black']),
            (f"Terrain: {self._compiled.terrain_names[int(s.terrain_idx[0])]}", self.font_small, COLORS['black']),
            (f"Visibility: {self._compiled.visibility[int(s.terrain_idx[0])].item()}", self.font_small, COLORS['black']),
        ]

        y_offset = ui_y
        for row in stats:
            text, font, color = row
            if text:
                surf = font.render(text, True, color)
                self.screen.blit(surf, (ui_x, y_offset))
            y_offset += 26 if font == self.font_medium else 20

    def draw_right_panel(self):
        right_x = self.map_size + self.minimap_size + 55
        y = 10

        controls_title = self.font_medium.render("Controls", True, COLORS['black'])
        self.screen.blit(controls_title, (right_x, y))
        y += 26
        for line in ["WASD / Arrows: Move", "Space: Stay", "R: Reset", "ESC: Quit"]:
            surf = self.font_small.render(line, True, COLORS['gray'])
            self.screen.blit(surf, (right_x, y))
            y += 20

        y += 20
        legend_title = self.font_medium.render("Terrain Legend", True, COLORS['black'])
        self.screen.blit(legend_title, (right_x, y))
        y += 26
        for i, name in enumerate(self._compiled.terrain_names):
            color = self.terrain_level_to_color(i)
            pygame.draw.rect(self.screen, color, (right_x, y, 14, 14))
            pygame.draw.rect(self.screen, COLORS['black'], (right_x, y, 14, 14), 1)
            cost = self._compiled.move_costs[i].item()
            label = f"{name.capitalize()} (cost {cost:.1f})"
            surf = self.font_small.render(label, True, COLORS['black'])
            self.screen.blit(surf, (right_x + 18, y))
            y += 20

    def draw_game_over(self):
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.set_alpha(180)
        overlay.fill(COLORS['black'])
        self.screen.blit(overlay, (0, 0))

        if self.won:
            title, color = "VICTORY!", COLORS['green_ui']
            lines = [
                f"Reached target in {self.moves_count} moves",
                f"Final time cost: {self.state.cost[0]:.2f}",
                f"Mean risk exposure: {self._risk_sum / max(self._risk_count, 1):.3f}",
                f"Map explored: {100.0 * self.seen_mask.sum() / self.seen_mask.size:.1f}%",
            ]
        else:
            title, color = "GAME OVER", COLORS['red']
            lines = [
                f"HP reached zero after {self.moves_count} moves",
                f"Time cost: {self.state.cost[0]:.2f}",
                f"Mean risk exposure: {self._risk_sum / max(self._risk_count, 1):.3f}",
            ]

        cy = self.window_height // 2
        title_surf = self.font_large.render(title, True, color)
        self.screen.blit(title_surf, title_surf.get_rect(center=(self.window_width // 2, cy - 60)))
        for k, line in enumerate(lines):
            surf = self.font_medium.render(line, True, COLORS['white'])
            self.screen.blit(surf, surf.get_rect(center=(self.window_width // 2, cy - 10 + k * 32)))
        restart_surf = self.font_small.render("Press R to restart", True, COLORS['white'])
        self.screen.blit(restart_surf, restart_surf.get_rect(center=(self.window_width // 2, cy + 130)))

    def run(self):
        while self.running:
            self.handle_events()
            self.screen.fill(COLORS['white'])
            self.draw_map()
            self.draw_minimap()
            self.draw_ui()
            self.draw_right_panel()
            if self.game_over:
                self.draw_game_over()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = IslandGameDemo()
    game.run()
