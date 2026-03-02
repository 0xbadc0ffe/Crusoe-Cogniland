#!/usr/bin/env python3
"""Interactive Pygame Demo for Island Navigation Game

Usage:
    python demo.py              # Interactive difficulty selection
    python demo.py hard         # Hard mode

Controls:
    Arrow keys or WASD: Move
    Space: Stay (useful for resource gathering)
    R: Reset game
    ESC: Quit
"""

import pygame
import torch
import sys

from cogniland.env.constants import ACTIONS, TERRAIN_LEVELS, VISIBILITY_RANGES, palette
from cogniland.env.types import EnvConfig
from cogniland.env.islands import Islands

# Initialize Pygame
pygame.init()

# Build color lookup from canonical palette + UI-only extras
COLORS = {k: tuple(v) for k, v in palette.items()}
COLORS.update({
    'player': (255, 0, 0),
    'target': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'gray': (128, 128, 128),
    'red': (255, 0, 0),
    'green_ui': (0, 255, 0),
    'blue_ui': (0, 0, 255),
})


class IslandGameDemo:
    def __init__(self, window_width=1200, window_height=800, hard_mode=False):
        self.window_width = window_width
        self.window_height = window_height
        self.map_size = 400  # Size of the main map display
        self.minimap_size = 200
        self.ui_width = 300
        self.hard_mode = hard_mode

        # Create display
        self.screen = pygame.display.set_mode((window_width, window_height))
        mode_text = "Hard Mode" if hard_mode else "Easy Mode"
        pygame.display.set_caption(f"Island Navigation Game - {mode_text}")

        # Fonts
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)

        # Initialize environment
        self.reset_environment()

        # Game state
        self.running = True
        self.clock = pygame.time.Clock()
        self.zoom_level = 1.0
        self.map_offset = [0, 0]

    def reset_environment(self):
        """Reset the environment to initial state"""
        seed = torch.randint(1, 1000, (1,)).item()
        config = EnvConfig(
            seed=seed,
            hard_mode=self.hard_mode,
            minimap_ray=5,
            minimap_occlude=True,
            minimap_min_clear_lv=0.25,
        )
        self.env = Islands(config)

        # Ensure minimum distance between spawn and target
        while True:
            self.state, self.target_pos = self.env.reset(batch_size=1, seed=seed)
            spawn = self.state.position[0]
            target = self.target_pos[0]
            dist = torch.norm((spawn - target).float()).item()
            if dist >= config.size * 0.25:
                break
            seed += 1

        self.game_over = False
        self.won = False
        self.moves_count = 0

    def handle_events(self):
        """Handle pygame events"""
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
        """Execute a move in the environment"""
        if self.game_over:
            return

        result = self.env.step(self.state, action, self.target_pos)
        self.state = result.state
        self.moves_count += 1

        alive = result.info["alive"][0]
        reached = result.info["reached"][0]

        if not alive:
            self.game_over = True
            self.won = False
        elif reached:
            self.game_over = True
            self.won = True

    def terrain_level_to_color(self, level):
        """Convert terrain level to color using canonical TERRAIN_LEVELS + palette."""
        info = TERRAIN_LEVELS.get(int(level))
        if info is None:
            return COLORS['white']
        return COLORS.get(info['color'], COLORS['white'])

    def world_map_to_surface(self, world_map, size):
        """Convert world map to pygame surface"""
        height, width = world_map.shape[:2]
        surface = pygame.Surface((width, height))

        for y in range(height):
            for x in range(width):
                height_val = world_map[y, x].item() if world_map.dim() == 2 else world_map[y, x, 0].item()
                # Determine terrain level
                terrain_level = 8
                for level in range(9):
                    if height_val <= TERRAIN_LEVELS[level]["threshold"]:
                        terrain_level = level
                        break

                color = self.terrain_level_to_color(terrain_level)
                surface.set_at((x, y), color)

        return pygame.transform.scale(surface, (size, size))

    def draw_map(self):
        """Draw the main world map"""
        map_surface = self.world_map_to_surface(self.env.world_map, self.map_size)
        map_rect = pygame.Rect(10, 10, self.map_size, self.map_size)
        self.screen.blit(map_surface, map_rect)

        # Draw player position
        player_pos = self.state.position[0]
        map_scale = self.map_size / self.env.world_map.shape[0]
        player_x = int(player_pos[1] * map_scale) + 10
        player_y = int(player_pos[0] * map_scale) + 10
        pygame.draw.circle(self.screen, COLORS['player'], (player_x, player_y), 5)

        # Draw target position
        target = self.target_pos[0]
        target_x = int(target[1] * map_scale) + 10
        target_y = int(target[0] * map_scale) + 10
        pygame.draw.circle(self.screen, COLORS['target'], (target_x, target_y), 5)

        # Draw border
        pygame.draw.rect(self.screen, COLORS['black'], map_rect, 2)

    def draw_minimap(self):
        """Draw the minimap (player's view)"""
        minimap_data = self.state.minimap[0, 0]  # [H, W] from [B, 1, H, W]
        minimap_surface = self.world_map_to_surface(minimap_data, self.minimap_size)
        minimap_rect = pygame.Rect(self.map_size + 30, 10, self.minimap_size, self.minimap_size)
        self.screen.blit(minimap_surface, minimap_rect)

        # Draw player in center of minimap
        center_x = minimap_rect.centerx
        center_y = minimap_rect.centery
        pygame.draw.circle(self.screen, COLORS['player'], (center_x, center_y), 3)

        # Draw border
        pygame.draw.rect(self.screen, COLORS['black'], minimap_rect, 2)

        # Minimap label
        label = self.font_medium.render("Minimap", True, COLORS['black'])
        self.screen.blit(label, (minimap_rect.x, minimap_rect.y - 25))

    def draw_ui(self):
        """Draw the UI panel with game information"""
        ui_x = self.map_size + 30
        ui_y = self.minimap_size + 50

        s = self.state

        # Game stats
        mode_text = "HARD MODE" if self.hard_mode else "EASY MODE"
        stats = [
            f"Mode: {mode_text}",
            f"HP: {s.hp[0]:.1f} / {self.env.config.max_hp}",
            f"Resources: {s.resources[0]:.1f}",
            f"Time Cost: {s.cost[0]:.2f}",
            f"Moves: {self.moves_count}",
            f"",
            f"Position: ({s.position[0][0]}, {s.position[0][1]})",
            f"Terrain Level: {int(s.terrain_lev[0])}",
            f"Terrain: {TERRAIN_LEVELS[int(s.terrain_lev[0])]['name']}",
            f"Terrain Clock: {int(s.terrain_clock[0])}",
            f"",
            f"Distance to Target: {torch.norm(s.compass[0].float()):.1f}",
            f"Target: ({self.target_pos[0][0]}, {self.target_pos[0][1]})",
            f"",
            f"Visibility Range: {VISIBILITY_RANGES[int(s.terrain_lev[0])]}",
        ]

        y_offset = ui_y
        for stat in stats:
            if stat:
                color = COLORS['black']
                font = self.font_small

                if "Mode:" in stat:
                    if self.hard_mode:
                        color = COLORS['red']
                    else:
                        color = COLORS['green_ui']
                    font = self.font_medium
                elif "HP:" in stat:
                    hp_ratio = s.hp[0] / self.env.config.max_hp
                    if hp_ratio < 0.3:
                        color = COLORS['red']
                    elif hp_ratio < 0.6:
                        color = (255, 165, 0)  # Orange
                    else:
                        color = COLORS['green_ui']

                text = font.render(stat, True, color)
                self.screen.blit(text, (ui_x, y_offset))
            y_offset += 25

        # Controls
        y_offset += 20
        controls = [
            "CONTROLS:",
            "Arrow Keys/WASD: Move",
            "Space: Stay",
            "R: Reset",
            "ESC: Quit",
            ""
        ]

        if self.hard_mode:
            controls.extend([
                "HARD MODE RULES:",
                "With resources: -0.25/turn, +1HP",
                "Without resources: -0.5HP/turn",
                "Find forests to survive!"
            ])

        for control in controls:
            if control:
                color = COLORS['gray']
                font = self.font_small

                if control in ["CONTROLS:", "HARD MODE RULES:"]:
                    color = COLORS['red'] if "HARD MODE" in control else COLORS['black']
                    font = self.font_medium

                text = font.render(control, True, color)
                self.screen.blit(text, (ui_x, y_offset))
                y_offset += 20 if font == self.font_medium else 18
            else:
                y_offset += 10

        # Terrain legend
        y_offset += 30
        legend_title = self.font_medium.render("TERRAIN LEGEND:", True, COLORS['black'])
        self.screen.blit(legend_title, (ui_x, y_offset))
        y_offset += 25

        terrain_names = [
            (i, f"{info['name'].capitalize()} ({info['cost']})")
            for i, info in TERRAIN_LEVELS.items()
        ]

        for level, name in terrain_names:
            color = self.terrain_level_to_color(level)
            pygame.draw.rect(self.screen, color, (ui_x, y_offset, 15, 15))
            pygame.draw.rect(self.screen, COLORS['black'], (ui_x, y_offset, 15, 15), 1)
            text = self.font_small.render(name, True, COLORS['black'])
            self.screen.blit(text, (ui_x + 20, y_offset))
            y_offset += 18

    def draw_game_over(self):
        """Draw game over screen"""
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.set_alpha(180)
        overlay.fill(COLORS['black'])
        self.screen.blit(overlay, (0, 0))

        if self.won:
            title = "VICTORY!"
            color = COLORS['green_ui']
            message = f"Reached target in {self.moves_count} moves"
            score_msg = f"Final time cost: {self.state.cost[0]:.2f}"
        else:
            title = "GAME OVER"
            color = COLORS['red']
            message = f"HP reached zero after {self.moves_count} moves"
            score_msg = f"Time cost: {self.state.cost[0]:.2f}"

        title_text = self.font_large.render(title, True, color)
        title_rect = title_text.get_rect(center=(self.window_width//2, self.window_height//2 - 50))
        self.screen.blit(title_text, title_rect)

        message_text = self.font_medium.render(message, True, COLORS['white'])
        message_rect = message_text.get_rect(center=(self.window_width//2, self.window_height//2))
        self.screen.blit(message_text, message_rect)

        score_text = self.font_medium.render(score_msg, True, COLORS['white'])
        score_rect = score_text.get_rect(center=(self.window_width//2, self.window_height//2 + 30))
        self.screen.blit(score_text, score_rect)

        restart_text = self.font_small.render("Press R to restart", True, COLORS['white'])
        restart_rect = restart_text.get_rect(center=(self.window_width//2, self.window_height//2 + 80))
        self.screen.blit(restart_text, restart_rect)

    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()

            self.screen.fill(COLORS['white'])

            self.draw_map()
            self.draw_minimap()
            self.draw_ui()

            if self.game_over:
                self.draw_game_over()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


def show_difficulty_selection():
    """Show difficulty selection screen"""
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Island Navigation - Select Difficulty")

    font_large = pygame.font.Font(None, 36)
    font_medium = pygame.font.Font(None, 28)
    font_small = pygame.font.Font(None, 20)

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return False  # Easy mode
                elif event.key == pygame.K_2:
                    return True   # Hard mode
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        screen.fill(COLORS['white'])

        title = font_large.render("Select Difficulty", True, COLORS['black'])
        title_rect = title.get_rect(center=(200, 60))
        screen.blit(title, title_rect)

        easy_text = font_medium.render("1 - Easy Mode", True, COLORS['green_ui'])
        easy_rect = easy_text.get_rect(center=(200, 120))
        screen.blit(easy_text, easy_rect)

        easy_desc = font_small.render("Standard gameplay with passive healing", True, COLORS['gray'])
        easy_desc_rect = easy_desc.get_rect(center=(200, 145))
        screen.blit(easy_desc, easy_desc_rect)

        hard_text = font_medium.render("2 - Hard Mode", True, COLORS['red'])
        hard_rect = hard_text.get_rect(center=(200, 180))
        screen.blit(hard_text, hard_rect)

        hard_desc1 = font_small.render("Resources required for survival!", True, COLORS['gray'])
        hard_desc1_rect = hard_desc1.get_rect(center=(200, 205))
        screen.blit(hard_desc1, hard_desc1_rect)

        hard_desc2 = font_small.render("No resources = -0.5 HP/turn", True, COLORS['gray'])
        hard_desc2_rect = hard_desc2.get_rect(center=(200, 220))
        screen.blit(hard_desc2, hard_desc2_rect)

        instruction = font_small.render("Press ESC to quit", True, COLORS['black'])
        instruction_rect = instruction.get_rect(center=(200, 260))
        screen.blit(instruction, instruction_rect)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        hard_mode = sys.argv[1].lower() in ('hard', 'h', '2')
    else:
        hard_mode = show_difficulty_selection()

    game = IslandGameDemo(hard_mode=hard_mode)
    game.run()
