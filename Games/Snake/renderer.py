"""Pygame-based visualization for Snake game."""

import os
from pathlib import Path

import pygame

from .logic.game_logic import SnakeLogic


class SnakeRenderer:
    """Renders Snake game state using Pygame."""
    
    # Cell size in pixels
    CELL_SIZE = 30
    BORDER_TILES = 1
    HUD_HEIGHT = 40
    
    # Colors (RGB)
    COLOR_BG = (26, 26, 26)      # Dark gray background
    COLOR_SNAKE = (76, 175, 80)  # Green snake
    COLOR_FOOD = (255, 87, 34)   # Orange food
    COLOR_GRID = (50, 50, 50)    # Dark grid lines
    
    def __init__(self, game: SnakeLogic):
        """Initialize the renderer."""
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError(
                "No graphical display detected (DISPLAY/WAYLAND_DISPLAY is empty). "
                "Run this on a desktop session or use SSH with X11 forwarding."
            )

        self.game = game
        self.window_width = (game.GRID_WIDTH + 2 * self.BORDER_TILES) * self.CELL_SIZE
        self.window_height = self.HUD_HEIGHT + (game.GRID_HEIGHT + 2 * self.BORDER_TILES) * self.CELL_SIZE
        self.play_origin_x = self.BORDER_TILES * self.CELL_SIZE
        self.play_origin_y = self.HUD_HEIGHT + self.BORDER_TILES * self.CELL_SIZE
        
        # Hide pygame startup banner and initialize only display/font to avoid ALSA noise.
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Snake - DQN")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self._load_textures()

    def _load_textures(self) -> None:
        """Load custom textures if available; keep shape fallback if missing."""
        textures_dir = Path(__file__).resolve().parent / "textures"

        self._tex_head = self._load_texture(textures_dir / "Snake_head.png")
        self._tex_body = self._load_texture(textures_dir / "Snake_body.png")
        self._tex_tail = self._load_texture(textures_dir / "Snake_tail.png")
        self._tex_food = self._load_texture(textures_dir / "Snake_apple.png")
        self._tex_wall = self._load_texture(textures_dir / "Snake_Wall.png")

    def _load_texture(self, path: Path) -> pygame.Surface | None:
        """Load and scale one texture to cell size."""
        if not path.exists():
            return None
        try:
            surface = pygame.image.load(str(path)).convert_alpha()
            return pygame.transform.smoothscale(surface, (self.CELL_SIZE, self.CELL_SIZE))
        except pygame.error:
            return None
        
    def render(self, fps: int = 5) -> bool:
        """
        Render current game state.
        Returns True if window is still open, False if closed.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        # Clear background
        self.screen.fill(self.COLOR_BG)

        # Draw wall border
        self._draw_walls()
        
        # Draw grid
        self._draw_grid()
        
        # Draw food
        self._draw_food()
        
        # Draw snake
        self._draw_snake()
        
        # Draw score
        self._draw_score()
        
        pygame.display.flip()
        self.clock.tick(fps)
        return True
    
    def _draw_grid(self) -> None:
        """Draw grid lines."""
        for x in range(self.game.GRID_WIDTH + 1):
            start_pos = (self.play_origin_x + x * self.CELL_SIZE, self.play_origin_y)
            end_pos = (
                self.play_origin_x + x * self.CELL_SIZE,
                self.play_origin_y + self.game.GRID_HEIGHT * self.CELL_SIZE,
            )
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        
        for y in range(self.game.GRID_HEIGHT + 1):
            start_pos = (self.play_origin_x, self.play_origin_y + y * self.CELL_SIZE)
            end_pos = (
                self.play_origin_x + self.game.GRID_WIDTH * self.CELL_SIZE,
                self.play_origin_y + y * self.CELL_SIZE,
            )
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _draw_walls(self) -> None:
        """Draw one-tile wall border around the playable area."""
        total_w = self.game.GRID_WIDTH + 2 * self.BORDER_TILES
        total_h = self.game.GRID_HEIGHT + 2 * self.BORDER_TILES

        for tx in range(total_w):
            for ty in range(total_h):
                is_border = tx in (0, total_w - 1) or ty in (0, total_h - 1)
                if not is_border:
                    continue

                pixel_x = tx * self.CELL_SIZE
                pixel_y = self.HUD_HEIGHT + ty * self.CELL_SIZE

                if self._tex_wall is not None:
                    self.screen.blit(self._tex_wall, (pixel_x, pixel_y))
                else:
                    wall_rect = pygame.Rect(pixel_x, pixel_y, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, (70, 70, 70), wall_rect)

    def _rotation_from_delta(self, dx: int, dy: int) -> float:
        """Convert direction vector to clockwise rotation degrees."""
        if (dx, dy) == (1, 0):
            return 0.0
        if (dx, dy) == (0, 1):
            return 90.0
        if (dx, dy) == (-1, 0):
            return 180.0
        if (dx, dy) == (0, -1):
            return 270.0
        return 0.0
    
    def _draw_snake(self) -> None:
        """Draw snake using textures with shape fallback."""
        body = self.game.body
        if not body:
            return

        # Draw middle/body segments
        for i in range(1, max(1, len(body) - 1)):
            x, y = body[i]
            pixel = (self.play_origin_x + x * self.CELL_SIZE, self.play_origin_y + y * self.CELL_SIZE)

            if self._tex_body is not None:
                self.screen.blit(self._tex_body, pixel)
            else:
                rect = pygame.Rect(pixel[0] + 2, pixel[1] + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
                pygame.draw.rect(self.screen, self.COLOR_SNAKE, rect)

        # Draw head
        hx, hy = body[0]
        head_pixel = (self.play_origin_x + hx * self.CELL_SIZE, self.play_origin_y + hy * self.CELL_SIZE)
        if len(body) > 1:
            nx, ny = body[1]
            head_rot = self._rotation_from_delta(hx - nx, hy - ny)
        else:
            head_rot = self._rotation_from_delta(1, 0)

        if self._tex_head is not None:
            head_sprite = pygame.transform.rotate(self._tex_head, -head_rot)
            head_rect = head_sprite.get_rect(center=(head_pixel[0] + self.CELL_SIZE // 2, head_pixel[1] + self.CELL_SIZE // 2))
            self.screen.blit(head_sprite, head_rect.topleft)
        else:
            rect = pygame.Rect(head_pixel[0] + 2, head_pixel[1] + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            pygame.draw.rect(self.screen, (100, 255, 100), rect)

        # Draw tail if there is one
        if len(body) > 1:
            tx, ty = body[-1]
            px, py = body[-2]
            tail_rot = self._rotation_from_delta(px - tx, py - ty)
            tail_pixel = (self.play_origin_x + tx * self.CELL_SIZE, self.play_origin_y + ty * self.CELL_SIZE)

            if self._tex_tail is not None:
                tail_sprite = pygame.transform.rotate(self._tex_tail, -tail_rot)
                tail_rect = tail_sprite.get_rect(center=(tail_pixel[0] + self.CELL_SIZE // 2, tail_pixel[1] + self.CELL_SIZE // 2))
                self.screen.blit(tail_sprite, tail_rect.topleft)
    
    def _draw_food(self) -> None:
        """Draw food."""
        foods = list(getattr(self.game, "foods", []) or [])
        primary_food = getattr(self.game, "food", None)
        if not foods and primary_food is not None:
            foods = [primary_food]

        for x, y in foods:
            pixel = (self.play_origin_x + x * self.CELL_SIZE, self.play_origin_y + y * self.CELL_SIZE)
            if self._tex_food is not None:
                self.screen.blit(self._tex_food, pixel)
            else:
                rect = pygame.Rect(pixel[0] + 5, pixel[1] + 5, self.CELL_SIZE - 10, self.CELL_SIZE - 10)
                pygame.draw.ellipse(self.screen, self.COLOR_FOOD, rect)
    
    def _draw_score(self) -> None:
        """Draw score at top."""
        score_text = self.font.render(f"Score: {self.game.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
    
    def close(self) -> None:
        """Close the renderer."""
        pygame.quit()
