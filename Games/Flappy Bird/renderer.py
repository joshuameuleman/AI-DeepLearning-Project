"""Pygame-based visualization for Flappy Bird game."""

from __future__ import annotations

import os
from pathlib import Path

import pygame


class FlappyBirdRenderer:
    """Renders Flappy Bird state using optional texture assets."""

    BG_COLOR = (118, 197, 255)
    GROUND_COLOR = (222, 216, 149)
    PIPE_COLOR = (74, 183, 73)

    def __init__(self, game) -> None:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError(
                "No graphical display detected (DISPLAY/WAYLAND_DISPLAY is empty). "
                "Run this on a desktop session or use SSH with X11 forwarding."
            )

        self.game = game
        self.scale = 2
        self.window_width = int(game.SCREEN_WIDTH * self.scale)
        self.window_height = int(game.SCREEN_HEIGHT * self.scale)

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        pygame.display.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Flappy Bird - DQN")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)

        self._bird_up = None
        self._bird_mid = None
        self._bird_down = None
        self._pipe_sprite = None
        self._load_textures()

    def _load_textures(self) -> None:
        base_dir = Path(__file__).resolve().parent
        textures_dir = base_dir / "textures"
        if not textures_dir.exists():
            textures_dir = base_dir / "logic" / "textures"

        bird_w = int(34 * self.scale)
        bird_h = int(24 * self.scale)
        pipe_w = int(self.game.PIPE_WIDTH * self.scale)

        self._bird_up = self._load_texture(textures_dir / "yellowbird-upflap.png", (bird_w, bird_h))
        self._bird_mid = self._load_texture(textures_dir / "yellowbird-midflap.png", (bird_w, bird_h))
        self._bird_down = self._load_texture(textures_dir / "yellowbird-downflap.png", (bird_w, bird_h))
        self._pipe_sprite = self._load_texture(textures_dir / "pipe-green.png", (pipe_w, int(320 * self.scale)))

    def _load_texture(self, path: Path, size: tuple[int, int]) -> pygame.Surface | None:
        if not path.exists():
            return None
        try:
            surface = pygame.image.load(str(path)).convert_alpha()
            return pygame.transform.smoothscale(surface, size)
        except pygame.error:
            return None

    def _bird_frame(self) -> pygame.Surface | None:
        if self.game.bird_velocity < -1.0:
            return self._bird_up or self._bird_mid or self._bird_down
        if self.game.bird_velocity > 1.5:
            return self._bird_down or self._bird_mid or self._bird_up
        return self._bird_mid or self._bird_up or self._bird_down

    def _draw_pipes(self) -> None:
        for pipe in self.game.pipes:
            pipe_x = int(float(pipe["x"]) * self.scale)
            gap_center = float(pipe["gap_y"]) * self.scale
            gap_half = float(self.game.PIPE_GAP * self.scale) / 2.0

            top_height = max(0, int(gap_center - gap_half))
            bottom_y = min(self.window_height, int(gap_center + gap_half))
            bottom_height = max(0, self.window_height - bottom_y)

            pipe_w = int(self.game.PIPE_WIDTH * self.scale)

            if self._pipe_sprite is not None:
                if top_height > 0:
                    top_surface = pygame.transform.smoothscale(self._pipe_sprite, (pipe_w, top_height))
                    top_surface = pygame.transform.flip(top_surface, False, True)
                    self.screen.blit(top_surface, (pipe_x, 0))
                if bottom_height > 0:
                    bottom_surface = pygame.transform.smoothscale(self._pipe_sprite, (pipe_w, bottom_height))
                    self.screen.blit(bottom_surface, (pipe_x, bottom_y))
            else:
                if top_height > 0:
                    pygame.draw.rect(self.screen, self.PIPE_COLOR, pygame.Rect(pipe_x, 0, pipe_w, top_height))
                if bottom_height > 0:
                    pygame.draw.rect(
                        self.screen,
                        self.PIPE_COLOR,
                        pygame.Rect(pipe_x, bottom_y, pipe_w, bottom_height),
                    )

    def _draw_bird(self) -> None:
        bird_x = int(self.game.BIRD_X * self.scale)
        bird_y = int(self.game.bird_y * self.scale)

        sprite = self._bird_frame()
        if sprite is not None:
            tilt = max(-30.0, min(30.0, -self.game.bird_velocity * 4.0))
            rotated = pygame.transform.rotate(sprite, tilt)
            rect = rotated.get_rect(center=(bird_x, bird_y))
            self.screen.blit(rotated, rect.topleft)
            return

        radius = max(6, int(self.game.BIRD_RADIUS * self.scale * 0.7))
        pygame.draw.circle(self.screen, (255, 225, 72), (bird_x, bird_y), radius)

    def _draw_hud(self) -> None:
        pygame.draw.rect(self.screen, self.GROUND_COLOR, pygame.Rect(0, self.window_height - 6, self.window_width, 6))
        text = self.font.render(f"Score: {self.game.score}", True, (33, 33, 33))
        self.screen.blit(text, (10, 10))

    def render(self, fps: int = 30) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.screen.fill(self.BG_COLOR)
        self._draw_pipes()
        self._draw_bird()
        self._draw_hud()

        pygame.display.flip()
        self.clock.tick(max(1, int(fps)))
        return True

    def close(self) -> None:
        pygame.quit()
