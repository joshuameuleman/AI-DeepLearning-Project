"""Manual Snake game powered by pygame."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from Games.Snake.logic.game_logic import Direction, SnakeLogic
from Games.Snake.renderer import SnakeRenderer


KEY_TO_DIRECTION = {
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_d: Direction.RIGHT,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_s: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_a: Direction.LEFT,
    pygame.K_UP: Direction.UP,
    pygame.K_w: Direction.UP,
}


def run_game(grid_size: int | None = None, fps: int = 12) -> None:
    """Launch a local playable Snake window."""
    game = SnakeLogic(grid_size=grid_size)
    game.reset()
    renderer = SnakeRenderer(game)
    pygame.display.set_caption("Snake - Play")

    target_direction = Direction(game.direction)
    paused = False
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key in (pygame.K_SPACE, pygame.K_p):
                        paused = not paused
                    elif event.key == pygame.K_r:
                        game.reset()
                        target_direction = Direction(game.direction)
                        paused = False
                    elif event.key in KEY_TO_DIRECTION:
                        target_direction = KEY_TO_DIRECTION[event.key]

            if not paused:
                result = game.step_towards(target_direction)
                if result.done:
                    pygame.time.delay(450)
                    game.reset()
                    target_direction = Direction(game.direction)

            if not renderer.render(fps=max(1, int(fps))):
                running = False
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Snake locally with pygame.")
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_game(grid_size=args.grid_size, fps=args.fps)


if __name__ == "__main__":
    main()
