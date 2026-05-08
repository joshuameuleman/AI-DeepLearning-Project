"""Interactive Snake visualization with Pygame."""

import os
import sys
from pathlib import Path

# Bootstrap path for imports
if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

# Hide pygame startup banner in terminal.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
try:
    # Works when loaded as a script/module outside package context.
    from Games.Snake.logic.game_logic import SnakeLogic
    from Games.Snake.renderer import SnakeRenderer
except ImportError:
    # Works when executed as part of the package.
    from .logic.game_logic import SnakeLogic
    from .renderer import SnakeRenderer


def main() -> None:
    """Run window-only Snake watch mode."""
    game = SnakeLogic()
    game.reset()
    renderer = SnakeRenderer(game)
    pygame.display.set_caption("Snake - Meekijken")

    is_running = True

    try:
        while is_running:
            # Simple watch behavior: mostly straight with occasional turns.
            action = 0
            if game.steps_since_food > 8:
                action = 1 if game.direction in (0, 2) else 2
            elif game.steps_since_food % 11 == 0 and game.steps_since_food != 0:
                action = 2

            result = game.step(action)

            if not renderer.render(fps=10):
                is_running = False

            # Auto-reset to keep watching continuously.
            if result.done:
                game.reset()
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
