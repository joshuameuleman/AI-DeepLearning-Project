from __future__ import annotations

import os


SNAKE_DEFAULT_GRID_SIZE = 32
SNAKE_MIN_GRID_SIZE = 4


def _is_valid_grid_size(value: int | None) -> bool:
    return value is not None and int(value) >= SNAKE_MIN_GRID_SIZE


def resolve_snake_grid_size(explicit: int | None = None) -> int:
    """Resolve grid size from explicit value, environment or project default."""
    if _is_valid_grid_size(explicit):
        return int(explicit)

    env_grid = os.environ.get("SNAKE_GRID_SIZE", "").strip()
    if env_grid.isdigit() and _is_valid_grid_size(int(env_grid)):
        return int(env_grid)

    return SNAKE_DEFAULT_GRID_SIZE


def apply_snake_grid_size(explicit: int | None = None) -> int:
    """Resolve and persist the grid size in SNAKE_GRID_SIZE for downstream logic."""
    grid_size = resolve_snake_grid_size(explicit)
    os.environ["SNAKE_GRID_SIZE"] = str(grid_size)
    return grid_size
