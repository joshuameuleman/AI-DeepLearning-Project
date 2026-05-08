from __future__ import annotations

import copy
import threading
import time
from typing import Any, Optional, Tuple


_condition = threading.Condition()
_version = 0
_latest_payload: Optional[dict[str, Any]] = None


def publish_state(payload: dict[str, Any]) -> None:
    global _version, _latest_payload
    with _condition:
        _latest_payload = copy.deepcopy(payload)
        _version += 1
        _condition.notify_all()


def current_state() -> Tuple[int, Optional[dict[str, Any]]]:
    with _condition:
        return _version, copy.deepcopy(_latest_payload)


def wait_for_update(last_version: int, timeout: float) -> Tuple[int, Optional[dict[str, Any]]]:
    with _condition:
        if _version == last_version:
            _condition.wait(timeout=timeout)
        return _version, copy.deepcopy(_latest_payload)


def build_snake_payload(
    logic: Any,
    *,
    training: bool,
    game: str,
    episode: int,
    total_episodes: int,
    step: int,
    episode_reward: float,
    epsilon: float,
    simulating: bool = False,
    done: bool = False,
    board_filled_count: int = 0,
    wall_collision_count: int = 0,
    self_collision_count: int = 0,
) -> Optional[dict[str, Any]]:
    """Create a normalized Snake payload for web live-feed publishing."""
    if logic is None or not hasattr(logic, "body") or not hasattr(logic, "food"):
        return None

    primary_food = getattr(logic, "food", None) or (0, 0)
    payload: dict[str, Any] = {
        "training": bool(training),
        "game": str(game),
        "episode": int(episode),
        "totalEpisodes": int(total_episodes),
        "step": int(step),
        "score": float(getattr(logic, "score", 0.0)),
        "episodeReward": float(episode_reward),
        "epsilon": float(epsilon),
        "gridWidth": int(getattr(logic, "GRID_WIDTH", 10)),
        "gridHeight": int(getattr(logic, "GRID_HEIGHT", 10)),
        "snake": [{"x": int(x), "y": int(y)} for x, y in list(getattr(logic, "body", []))],
        "foods": [{"x": int(x), "y": int(y)} for x, y in list(getattr(logic, "foods", []))],
        "food": {
            "x": int(primary_food[0]),
            "y": int(primary_food[1]),
        },
        "foodCount": int(len(getattr(logic, "foods", []))),
        "targetFoodCount": int(getattr(logic, "target_food_count", 1)),
        "boardFilledCount": int(board_filled_count),
        "wallCollisionCount": int(wall_collision_count),
        "selfCollisionCount": int(self_collision_count),
    }

    if simulating:
        payload["simulating"] = True
        payload["done"] = bool(done)
    payload["updatedAt"] = float(time.time())

    return payload


def build_flappy_payload(
    logic: Any,
    *,
    training: bool,
    game: str,
    episode: int,
    total_episodes: int,
    step: int,
    episode_reward: float,
    epsilon: float,
    simulating: bool = False,
    done: bool = False,
) -> Optional[dict[str, Any]]:
    """Create a normalized Flappy payload for web live-feed publishing."""
    if logic is None or not hasattr(logic, "pipes") or not hasattr(logic, "bird_y"):
        return None

    pipes = []
    for pipe in list(getattr(logic, "pipes", [])):
        try:
            pipes.append(
                {
                    "x": float(pipe.get("x", 0.0)),
                    "gapY": float(pipe.get("gap_y", 0.0)),
                    "passed": bool(pipe.get("passed", False)),
                }
            )
        except Exception:
            continue

    payload: dict[str, Any] = {
        "training": bool(training),
        "game": str(game),
        "episode": int(episode),
        "totalEpisodes": int(total_episodes),
        "step": int(step),
        "score": float(getattr(logic, "score", 0.0)),
        "episodeReward": float(episode_reward),
        "epsilon": float(epsilon),
        "screenWidth": int(getattr(logic, "SCREEN_WIDTH", 288)),
        "screenHeight": int(getattr(logic, "SCREEN_HEIGHT", 512)),
        "pipeWidth": float(getattr(logic, "PIPE_WIDTH", 52.0)),
        "pipeGap": float(getattr(logic, "PIPE_GAP", 120.0)),
        "bird": {
            "x": float(getattr(logic, "BIRD_X", 56.0)),
            "y": float(getattr(logic, "bird_y", 0.0)),
            "radius": float(getattr(logic, "BIRD_RADIUS", 12.0)),
            "velocity": float(getattr(logic, "bird_velocity", 0.0)),
        },
        "pipes": pipes,
        "updatedAt": float(time.time()),
    }

    if simulating:
        payload["simulating"] = True
        payload["done"] = bool(done)

    return payload