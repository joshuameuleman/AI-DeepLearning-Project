"""Core game loop logic for 2048."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class StepResult:
    state: Any
    reward: float
    done: bool
    info: Dict[str, Any]


class Game2048Logic:
    """Pure game logic: rules, state transitions and terminal checks."""

    def reset(self) -> Any:
        raise NotImplementedError("Implement initial state for 2048")

    def step(self, action: int) -> StepResult:
        raise NotImplementedError("Implement step transition for 2048")

    def get_state(self) -> Any:
        raise NotImplementedError("Implement state extraction for 2048")

    def action_space(self) -> Tuple[int, ...]:
        return (0, 1, 2, 3)
