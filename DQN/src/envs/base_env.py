from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Sequence

# Common environment interface used by DQN training and visualization.
@dataclass(frozen=True)
class StepOutcome:
    state: Any
    reward: float
    done: bool
    info: Dict[str, Any]

# Common environment interface used by DQN training and visualization.
class BaseEnv(ABC):
    """Common environment interface used by DQN training and visualization."""

    game_name: str

    @abstractmethod
    def reset(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> StepOutcome:
        raise NotImplementedError

    @abstractmethod
    def action_space(self) -> Sequence[int]:
        raise NotImplementedError

    def render(self) -> None:
        """Optional human-readable render hook."""
        return None
