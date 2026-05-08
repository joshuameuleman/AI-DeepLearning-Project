from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from DQN.src.envs.game_catalog import GAME_LOGIC_FILE_BY_GAME, SUPPORTED_GAMES
from DQN.src.envs.base_env import BaseEnv, StepOutcome

# Adapter around a game-specific logic module.
@dataclass
class GameMetadata:
    game_name: str
    logic_path: Path
    action_count: int
    state_size: int

# The actual game logic can be implemented inside Games/<game>/logic/game_logic.py.
class GameEnvironment(BaseEnv):
    """Adapter around a game-specific logic module.

    The actual game logic can be implemented inside Games/<game>/logic/game_logic.py.
    For now this adapter provides a stable RL interface and a graceful fallback.
    """
    # Initialize the game environment by loading the corresponding game logic module if available.
    def __init__(self, game_name: str, allow_fallback: bool = True) -> None:
        normalized_game = game_name.lower().strip()
        if normalized_game not in SUPPORTED_GAMES:
            raise ValueError(f"Unsupported game '{game_name}'. Supported values: {', '.join(SUPPORTED_GAMES)}")

        self.game_name = normalized_game
        self.allow_fallback = allow_fallback
        self.metadata = self._build_metadata(normalized_game)
        self._logic = self._load_logic()
        self._step_count = 0
        self._state = self.reset()
    # Build game metadata based on the game name, including logic file path, action count and state size.
    def _build_metadata(self, game_name: str) -> GameMetadata:
        if game_name == "snake":
            return GameMetadata(game_name, GAME_LOGIC_FILE_BY_GAME[game_name], action_count=3, state_size=11)
        if game_name == "flappy":
            return GameMetadata(game_name, GAME_LOGIC_FILE_BY_GAME[game_name], action_count=2, state_size=8)
        return GameMetadata(game_name, GAME_LOGIC_FILE_BY_GAME[game_name], action_count=4, state_size=16)
    # Dynamically load the game logic module from the specified path, if it exists.
    def _load_logic(self) -> Optional[Any]:
        logic_path = self.metadata.logic_path
        if not logic_path.exists():
            return None

        spec = importlib.util.spec_from_file_location(f"{self.game_name}_logic", logic_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    # Reset the environment to the initial state using the game logic if available, otherwise use a fallback state.
    def reset(self) -> Any:
        self._step_count = 0
        if self._logic is not None:
            for candidate in ("SnakeLogic", "FlappyBirdLogic", "Game2048Logic"):
                logic_class = getattr(self._logic, candidate, None)
                if logic_class is not None:
                    try:
                        logic_instance = logic_class()
                        if hasattr(logic_instance, "reset"):
                            self._logic_instance = logic_instance
                            self._state = logic_instance.reset()
                            if hasattr(self._state, "__len__"):
                                self.metadata.state_size = int(len(self._state))
                            return self._state
                    except Exception as exc:
                        print(f"[DQN] Warning: failed to initialize logic class '{candidate}': {exc}")
                        self._logic_instance = None

        if not self.allow_fallback:
            if self._logic is None:
                raise RuntimeError(
                    f"[DQN] Game logic module missing for '{self.game_name}' at {self.metadata.logic_path}. "
                    "Fallback is disabled."
                )
            raise RuntimeError(
                f"[DQN] Could not initialize logic class for '{self.game_name}'. "
                "Fallback is disabled."
            )

        self._logic_instance = None
        self._state = [0] * self.metadata.state_size
        return self._state
    # Step the environment using the game logic if available, otherwise use a fallback reward and state transition.
    def step(self, action: int) -> StepOutcome:
        self._step_count += 1
        if getattr(self, "_logic_instance", None) is not None:
            try:
                raw_outcome = self._logic_instance.step(action)
                return StepOutcome(
                    state=raw_outcome.state,
                    reward=float(raw_outcome.reward),
                    done=bool(raw_outcome.done),
                    info=dict(raw_outcome.info),
                )
            except Exception as exc:
                print(f"[DQN] Warning: logic step failed, switching to fallback env: {exc}")
                self._logic_instance = None
                if not self.allow_fallback:
                    raise RuntimeError(
                        f"[DQN] Logic step failed for '{self.game_name}'. Fallback is disabled."
                    ) from exc

        if not self.allow_fallback:
            raise RuntimeError(
                f"[DQN] Logic instance unavailable for '{self.game_name}'. Fallback is disabled."
            )

        reward = 1.0 if action == 0 else -0.1
        done = self._step_count >= 25
        self._state = [action, self._step_count] + [0] * max(0, self.metadata.state_size - 2)
        return StepOutcome(state=self._state, reward=reward, done=done, info={"fallback": True})
    # Return the action space as a tuple of valid action indices based on the game metadata.
    def action_space(self):
        return tuple(range(self.metadata.action_count))
