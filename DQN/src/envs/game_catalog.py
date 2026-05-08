from __future__ import annotations

from DQN.src.utils.paths import workspace_root

SUPPORTED_GAMES = ("snake", "flappy", "2048")

ROOT_DIR = workspace_root()

GAME_LOGIC_FILE_BY_GAME = {
    "snake": ROOT_DIR / "Games" / "Snake" / "logic" / "game_logic.py",
    "flappy": ROOT_DIR / "Games" / "Flappy Bird" / "logic" / "game_logic.py",
    "2048": ROOT_DIR / "Games" / "2048" / "logic" / "game_logic.py",
}
