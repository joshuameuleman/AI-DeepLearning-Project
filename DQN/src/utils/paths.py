from __future__ import annotations

from pathlib import Path
from typing import Tuple


def dqn_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_run_dirs(game_name: str) -> Tuple[Path, Path]:
    root = dqn_root()
    ckpt_dir = root / "checkpoints" / game_name
    logs_dir = root / "logs" / game_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, logs_dir
