from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from DQN.src.training.config import TrainConfig
from DQN.src.training.trainer import Trainer
from DQN.src.utils.paths import ensure_run_dirs
from DQN.src.utils.snake_config import SNAKE_DEFAULT_GRID_SIZE, apply_snake_grid_size

SUPPORTED_GAMES = ("snake", "flappy", "2048")


def run_training(
    game: str,
    episodes: int,
    web_feed_path: Optional[str] = None,
    resume: bool = True,
    grid_size: Optional[int] = None,
    enable_live_feed: bool = False,
) -> None:
    resolved_grid_size: Optional[int] = None
    if game == "snake":
        resolved_grid_size = apply_snake_grid_size(grid_size)

    cfg = TrainConfig(game=game, episodes=episodes)
    # Stretch epsilon schedule across most of the run instead of collapsing early.
    decay_horizon = max(1, int(episodes * 0.97))
    cfg.epsilon_decay = (cfg.epsilon_end / max(cfg.epsilon_start, 1e-9)) ** (1.0 / decay_horizon)
    run_name = game
    if game == "flappy":
        # Flappy learns poorly with very high epsilon for too long because random flaps
        # terminate early before ever reaching/passing pipes.
        cfg.epsilon_start = 0.35
        cfg.epsilon_end = 0.02
        # Keep exploration alive longer; 12% horizon collapsed too quickly on long runs.
        decay_horizon = max(12_000, int(episodes * 0.55))
        cfg.epsilon_decay = (cfg.epsilon_end / max(cfg.epsilon_start, 1e-9)) ** (1.0 / decay_horizon)
        cfg.learning_rate = 1.5e-4
        cfg.hidden_size = 192
        cfg.batch_size = 96
        cfg.learning_starts = 1_000
        cfg.learn_every_n_steps = 2
        cfg.target_update_every_episodes = 6
        cfg.max_steps_per_episode = 6_000
        cfg.eval_episodes = 50
        cfg.eval_max_steps = 12_000
        cfg.eval_every_episodes = 500
        cfg.save_best_eval_checkpoint = True

    if game == "snake":
        assert resolved_grid_size is not None
        run_name = f"snake_{resolved_grid_size}x{resolved_grid_size}"
        cfg.epsilon_end = 0.005
        decay_horizon = max(1, int(episodes * 0.97))
        cfg.epsilon_decay = (cfg.epsilon_end / max(cfg.epsilon_start, 1e-9)) ** (1.0 / decay_horizon)
        # Large grids need an area-based cap; otherwise episodes end far too early.
        cfg.max_steps_per_episode = max(cfg.max_steps_per_episode, resolved_grid_size * resolved_grid_size * 4)
        cfg.learn_every_n_steps = 3
        cfg.target_update_every_episodes = 6
        cfg.learning_starts = max(2_000, resolved_grid_size * 32)
        cfg.memory_size = min(max(cfg.memory_size, resolved_grid_size * resolved_grid_size * 16), 500_000)
        if resolved_grid_size >= 64:
            # Small anti-plateau tuning for large Snake boards:
            # - lower LR to reduce policy oscillation in late-game body navigation
            # - keep a tiny bit more exploration to escape self-collision loops
            cfg.learning_rate = min(cfg.learning_rate, 2.0e-4)
            # Plan C: slightly reduce residual exploration for stronger exploitation.
            cfg.epsilon_end = max(cfg.epsilon_end, 0.007)
            cfg.epsilon_decay = (cfg.epsilon_end / max(cfg.epsilon_start, 1e-9)) ** (1.0 / decay_horizon)
            # Plan B: emphasize long-horizon planning and smooth target drift.
            cfg.gamma = max(cfg.gamma, 0.995)
            cfg.hidden_size = max(cfg.hidden_size, 384)
            cfg.batch_size = max(cfg.batch_size, 192)
            cfg.learning_starts = max(cfg.learning_starts, 4_000)
            cfg.learn_every_n_steps = 2
            cfg.target_update_every_episodes = 5
        if resolved_grid_size >= 128:
            cfg.hidden_size = 384
            cfg.batch_size = 192
            cfg.memory_size = max(cfg.memory_size, 400_000)
            cfg.learning_starts = max(cfg.learning_starts, 8_000)
            cfg.learn_every_n_steps = 4
            cfg.web_feed_every_n_steps = 10
            cfg.target_update_every_episodes = 8

    ckpt_dir, logs_dir = ensure_run_dirs(run_name)
    print(f"[DQN] Training start for game={cfg.game}")
    print(f"[DQN] Episodes: {cfg.episodes}")
    print(f"[DQN] Epsilon schedule: start={cfg.epsilon_start:.4f}, end={cfg.epsilon_end:.4f}, decay={cfg.epsilon_decay:.6f}")
    if game == "snake":
        assert resolved_grid_size is not None
        print(f"[DQN] Grid size: {resolved_grid_size}x{resolved_grid_size}")
    print(f"[DQN] Checkpoints: {ckpt_dir}")
    print(f"[DQN] Logs: {logs_dir}")
    print(f"[DQN] Resume mode: {'ON' if resume else 'OFF (fresh start)'}")
    print(f"[DQN] Device: {cfg.device}")
    print(f"[DQN] Live feed: {'ON' if enable_live_feed else 'OFF'}")
    if web_feed_path:
        print(f"[DQN] Web feed: {web_feed_path}")

    trainer = Trainer(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        logs_dir=logs_dir,
        web_feed_path=Path(web_feed_path) if web_feed_path else None,
        resume=resume,
        enable_live_feed=enable_live_feed,
    )
    print(f"[DQN] Runtime device: {trainer.device}")
    result = trainer.train()

    print(f"[DQN] Episodes trained: {result.episodes}")
    print(f"[DQN] Best reward: {result.best_reward}")
    print(f"[DQN] Final epsilon: {result.final_epsilon:.4f}")
    print(f"[DQN] Checkpoint saved: {result.checkpoint_path}")

# Interactive prompts for missing arguments
def prompt_game() -> str:
    print("Kies een game om te trainen:")
    for index, game in enumerate(SUPPORTED_GAMES, start=1):
        print(f"  {index}. {game}")

    valid_by_index = {str(i): game for i, game in enumerate(SUPPORTED_GAMES, start=1)}

    while True:
        user_input = input("Jouw keuze (nummer of naam): ").strip().lower()
        if user_input in valid_by_index:
            return valid_by_index[user_input]
        if user_input in SUPPORTED_GAMES:
            return user_input
        print(f"Ongeldige keuze. Gebruik een nummer 1-{len(SUPPORTED_GAMES)} of een geldige naam.")

# Prompt for number of episodes with validation
def prompt_episodes(default: int = 1000) -> int:
    raw = input(f"Aantal episodes [{default}]: ").strip()
    if not raw:
        return default
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    print(f"Ongeldige invoer, standaardwaarde {default} wordt gebruikt.")
    return default

# Argument parsing with optional overrides
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent.")
    parser.add_argument("--game", default=None, choices=list(SUPPORTED_GAMES))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="Start from scratch and ignore latest checkpoint")
    parser.add_argument("--grid-size", type=int, default=None, help=f"Snake grid size (default {SNAKE_DEFAULT_GRID_SIZE})")
    return parser.parse_args()

# Main entry point
def main() -> None:
    args = parse_args()
    game = args.game or prompt_game()
    episodes = args.episodes if args.episodes is not None else prompt_episodes()
    run_training(game=game, episodes=episodes, resume=not args.fresh, grid_size=args.grid_size)


if __name__ == "__main__":
    main()
