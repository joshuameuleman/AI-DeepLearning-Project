from __future__ import annotations

"""Trainingsentrypoint voor de DQN-agents.

Dit bestand vertaalt CLI-argumenten zoals `--game`, `--episodes` en `--profile`
naar een `TrainConfig`. Daarna start het de algemene `Trainer`.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    # Zorgt dat `python DQN/train.py` werkt vanaf de projectroot.
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from DQN.src.training.config import TrainConfig
from DQN.src.utils.paths import ensure_run_dirs
from DQN.src.utils.snake_config import SNAKE_DEFAULT_GRID_SIZE, apply_snake_grid_size

SUPPORTED_GAMES = ("snake", "flappy", "2048")


def _infer_qnetwork_dims_from_checkpoint(checkpoint_path: Path) -> tuple[int, int, int] | None:
    """Lees modeldimensies uit een checkpoint zonder het hele model te bouwen.

    Return:
        (input_size, hidden_size, output_size), of None als het checkpoint niet
        gelezen kan worden. Dit is nodig wanneer een bestaande checkpoint een
        andere hidden_size gebruikt dan het gekozen trainingsprofiel.
    """
    try:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        return None

    # Nieuwere checkpoints kunnen een Dueling DQN-layout gebruiken.
    # Dan zitten weights in `feature_layer` en `advantage_stream`.
    if "feature_layer.0.weight" in state_dict:
        w_in = state_dict["feature_layer.0.weight"]
        w_out = state_dict["advantage_stream.2.weight"]
        return int(w_in.shape[1]), int(w_in.shape[0]), int(w_out.shape[0])

    # Oudere standaard-DQN checkpoints hebben een `net` Sequential-layout.
    first_layer = state_dict.get("net.0.weight")
    middle_layer = state_dict.get("net.2.weight")
    last_layer = state_dict.get("net.4.weight")
    if first_layer is None or middle_layer is None or last_layer is None:
        return None

    try:
        input_size = int(first_layer.shape[1])
        hidden_size = int(first_layer.shape[0])
        output_size = int(last_layer.shape[0])
    except Exception:
        return None

    if int(middle_layer.shape[0]) != hidden_size or int(middle_layer.shape[1]) != hidden_size:
        return None

    return (input_size, hidden_size, output_size)


def _preferred_resume_checkpoint(checkpoint_dir: Path) -> Path:
    """Kies welk checkpoint het best gebruikt wordt om training te hervatten."""
    best_eval_checkpoint = checkpoint_dir / "best_eval.pth"
    if best_eval_checkpoint.exists():
        return best_eval_checkpoint
    return checkpoint_dir / "latest.pth"


def run_training(
    game: str,
    episodes: int,
    web_feed_path: Optional[str] = None,
    resume: bool = True,
    grid_size: Optional[int] = None,
    enable_live_feed: bool = False,
    profile: str = "balanced",
    cpu_threads: int = 0,
    device: str = "auto",
) -> None:
    """Configureer en start een DQN-training.

    game:
        Welke game getraind wordt.
    episodes:
        Aantal volledige episodes/runs.
    resume:
        True = hervatten vanaf checkpoint als die bestaat.
    grid_size:
        Alleen voor Snake: breedte/hoogte van het speelveld.
    profile:
        Snake-presets: fast, balanced of quality.
    device:
        "auto", "cpu" of "cuda".
    """
    resolved_grid_size: Optional[int] = None
    if game == "snake":
        # Zet SNAKE_GRID_SIZE ook in de environment, zodat de Snake-logica dezelfde maat gebruikt.
        resolved_grid_size = apply_snake_grid_size(grid_size)

    # TrainConfig bevat de algemene DQN-hyperparameters.
    # Game-specifieke overrides worden hieronder toegepast.
    cfg = TrainConfig(
        game=game,
        episodes=episodes,
        cpu_threads=max(0, int(cpu_threads)),
        device=device,
    )
    profile = profile.lower().strip()
    if profile not in ("fast", "balanced", "quality"):
        profile = "balanced"

    # Stretch epsilon schedule across most of the run instead of collapsing early.
    # decay_horizon = over hoeveel episodes epsilon ongeveer mag zakken.
    decay_horizon = max(1, int(episodes * 0.97))
    cfg.epsilon_decay = (cfg.epsilon_end / max(cfg.epsilon_start, 1e-9)) ** (1.0 / decay_horizon)

    # run_name bepaalt waar checkpoints en logs terechtkomen.
    run_name = game
    if game == "flappy":
        # Flappy learns poorly with very high epsilon for too long because random flaps
        # terminate early before ever reaching/passing pipes.
        cfg.epsilon_start = 0.35
        cfg.epsilon_end = 0.02
        # Keep exploration alive longer; 12% horizon collapsed too quickly on long runs.
        decay_horizon = max(12_000, int(episodes * 0.55))
        cfg.epsilon_decay = (cfg.epsilon_end / max(cfg.epsilon_start, 1e-9)) ** (1.0 / decay_horizon)

        # Flappy-specifieke tuning: sneller leren, kleinere hidden layer,
        # en langere evaluaties zodat pipes passeren meetbaar wordt.
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

        # Snake gebruikt een lage eind-epsilon en action masking:
        # de agent mag leren, maar hoeft geen duidelijk dodelijke acties te proberen.
        cfg.epsilon_end = 0.005
        cfg.mask_unsafe_actions = True

        # Hogere gamma helpt Snake met lange-termijnplanning: niet alleen de volgende appel,
        # maar ook ruimte houden om later niet vast te lopen.
        cfg.gamma = max(cfg.gamma, 0.995)
        cfg.learning_rate = min(cfg.learning_rate, 2.0e-4)
        cfg.checkpoint_every_episodes = 50 if profile == "fast" else 25
        cfg.web_feed_every_n_steps = 100 if profile == "fast" else 25
        cfg.eval_enabled = True
        decay_horizon = max(1, int(episodes * 0.97))
        cfg.epsilon_decay = (cfg.epsilon_end / max(cfg.epsilon_start, 1e-9)) ** (1.0 / decay_horizon)

        # Large grids need a fill-cap, not a short survival cap. A 32x32 board
        # needs at least 1021 growth events, and random food can require many
        # laps around a safe cycle before the final cells are reached.
        board_cells = resolved_grid_size * resolved_grid_size
        cfg.max_steps_per_episode = max(
            cfg.max_steps_per_episode,
            board_cells * min(board_cells, 256),
        )
        cfg.eval_max_steps = min(cfg.max_steps_per_episode, board_cells * 64)
        cfg.eval_episodes = 25 if profile == "fast" else 50
        cfg.eval_every_episodes = max(100, min(1_000, episodes // 100 if episodes >= 10_000 else episodes))
        cfg.save_best_eval_checkpoint = True
        cfg.restore_best_eval_on_regression = True
        cfg.eval_regression_ratio = 0.75
        cfg.eval_regression_min_gap = 50.0
        cfg.learn_every_n_steps = 2
        cfg.target_update_every_episodes = 2
        cfg.learning_starts = max(6_000, resolved_grid_size * 96)
        cfg.memory_size = min(max(cfg.memory_size, resolved_grid_size * resolved_grid_size * 24), 750_000)

        # Profielen passen snelheid/kwaliteit aan zonder dat je handmatig alle hyperparameters wijzigt.
        if profile == "fast":
            # Fast: lager geheugengebruik en minder vaak leren, handig om snel te testen.
            cfg.hidden_size = 192
            cfg.batch_size = 96
            cfg.learn_every_n_steps = 4
            cfg.memory_size = min(cfg.memory_size, 200_000)
            cfg.learning_starts = max(2_000, resolved_grid_size * 32)
            cfg.max_steps_per_episode = max(cfg.max_steps_per_episode // 2, board_cells * 64)
            os.environ.setdefault("SNAKE_SPACE_REWARD_EVERY", "16" if resolved_grid_size <= 32 else "64")
        elif profile == "balanced":
            # Balanced: standaard voor normale runs.
            cfg.hidden_size = max(cfg.hidden_size, 256)
            cfg.batch_size = max(cfg.batch_size, 192)
            cfg.learn_every_n_steps = 3
            os.environ.setdefault("SNAKE_SPACE_REWARD_EVERY", "8" if resolved_grid_size <= 32 else "32")
        else:
            # Quality: grotere batches en hidden layer, dus zwaarder maar mogelijk beter.
            cfg.hidden_size = max(cfg.hidden_size, 384)
            cfg.batch_size = max(cfg.batch_size, 256)

        if cfg.device == "cuda":
            # Op GPU kunnen grotere batches vaak efficienter verwerkt worden.
            if profile == "fast":
                cfg.batch_size = max(cfg.batch_size, 192)
            elif profile == "balanced":
                cfg.batch_size = max(cfg.batch_size, 256)
            else:
                cfg.batch_size = max(cfg.batch_size, 512)
        if resolved_grid_size >= 64:
            # Small anti-plateau tuning for large Snake boards:
            # - lower LR to reduce policy oscillation in late-game body navigation
            # - keep a tiny bit more exploration to escape self-collision loops
            # Plan C: slightly reduce residual exploration for stronger exploitation.
            cfg.epsilon_end = max(cfg.epsilon_end, 0.007)
            cfg.epsilon_decay = (cfg.epsilon_end / max(cfg.epsilon_start, 1e-9)) ** (1.0 / decay_horizon)
            # Plan B: emphasize long-horizon planning and smooth target drift.
            cfg.learning_starts = max(cfg.learning_starts, 8_000)
        if resolved_grid_size >= 128:
            # Zeer grote Snake-boards krijgen extra memory en langere warmup.
            cfg.hidden_size = 384
            cfg.batch_size = 256
            cfg.memory_size = max(cfg.memory_size, 500_000)
            cfg.learning_starts = max(cfg.learning_starts, 12_000)
            cfg.learn_every_n_steps = 4
            cfg.web_feed_every_n_steps = 50
            cfg.target_update_every_episodes = 2

    ckpt_dir, logs_dir = ensure_run_dirs(run_name)
    resume_checkpoint = _preferred_resume_checkpoint(ckpt_dir)
    if resume and resume_checkpoint.exists():
        # Bij resume moet de netwerkvorm overeenkomen met het checkpoint.
        # Daarom nemen we hidden_size over uit het checkpoint als dat nodig is.
        inferred_dims = _infer_qnetwork_dims_from_checkpoint(resume_checkpoint)
        if inferred_dims is not None:
            ckpt_state_size, ckpt_hidden_size, ckpt_action_count = inferred_dims
            expected_action_count = 3 if game == "snake" else (2 if game == "flappy" else 4)
            if ckpt_action_count == expected_action_count:
                if cfg.hidden_size != ckpt_hidden_size:
                    print(
                        f"[DQN] Checkpoint model shape detected; using hidden_size={ckpt_hidden_size} "
                        f"instead of profile hidden_size={cfg.hidden_size} for resume."
                    )
                cfg.hidden_size = ckpt_hidden_size
            else:
                print(
                    f"[DQN] Checkpoint action size ({ckpt_action_count}) does not match "
                    f"expected action size ({expected_action_count}); resume may start fresh."
                )

    print(f"[DQN] Training start for game={cfg.game}")
    print(f"[DQN] Episodes: {cfg.episodes}")
    print(f"[DQN] Epsilon schedule: start={cfg.epsilon_start:.4f}, end={cfg.epsilon_end:.4f}, decay={cfg.epsilon_decay:.6f}")
    if game == "snake":
        assert resolved_grid_size is not None
        print(f"[DQN] Grid size: {resolved_grid_size}x{resolved_grid_size}")
        print(f"[DQN] Snake profile: {profile}")
    print(f"[DQN] Checkpoints: {ckpt_dir}")
    print(f"[DQN] Logs: {logs_dir}")
    print(f"[DQN] Resume mode: {'ON' if resume else 'OFF (fresh start)'}")
    if resume and resume_checkpoint.exists():
        print(f"[DQN] Resume checkpoint preference: {resume_checkpoint}")
    print(f"[DQN] Algorithm: {'Double DQN' if cfg.double_dqn else 'DQN'}")
    print(f"[DQN] Device: {cfg.device}")
    if cfg.cpu_threads > 0:
        print(f"[DQN] CPU threads: {cfg.cpu_threads}")
    print(f"[DQN] Live feed: {'ON' if enable_live_feed else 'OFF'}")
    if web_feed_path:
        print(f"[DQN] Web feed: {web_feed_path}")

    from DQN.src.training.trainer import Trainer

    # De Trainer bevat de echte trainingsloop: env resetten, acties kiezen,
    # transitions opslaan, batches leren, evalueren en checkpoints schrijven.
    trainer = Trainer(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        logs_dir=logs_dir,
        web_feed_path=Path(web_feed_path) if web_feed_path else None,
        resume=resume,
        enable_live_feed=enable_live_feed,
    )
    print(f"[DQN] Runtime device: {trainer.device}")
    if trainer.device.type == "cuda":
        import torch

        device_index = trainer.device.index if trainer.device.index is not None else 0
        print(f"[DQN] CUDA GPU: {torch.cuda.get_device_name(device_index)}")
        print(f"[DQN] CUDA memory allocated: {torch.cuda.memory_allocated(device_index) / 1024**2:.1f} MiB")
    result = trainer.train()

    print(f"[DQN] Episodes trained: {result.episodes}")
    print(f"[DQN] Best reward: {result.best_reward}")
    print(f"[DQN] Final epsilon: {result.final_epsilon:.4f}")
    print(f"[DQN] Checkpoint saved: {result.checkpoint_path}")
    if result.best_eval_checkpoint_path:
        print(f"[DQN] Best eval checkpoint saved: {result.best_eval_checkpoint_path}")

def prompt_game() -> str:
    """Vraag interactief welke game getraind moet worden."""
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

def prompt_episodes(default: int = 1000) -> int:
    """Vraag interactief hoeveel episodes getraind worden."""
    raw = input(f"Aantal episodes [{default}]: ").strip()
    if not raw:
        return default
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    print(f"Ongeldige invoer, standaardwaarde {default} wordt gebruikt.")
    return default

def parse_args() -> argparse.Namespace:
    """Definieer CLI-argumenten voor training."""
    parser = argparse.ArgumentParser(description="Train a DQN agent.")
    parser.add_argument("--game", default=None, choices=list(SUPPORTED_GAMES))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="Start from scratch and ignore latest checkpoint")
    parser.add_argument("--grid-size", type=int, default=None, help=f"Snake grid size (default {SNAKE_DEFAULT_GRID_SIZE})")
    parser.add_argument("--profile", choices=("fast", "balanced", "quality"), default="balanced", help="Snake training speed/quality preset")
    parser.add_argument("--cpu-threads", type=int, default=0, help="Limit PyTorch CPU threads (0 = PyTorch default)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Training device")
    return parser.parse_args()

def main() -> None:
    """Lees CLI-argumenten of prompts en start training."""
    args = parse_args()
    game = args.game or prompt_game()
    episodes = args.episodes if args.episodes is not None else prompt_episodes()
    run_training(
        game=game,
        episodes=episodes,
        resume=not args.fresh,
        grid_size=args.grid_size,
        profile=args.profile,
        cpu_threads=args.cpu_threads,
        device=args.device,
    )


if __name__ == "__main__":
    main()
