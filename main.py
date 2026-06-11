"""Interactive AI Deep Learning Project launcher."""

from __future__ import annotations

import importlib.util
import sys
import webbrowser
from pathlib import Path

from DQN.simulate import SUPPORTED_GAMES, run_simulation
from DQN.src.utils.snake_config import SNAKE_DEFAULT_GRID_SIZE

SUPPORTED_MODES = ("play", "simulate", "visualize", "train")


def prompt_mode() -> str:
    print("Kies een modus:")
    for index, mode in enumerate(SUPPORTED_MODES, start=1):
        print(f"  {index}. {mode}")

    valid_by_index = {str(i): mode for i, mode in enumerate(SUPPORTED_MODES, start=1)}

    while True:
        user_input = input("Jouw keuze (nummer of naam): ").strip().lower()
        if user_input in valid_by_index:
            return valid_by_index[user_input]
        if user_input in ("spelen",):
            return "play"
        if user_input in ("meekijken", "watch"):
            return "visualize"
        if user_input in SUPPORTED_MODES:
            return user_input
        print(f"Ongeldige keuze. Gebruik een nummer 1-{len(SUPPORTED_MODES)} of een geldige naam.")


def prompt_game() -> str:
    print("Kies een game:")
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


def prompt_episodes(default: int = 5) -> int:
    raw = input(f"Aantal episodes [{default}]: ").strip()
    if not raw:
        return default
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    print(f"Ongeldige invoer, standaardwaarde {default} wordt gebruikt.")
    return default


def prompt_checkpoint(default: str = "auto") -> str:
    raw = input(f"Checkpoint [{default}]: ").strip()
    return raw if raw else default


def prompt_solver(game: str, default: str = "dqn") -> str:
    if game != "snake":
        return "dqn"

    print("Snake solver (optioneel):")
    print("  1. dqn (reinforcement learning checkpoint, aanbevolen)")
    print("  2. hamiltonian (optionele benchmark, geen training)")
    while True:
        raw = input(f"Jouw keuze [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("1", "dqn", "ai", "checkpoint", "deeplearning", "deep learning"):
            return "dqn"
        if raw in ("2", "hamiltonian", "solver", "benchmark"):
            return "hamiltonian"
        print("Ongeldige keuze. Gebruik 1/dqn of 2/hamiltonian.")


def prompt_training_strategy() -> bool:
    print("Trainingsmodus:")
    print("  1. verder zetten (resume vanaf best_eval checkpoint als die bestaat)")
    print("  2. opnieuw beginnen (fresh start)")
    while True:
        raw = input("Jouw keuze [1]: ").strip().lower()
        if raw in ("", "1", "verder", "resume"):
            return True
        if raw in ("2", "opnieuw", "fresh"):
            return False
        print("Ongeldige keuze. Gebruik 1 of 2.")


def prompt_live_follow(default: bool = True) -> bool:
    default_label = "ja" if default else "nee"
    raw = input(f"Live meevolgen? (ja/nee) [{default_label}]: ").strip().lower()
    if not raw:
        return default
    if raw in ("j", "ja", "y", "yes"):
        return True
    if raw in ("n", "nee", "no"):
        return False
    print(f"Ongeldige invoer, standaardwaarde '{default_label}' wordt gebruikt.")
    return default


def prompt_open_browser(default: bool = False) -> bool:
    default_label = "ja" if default else "nee"
    raw = input(f"Browser openen? (ja/nee) [{default_label}]: ").strip().lower()
    if not raw:
        return default
    if raw in ("j", "ja", "y", "yes"):
        return True
    if raw in ("n", "nee", "no"):
        return False
    print(f"Ongeldige invoer, standaardwaarde '{default_label}' wordt gebruikt.")
    return default


def prompt_snake_grid_size(default: int = SNAKE_DEFAULT_GRID_SIZE) -> int:
    raw = input(f"Snake grid grootte [{default}] (bijv. 64/128/256): ").strip()
    if not raw:
        return default
    if raw.isdigit() and int(raw) > 3:
        return int(raw)
    print(f"Ongeldige invoer, standaardwaarde {default} wordt gebruikt.")
    return default


def prompt_training_profile(game: str, default: str = "balanced") -> str:
    if game != "snake":
        return "balanced"

    print("Snake trainingsprofiel:")
    print("  1. fast - kies dit om snel te testen of je CPU te sparen")
    print("  2. balanced - kies dit voor normale training en goede snelheid/kwaliteit")
    print("  3. quality - kies dit voor langere runs met meer kans op betere board-fill")
    while True:
        raw = input(f"Jouw keuze [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("1", "fast", "snel"):
            return "fast"
        if raw in ("2", "balanced", "balance", "normaal"):
            return "balanced"
        if raw in ("3", "quality", "kwaliteit", "beste"):
            return "quality"
        print("Ongeldige keuze. Gebruik 1/fast, 2/balanced of 3/quality.")


def prompt_cpu_threads(default: int = 1) -> int:
    print("CPU threads voor training:")
    print("  0. PyTorch kiest zelf (kan veel CPU gebruiken)")
    print("  1. rustig voor je systeem, meestal beste keuze tijdens werken")
    print("  2-4. sneller als je genoeg cores hebt, maar zwaarder voor je systeem")
    raw = input(f"Aantal CPU threads [{default}]: ").strip()
    if not raw:
        return default
    if raw.isdigit() and int(raw) >= 0:
        return int(raw)
    print(f"Ongeldige invoer, standaardwaarde {default} wordt gebruikt.")
    return default


def prompt_training_device(default: str = "auto") -> str:
    print("Training device:")
    print("  1. auto - gebruikt CUDA als PyTorch je GPU ziet")
    print("  2. cuda - forceer GPU; geeft fout als CUDA niet beschikbaar is")
    print("  3. cpu - forceer CPU")
    while True:
        raw = input(f"Jouw keuze [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("1", "auto"):
            return "auto"
        if raw in ("2", "cuda", "gpu"):
            return "cuda"
        if raw in ("3", "cpu"):
            return "cpu"
        print("Ongeldige keuze. Gebruik 1/auto, 2/cuda of 3/cpu.")


def load_and_run_training(
    game: str,
    episodes: int,
    resume: bool,
    grid_size: int | None = None,
    live_follow: bool = True,
    profile: str = "balanced",
    cpu_threads: int = 0,
    device: str = "auto",
    open_browser: bool = False,
) -> None:
    """Load and run training from DQN/train.py with optional web feed output."""
    train_file = Path(__file__).parent / "DQN" / "train.py"

    if game in ("snake", "flappy") and live_follow:
        try:
            from serve_web import start_server_in_thread

            start_server_in_thread(host="0.0.0.0", port=8000)
            if open_browser:
                webbrowser.open("http://127.0.0.1:8000/web/")
        except OSError as exc:
            print(f"Web server could not start on port 8000: {exc}")
    
    if not train_file.exists():
        print(f"Training script not found at {train_file}")
        return
    
    spec = importlib.util.spec_from_file_location("dqn_train", train_file)
    if spec is None or spec.loader is None:
        print("Could not load training script")
        return
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["dqn_train"] = module
    spec.loader.exec_module(module)
    
    if hasattr(module, 'run_training'):
        try:
            module.run_training(
                game=game,
                episodes=episodes,
                resume=resume,
                grid_size=grid_size,
                enable_live_feed=live_follow,
                profile=profile,
                cpu_threads=cpu_threads,
                device=device,
            )
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()


def load_and_run_snake_play(grid_size: int | None = None) -> None:
    """Start the manual Pygame Snake mode."""
    try:
        from Games.Snake.play import run_game

        run_game(grid_size=grid_size)
    except ImportError as exc:
        print(f"Snake play mode could not start because pygame/imports are missing: {exc}")
    except RuntimeError as exc:
        print(f"Snake play mode could not start: {exc}")


def main() -> None:
    """Main entry point for the AI Deep Learning Project."""
    mode = prompt_mode()
    game = prompt_game()

    if mode == "play":
        if game != "snake":
            print("Play mode is momenteel alleen beschikbaar voor snake.")
            return
        grid_size = prompt_snake_grid_size()
        load_and_run_snake_play(grid_size)
    elif mode == "train":
        resume = prompt_training_strategy()
        live_follow = prompt_live_follow(default=False)
        open_browser = prompt_open_browser(default=False) if live_follow else False
        episodes = prompt_episodes(default=5)
        grid_size = prompt_snake_grid_size() if game == "snake" else None
        profile = prompt_training_profile(game)
        cpu_threads = prompt_cpu_threads(default=1 if game == "snake" else 0)
        device = prompt_training_device()
        load_and_run_training(
            game,
            episodes,
            resume,
            grid_size,
            live_follow,
            profile,
            cpu_threads,
            device,
            open_browser,
        )
    else:
        episodes = prompt_episodes(default=1 if mode == "visualize" else 5)
        grid_size = prompt_snake_grid_size() if game == "snake" else None
        solver = prompt_solver(game) if game == "snake" else "dqn"
        checkpoint = "auto" if solver == "hamiltonian" else prompt_checkpoint()

        if mode == "simulate":
            live_follow = prompt_live_follow(default=False) if game in ("snake", "flappy") else False
            open_browser = prompt_open_browser(default=False) if live_follow else False
            run_simulation(
                game=game,
                checkpoint=checkpoint,
                episodes=episodes,
                grid_size=grid_size,
                solver=solver,
                live_feed=live_follow,
                live_every_n_steps=25 if solver == "hamiltonian" else 1,
                serve_live=live_follow,
                open_browser=open_browser,
            )
        else:  # visualize
            run_simulation(
                game=game,
                checkpoint=checkpoint,
                episodes=episodes,
                grid_size=grid_size,
                solver=solver,
                render=True,
                live_feed=False,
                serve_live=False,
            )


if __name__ == "__main__":
    main()
