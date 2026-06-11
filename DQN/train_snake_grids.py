from __future__ import annotations

"""Train Snake-modellen voor meerdere gridgroottes na elkaar.

Dit script is handig als je bijvoorbeeld aparte checkpoints wilt voor 32x32,
64x64 en 128x128 zonder drie losse commando's te starten.
"""

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    # Zorgt dat `python DQN/train_snake_grids.py` imports kan vinden vanaf de projectroot.
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from DQN.train import run_training


DEFAULT_GRIDS = (32, 64, 128)


def parse_args() -> argparse.Namespace:
    """Lees CLI-opties voor een batch Snake-training."""
    parser = argparse.ArgumentParser(description="Train Snake DQN models for multiple grid sizes.")

    # --grids bepaalt welke bordgroottes achter elkaar getraind worden.
    parser.add_argument("--grids", type=int, nargs="+", default=list(DEFAULT_GRIDS), help="Grid sizes to train")

    # --episodes geldt per grid, dus 10_000 met drie grids is 30_000 episodes totaal.
    parser.add_argument("--episodes", type=int, default=10_000, help="Episodes per grid")

    # Profiel en device worden doorgegeven aan DQN/train.py.
    parser.add_argument("--profile", choices=("fast", "balanced", "quality"), default="balanced")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    """Train elke opgegeven gridgrootte met dezelfde instellingen."""
    args = parse_args()
    for grid_size in args.grids:
        print(f"[DQN] === Training Snake {grid_size}x{grid_size} ===", flush=True)

        # resume=True betekent: verder trainen vanaf best_eval/latest als die bestaat.
        run_training(
            game="snake",
            episodes=max(1, int(args.episodes)),
            resume=True,
            grid_size=int(grid_size),
            enable_live_feed=False,
            profile=args.profile,
            device=args.device,
        )


if __name__ == "__main__":
    main()
