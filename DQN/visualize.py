from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from DQN.simulate import run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained DQN agent.")
    parser.add_argument("--game", default="snake", choices=["snake", "flappy", "2048"])
    parser.add_argument("--checkpoint", default="latest.pth")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=None, help="Snake grid size (e.g. 32, 64, 128)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_simulation(
        game=args.game,
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        grid_size=args.grid_size,
        render=True,
        live_feed=False,
        serve_live=False,
    )


if __name__ == "__main__":
    main()
