from __future__ import annotations

import argparse
import importlib.util
import sys
import webbrowser
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from DQN.src.agents.dqn_agent import DQNAgent
from DQN.src.envs.game_catalog import GAME_LOGIC_FILE_BY_GAME, SUPPORTED_GAMES
from DQN.src.envs.game_env import GameEnvironment
from DQN.src.models.checkpoint import load_checkpoint
from DQN.src.models.q_network import QNetwork
from DQN.src.utils.live_feed import build_flappy_payload, build_snake_payload, publish_state
from DQN.src.utils.paths import workspace_root
from DQN.src.utils.snake_config import apply_snake_grid_size, resolve_snake_grid_size

AGENT_BY_GAME = {
    "snake": "SnakeDQNAgent",
    "flappy": "FlappyBirdDQNAgent",
    "2048": "Game2048DQNAgent",
}


def _infer_qnetwork_dims_from_checkpoint(checkpoint_path: Path) -> tuple[int, int, int] | None:
    """Return (input_size, hidden_size, output_size) inferred from QNetwork weights."""
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        return None

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

    # Basic sanity check for the fixed 3-layer MLP architecture.
    if int(middle_layer.shape[0]) != hidden_size or int(middle_layer.shape[1]) != hidden_size:
        return None

    return (input_size, hidden_size, output_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulation with a trained DQN model.")
    parser.add_argument("--game", default="snake", choices=list(SUPPORTED_GAMES))
    parser.add_argument("--checkpoint", default="latest.pth")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--loop", action="store_true", help="Play continuously until interrupted")
    parser.add_argument("--grid-size", type=int, default=None, help="Snake grid size (e.g. 32, 64, 128)")
    parser.add_argument("--max-steps", type=int, default=0, help="Per-episode step cap (0 means no cap)")
    parser.add_argument("--render", action="store_true", help="Render game in a live window while simulating")
    parser.add_argument("--fps", type=int, default=12, help="Render frame rate for --render mode")
    parser.add_argument("--live-feed", action="store_true", help="Publish simulation state to the web live feed")
    parser.add_argument("--live-every", type=int, default=1, help="Publish live feed every N steps")
    parser.add_argument("--serve-live", action="store_true", help="Start live web server on port 8000")
    parser.add_argument("--open-browser", action="store_true", help="Open browser to /web/ when live server starts")
    return parser.parse_args()


def _resolve_run_name(game: str, grid_size: int | None) -> str:
    if game != "snake":
        return game
    grid_size = resolve_snake_grid_size(grid_size)
    return f"snake_{grid_size}x{grid_size}"


def _resolve_checkpoint_path(game: str, checkpoint: str, grid_size: int | None) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_absolute():
        return checkpoint_path

    if checkpoint_path.parent != Path("."):
        return (workspace_root() / checkpoint_path).resolve()

    dqn_root = workspace_root() / "DQN"
    run_name = _resolve_run_name(game, grid_size)
    return (dqn_root / "checkpoints" / run_name / checkpoint_path.name).resolve()


def _publish_snake_live_feed(
    env: GameEnvironment,
    *,
    episode: int,
    total_episodes: int,
    step: int,
    episode_reward: float,
    done: bool,
) -> None:
    logic = getattr(env, "_logic_instance", None)
    payload = build_snake_payload(
        logic,
        training=False,
        simulating=True,
        done=done,
        game="snake",
        episode=episode,
        total_episodes=total_episodes,
        step=step,
        episode_reward=episode_reward,
        epsilon=0.0,
    )
    if payload is None:
        return
    publish_state(payload)


def _publish_flappy_live_feed(
    env: GameEnvironment,
    *,
    episode: int,
    total_episodes: int,
    step: int,
    episode_reward: float,
    done: bool,
) -> None:
    logic = getattr(env, "_logic_instance", None)
    payload = build_flappy_payload(
        logic,
        training=False,
        simulating=True,
        done=done,
        game="flappy",
        episode=episode,
        total_episodes=total_episodes,
        step=step,
        episode_reward=episode_reward,
        epsilon=0.0,
    )
    if payload is None:
        return
    publish_state(payload)


def _publish_live_feed(
    env: GameEnvironment,
    *,
    game: str,
    episode: int,
    total_episodes: int,
    step: int,
    episode_reward: float,
    done: bool,
) -> None:
    if game == "snake":
        _publish_snake_live_feed(
            env,
            episode=episode,
            total_episodes=total_episodes,
            step=step,
            episode_reward=episode_reward,
            done=done,
        )
    elif game == "flappy":
        _publish_flappy_live_feed(
            env,
            episode=episode,
            total_episodes=total_episodes,
            step=step,
            episode_reward=episode_reward,
            done=done,
        )


def run_simulation(
    game: str,
    checkpoint: str = "latest.pth",
    episodes: int = 5,
    *,
    loop: bool = False,
    grid_size: int | None = None,
    max_steps: int = 0,
    render: bool = False,
    fps: int = 12,
    live_feed: bool = False,
    live_every_n_steps: int = 1,
    serve_live: bool = False,
    open_browser: bool = False,
) -> None:
    selected_game = game.lower().strip()
    if selected_game not in SUPPORTED_GAMES:
        raise ValueError(f"Unsupported game '{game}'. Supported values: {', '.join(SUPPORTED_GAMES)}")

    resolved_grid_size: int | None = None
    if selected_game == "snake" and grid_size is not None and grid_size > 3:
        resolved_grid_size = apply_snake_grid_size(grid_size)
    elif selected_game == "snake":
        resolved_grid_size = apply_snake_grid_size(None)

    if selected_game not in ("snake", "flappy"):
        live_feed = False
        serve_live = False

    live_server = None
    if serve_live and live_feed:
        try:
            from serve_web import start_server_in_thread

            live_server, _ = start_server_in_thread(host="0.0.0.0", port=8000)
            print("[DQN] Live server started at http://127.0.0.1:8000/web/")
            if open_browser:
                webbrowser.open("http://127.0.0.1:8000/web/")
        except OSError as exc:
            print(f"[DQN] Live server could not start on port 8000: {exc}")

    selected_agent = AGENT_BY_GAME[selected_game]
    game_logic_file = GAME_LOGIC_FILE_BY_GAME[selected_game]
    checkpoint_path = _resolve_checkpoint_path(selected_game, checkpoint, resolved_grid_size)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Use --checkpoint with an absolute path or a checkpoint filename in DQN/checkpoints/<run_name>/"
        )

    env = GameEnvironment(selected_game)
    n_actions = len(env.action_space())
    state_size = int(env.metadata.state_size)

    renderer = None
    should_stop = False
    if render:
        if selected_game == "snake":
            try:
                from Games.Snake.renderer import SnakeRenderer

                logic_instance = getattr(env, "_logic_instance", None)
                if logic_instance is None:
                    raise RuntimeError("Snake logic is unavailable; cannot render simulation")
                renderer = SnakeRenderer(logic_instance)
            except Exception as exc:
                print(f"[DQN] Could not initialize renderer ({exc}). Continuing without window rendering.")
                renderer = None
        elif selected_game == "flappy":
            try:
                renderer_path = workspace_root() / "Games" / "Flappy Bird" / "renderer.py"
                spec = importlib.util.spec_from_file_location("flappy_renderer", renderer_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError("Could not load Flappy renderer module")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                renderer_class = getattr(module, "FlappyBirdRenderer", None)
                if renderer_class is None:
                    raise RuntimeError("FlappyBirdRenderer class not found")

                logic_instance = getattr(env, "_logic_instance", None)
                if logic_instance is None:
                    raise RuntimeError("Flappy logic is unavailable; cannot render simulation")
                renderer = renderer_class(logic_instance)
            except Exception as exc:
                print(f"[DQN] Could not initialize renderer ({exc}). Continuing without window rendering.")
                renderer = None
        else:
            print("[DQN] Render mode is currently supported for snake/flappy only. Continuing without window rendering.")

    inferred_dims = _infer_qnetwork_dims_from_checkpoint(checkpoint_path)
    hidden_size = 256
    if inferred_dims is not None:
        ckpt_state_size, ckpt_hidden_size, ckpt_actions = inferred_dims
        if ckpt_state_size != state_size:
            raise RuntimeError(
                f"Checkpoint state size mismatch: checkpoint expects {ckpt_state_size}, "
                f"but environment provides {state_size}."
            )
        if ckpt_actions != n_actions:
            raise RuntimeError(
                f"Checkpoint action size mismatch: checkpoint expects {ckpt_actions}, "
                f"but environment provides {n_actions}."
            )
        hidden_size = ckpt_hidden_size

    policy_net = QNetwork(state_size, hidden_size=hidden_size, output_size=n_actions)
    metadata = load_checkpoint(checkpoint_path, policy_net, optimizer=None)
    policy_net.eval()

    inference_agent = DQNAgent(epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0)

    requested_episodes = max(1, int(episodes))
    total_episodes = float("inf") if loop else requested_episodes
    episode_idx = 0

    print(f"[DQN] Simulation start for game={selected_game}")
    print(f"[DQN] Agent selected: {selected_agent}")
    print(f"[DQN] Game logic: {game_logic_file}")
    print(f"[DQN] Checkpoint: {checkpoint_path}")
    if metadata:
        print(f"[DQN] Checkpoint metadata: {metadata}")
    print(f"[DQN] Episodes: {'infinite (--loop)' if loop else requested_episodes}")
    print(f"[DQN] Live feed: {'ON' if live_feed else 'OFF'}")

    try:
        while episode_idx < total_episodes:
            episode_idx += 1
            state = env.reset()
            if live_feed and selected_game in ("snake", "flappy"):
                _publish_live_feed(
                    env,
                    game=selected_game,
                    episode=episode_idx,
                    total_episodes=requested_episodes,
                    step=0,
                    episode_reward=0.0,
                    done=False,
                )
            if renderer is not None and not renderer.render(fps=max(1, int(fps))):
                should_stop = True
                break
            done = False
            steps = 0
            total_reward = 0.0
            last_info = {}

            while not done and (max_steps <= 0 or steps < max_steps):
                action = inference_agent.select_action(
                    state,
                    n_actions=n_actions,
                    policy_net=policy_net,
                )
                outcome = env.step(action)
                state = outcome.state
                done = outcome.done
                total_reward += float(outcome.reward)
                last_info = dict(outcome.info)
                steps += 1

                if live_feed and selected_game in ("snake", "flappy") and (
                    steps % max(1, int(live_every_n_steps)) == 0 or done
                ):
                    _publish_live_feed(
                        env,
                        game=selected_game,
                        episode=episode_idx,
                        total_episodes=requested_episodes,
                        step=steps,
                        episode_reward=total_reward,
                        done=done,
                    )

                if renderer is not None and not renderer.render(fps=max(1, int(fps))):
                    should_stop = True
                    break

            if should_stop:
                print("[DQN] Render window closed. Stopping simulation.")
                break

            end_reason = str(last_info.get("reason", "max_steps" if max_steps > 0 and not done else "terminal"))
            score = getattr(getattr(env, "_logic_instance", None), "score", None)
            if score is None:
                print(
                    f"[SIM {episode_idx}] steps={steps} reward={total_reward:.3f} reason={end_reason}",
                    flush=True,
                )
            else:
                print(
                    f"[SIM {episode_idx}] steps={steps} score={int(score)} reward={total_reward:.3f} reason={end_reason}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[DQN] Simulation interrupted by user.")
    finally:
        if renderer is not None:
            renderer.close()
        if live_server is not None:
            live_server.shutdown()
            live_server.server_close()


def main() -> None:
    args = parse_args()
    run_simulation(
        game=args.game,
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        loop=args.loop,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        render=args.render,
        fps=args.fps,
        live_feed=args.live_feed,
        live_every_n_steps=args.live_every,
        serve_live=args.serve_live,
        open_browser=args.open_browser,
    )


if __name__ == "__main__":
    main()
