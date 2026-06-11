from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from DQN.src.agents.dqn_agent import DQNAgent
from DQN.src.agents.replay_memory import ReplayMemory, Transition
from DQN.src.envs.game_env import GameEnvironment
from DQN.src.models.checkpoint import load_checkpoint, save_checkpoint
from DQN.src.models.q_network import QNetwork
from DQN.src.training.config import TrainConfig
from DQN.src.utils.live_feed import build_flappy_payload, build_snake_payload, publish_state


@dataclass
class TrainingResult:
    episodes: int
    best_reward: float
    final_epsilon: float
    checkpoint_path: Path
    best_eval_checkpoint_path: Optional[Path] = None


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        checkpoint_dir: Path,
        logs_dir: Path,
        web_feed_path: Optional[Path] = None,
        resume: bool = True,
        enable_live_feed: bool = True,
    ) -> None:
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.web_feed_path = web_feed_path
        self.resume = resume
        self.enable_live_feed = enable_live_feed
        if config.cpu_threads > 0:
            torch.set_num_threads(int(config.cpu_threads))
            try:
                torch.set_num_interop_threads(max(1, min(4, int(config.cpu_threads))))
            except RuntimeError:
                pass
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        else:
            self.device = torch.device(config.device)

        if self.device.type == "cuda":
            # Favor throughput on modern NVIDIA GPUs for matmul-heavy training steps.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        self.env = GameEnvironment(config.game, allow_fallback=False)
        self.agent = DQNAgent(
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
        )
        self.memory = ReplayMemory(
            config.memory_size,
            prioritized=config.prioritized_replay,
            alpha=config.per_alpha,
            beta_start=config.per_beta_start,
            beta_frames=config.per_beta_frames,
            priority_epsilon=config.per_priority_epsilon,
        )
        self.policy_net = QNetwork(self.env.metadata.state_size, config.hidden_size, len(self.env.action_space())).to(self.device)
        self.target_net = QNetwork(self.env.metadata.state_size, config.hidden_size, len(self.env.action_space())).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _write_web_feed(
        self,
        *,
        training: bool,
        episode: int,
        step: int,
        episode_reward: float = 0.0,
        board_filled_count: int = 0,
        wall_collision_count: int = 0,
        self_collision_count: int = 0,
    ) -> None:
        if not self.enable_live_feed:
            return

        logic = getattr(self.env, "_logic_instance", None)
        payload = None
        if self.config.game == "snake":
            payload = build_snake_payload(
                logic,
                training=training,
                game=self.config.game,
                episode=episode,
                total_episodes=self.config.episodes,
                step=step,
                episode_reward=episode_reward,
                epsilon=self.agent.state.epsilon,
                board_filled_count=board_filled_count,
                wall_collision_count=wall_collision_count,
                self_collision_count=self_collision_count,
            )
        elif self.config.game == "flappy":
            payload = build_flappy_payload(
                logic,
                training=training,
                game=self.config.game,
                episode=episode,
                total_episodes=self.config.episodes,
                step=step,
                episode_reward=episode_reward,
                epsilon=self.agent.state.epsilon,
            )

        if payload is None:
            return
        publish_state(payload)

    def _sync_target_network(self) -> None:
        # Support soft (Polyak) updates for stability when enabled in config.
        if getattr(self.config, "use_polyak_target", False):
            tau = float(getattr(self.config, "polyak_tau", 0.005))
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(policy_param.data * tau)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _action_mask_from_state(self, state: Any) -> list[bool] | None:
        if not self.config.mask_unsafe_actions or self.config.game != "snake":
            return None
        try:
            return [float(state[index]) < 0.5 for index in range(3)]
        except (TypeError, ValueError, IndexError):
            return None

    def _masked_next_policy_q(self, next_states: torch.Tensor, next_states_np: np.ndarray) -> torch.Tensor:
        next_policy_q = self.policy_net(next_states)
        if self.config.mask_unsafe_actions and self.config.game == "snake":
            safe_mask_np = next_states_np[:, :3] < 0.5
            all_blocked = ~safe_mask_np.any(axis=1)
            safe_mask_np[all_blocked] = True
            safe_mask = torch.from_numpy(safe_mask_np).to(self.device, non_blocking=True)
            next_policy_q = next_policy_q.masked_fill(~safe_mask, -1.0e9)
        return next_policy_q

    def _run_greedy_evaluation(self) -> Dict[str, float]:
        eval_env = GameEnvironment(self.config.game, allow_fallback=False)
        old_epsilon = float(self.agent.state.epsilon)
        self.agent.state.epsilon = 0.0
        eval_step_cap = max(1, int(self.config.eval_max_steps or self.config.max_steps_per_episode))

        total_reward = 0.0
        total_steps = 0
        total_score = 0
        reason_counts: Dict[str, int] = {}

        for _ in range(max(1, self.config.eval_episodes)):
            state = eval_env.reset()
            episode_reward = 0.0
            steps = 0
            done = False
            end_reason = "max_steps_per_episode"

            while not done and steps < eval_step_cap:
                action = self.agent.select_action(
                    state,
                    len(eval_env.action_space()),
                    policy_net=self.policy_net,
                    device=self.device,
                    action_mask=self._action_mask_from_state(state),
                )
                outcome = eval_env.step(action)
                state = outcome.state
                episode_reward += float(outcome.reward)
                done = bool(outcome.done)
                if done:
                    end_reason = str(outcome.info.get("reason", "terminal"))
                steps += 1

            episode_score = int(getattr(getattr(eval_env, "_logic_instance", None), "score", 0))
            total_reward += episode_reward
            total_steps += steps
            total_score += episode_score
            reason_counts[end_reason] = reason_counts.get(end_reason, 0) + 1

        self.agent.state.epsilon = old_epsilon

        eval_runs = max(1, self.config.eval_episodes)
        return {
            "eval_episodes": float(eval_runs),
            "avg_reward": total_reward / eval_runs,
            "avg_steps": total_steps / eval_runs,
            "avg_score": total_score / eval_runs,
            "board_filled_pct": 100.0 * reason_counts.get("board_filled", 0) / eval_runs,
            "self_collision_pct": 100.0 * reason_counts.get("self_collision", 0) / eval_runs,
            "wall_collision_pct": 100.0 * reason_counts.get("wall_collision", 0) / eval_runs,
            "stagnation_timeout_pct": 100.0 * reason_counts.get("stagnation_timeout", 0) / eval_runs,
            "pipe_collision_pct": 100.0 * reason_counts.get("pipe_collision", 0) / eval_runs,
            "ground_collision_pct": 100.0 * reason_counts.get("ground_collision", 0) / eval_runs,
            "ceiling_collision_pct": 100.0 * reason_counts.get("ceiling_collision", 0) / eval_runs,
            "max_steps_reached_pct": 100.0 * reason_counts.get("max_steps_per_episode", 0) / eval_runs,
        }

    def _append_eval_metrics(self, metrics: Dict[str, float], checkpoint_path: Path) -> None:
        eval_path = self.logs_dir / "eval_metrics.csv"
        fieldnames = [
            "timestamp_utc",
            "game",
            "checkpoint",
            "eval_episodes",
            "avg_reward",
            "avg_steps",
            "avg_score",
            "board_filled_pct",
            "self_collision_pct",
            "wall_collision_pct",
            "stagnation_timeout_pct",
            "pipe_collision_pct",
            "ground_collision_pct",
            "ceiling_collision_pct",
            "max_steps_reached_pct",
        ]
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "game": self.config.game,
            "checkpoint": str(checkpoint_path),
            "eval_episodes": int(metrics["eval_episodes"]),
            "avg_reward": f"{metrics['avg_reward']:.4f}",
            "avg_steps": f"{metrics['avg_steps']:.2f}",
            "avg_score": f"{metrics['avg_score']:.2f}",
            "board_filled_pct": f"{metrics['board_filled_pct']:.2f}",
            "self_collision_pct": f"{metrics['self_collision_pct']:.2f}",
            "wall_collision_pct": f"{metrics['wall_collision_pct']:.2f}",
            "stagnation_timeout_pct": f"{metrics['stagnation_timeout_pct']:.2f}",
            "pipe_collision_pct": f"{metrics['pipe_collision_pct']:.2f}",
            "ground_collision_pct": f"{metrics['ground_collision_pct']:.2f}",
            "ceiling_collision_pct": f"{metrics['ceiling_collision_pct']:.2f}",
            "max_steps_reached_pct": f"{metrics['max_steps_reached_pct']:.2f}",
        }

        write_header = not eval_path.exists()
        with eval_path.open("a", newline="", encoding="utf-8") as eval_file:
            writer = csv.DictWriter(eval_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _eval_metric_name(self) -> str:
        if self.config.game == "flappy":
            return "avg_steps"
        if self.config.game == "snake":
            return "avg_score"
        return "avg_reward"

    def _eval_metric_value(self, metrics: Dict[str, float]) -> float:
        return float(metrics[self._eval_metric_name()])

    def _checkpoint_metadata(self, checkpoint_path: Path) -> Dict[str, Any]:
        if not checkpoint_path.exists():
            return {}
        try:
            payload = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            return {}
        metadata = payload.get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    def _best_eval_score_from_metadata(self, metadata: Dict[str, Any]) -> float:
        metric_name = self._eval_metric_name()
        metadata_key = f"eval_{metric_name}"
        try:
            return float(metadata[metadata_key])
        except (KeyError, TypeError, ValueError):
            return float("-inf")

    def _preferred_resume_checkpoint(self) -> Path:
        best_eval_checkpoint = self.checkpoint_dir / "best_eval.pth"
        if best_eval_checkpoint.exists():
            return best_eval_checkpoint
        return self.checkpoint_dir / "latest.pth"

    def _resume_candidates(self) -> list[Path]:
        best_eval_checkpoint = self.checkpoint_dir / "best_eval.pth"
        latest_checkpoint = self.checkpoint_dir / "latest.pth"
        candidates = []
        if best_eval_checkpoint.exists():
            candidates.append(best_eval_checkpoint)
        if latest_checkpoint.exists() and latest_checkpoint not in candidates:
            candidates.append(latest_checkpoint)
        return candidates

    def _save_best_eval_if_improved(
        self,
        *,
        metrics: Dict[str, float],
        episode: int,
        checkpoint_path: Path,
        best_eval_score: float,
    ) -> float:
        eval_metric = self._eval_metric_value(metrics)
        if not self.config.save_best_eval_checkpoint or eval_metric <= best_eval_score:
            return best_eval_score

        save_checkpoint(
            path=checkpoint_path,
            model=self.policy_net,
            optimizer=self.optimizer,
            metadata={
                "game": self.config.game,
                "episode": episode,
                "epsilon": self.agent.state.epsilon,
                "eval_avg_score": float(metrics["avg_score"]),
                "eval_avg_reward": float(metrics["avg_reward"]),
                "eval_avg_steps": float(metrics["avg_steps"]),
                "eval_metric": self._eval_metric_name(),
                "eval_metric_value": eval_metric,
                "algorithm": "double_dqn" if self.config.double_dqn else "dqn",
            },
        )
        print(
            f"[DQN] New best eval checkpoint: {checkpoint_path} "
            f"(episode={episode}, {self._eval_metric_name()}={eval_metric:.2f})",
            flush=True,
        )
        return eval_metric

    def _restore_best_eval_if_regressed(
        self,
        *,
        metrics: Dict[str, float],
        best_eval_checkpoint: Path,
        last_checkpoint: Path,
        best_eval_score: float,
    ) -> bool:
        if (
            not self.config.restore_best_eval_on_regression
            or not best_eval_checkpoint.exists()
            or best_eval_score == float("-inf")
        ):
            return False

        eval_metric = self._eval_metric_value(metrics)
        regression_threshold = min(
            best_eval_score * float(self.config.eval_regression_ratio),
            best_eval_score - float(self.config.eval_regression_min_gap),
        )
        if eval_metric >= regression_threshold:
            return False

        metadata = load_checkpoint(best_eval_checkpoint, self.policy_net, self.optimizer)
        self.policy_net.to(self.device)
        self._move_optimizer_state_to_device()
        self._sync_target_network()
        if "epsilon" in metadata:
            self.agent.state.epsilon = max(
                float(self.agent.state.epsilon),
                float(metadata["epsilon"]),
                float(self.config.epsilon_end),
            )
        save_checkpoint(
            path=last_checkpoint,
            model=self.policy_net,
            optimizer=self.optimizer,
            metadata={
                **metadata,
                "restored_from_best_eval": True,
                "restored_after_eval_metric": eval_metric,
                "restored_best_eval_metric": best_eval_score,
                "algorithm": "double_dqn" if self.config.double_dqn else "dqn",
            },
        )
        print(
            f"[DQN] Eval regression detected ({self._eval_metric_name()}={eval_metric:.2f}, "
            f"best={best_eval_score:.2f}); restored {best_eval_checkpoint} -> {last_checkpoint}",
            flush=True,
        )
        return True

    def _print_eval_summary(self, metrics: Dict[str, float], *, episode: Optional[int] = None) -> None:
        prefix = f"[DQN] Eval@{episode}" if episode is not None else "[DQN] Eval"
        if self.config.game == "flappy":
            print(
                f"{prefix} (greedy, eps=0) | episodes={int(metrics['eval_episodes'])} | "
                f"avg_score={metrics['avg_score']:.2f} | "
                f"avg_reward={metrics['avg_reward']:.2f} | "
                f"avg_steps={metrics['avg_steps']:.2f} | "
                f"cap_hit={metrics['max_steps_reached_pct']:.2f}% | "
                f"pipe={metrics['pipe_collision_pct']:.2f}% | "
                f"ground={metrics['ground_collision_pct']:.2f}% | "
                f"ceiling={metrics['ceiling_collision_pct']:.2f}%",
                flush=True,
            )
            return

        print(
            f"{prefix} (greedy, eps=0) | episodes={int(metrics['eval_episodes'])} | "
            f"avg_score={metrics['avg_score']:.2f} | "
            f"avg_reward={metrics['avg_reward']:.2f} | "
            f"avg_steps={metrics['avg_steps']:.2f} | "
            f"board_filled={metrics['board_filled_pct']:.2f}% | "
            f"self={metrics['self_collision_pct']:.2f}% | "
            f"wall={metrics['wall_collision_pct']:.2f}% | "
            f"stagnation={metrics['stagnation_timeout_pct']:.2f}%",
            flush=True,
        )

    def train(self) -> TrainingResult:
        best_reward = float("-inf")
        best_eval_score = float("-inf")
        total_reward = 0.0
        board_filled_count = 0
        wall_collision_count = 0
        self_collision_count = 0
        pipes_passed_total = 0
        pipe_collision_count = 0
        ground_collision_count = 0
        ceiling_collision_count = 0
        last_checkpoint = self.checkpoint_dir / "latest.pth"
        best_eval_checkpoint = self.checkpoint_dir / "best_eval.pth"
        resume_checkpoint = self._preferred_resume_checkpoint()
        metrics_path = self.logs_dir / "metrics.csv"
        best_eval_score = self._best_eval_score_from_metadata(
            self._checkpoint_metadata(best_eval_checkpoint)
        )
        resume_episode_offset = 0

        if self.resume and resume_checkpoint.exists():
            for candidate in self._resume_candidates():
                try:
                    metadata = load_checkpoint(candidate, self.policy_net, self.optimizer)
                    self.policy_net.to(self.device)
                    self._move_optimizer_state_to_device()
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    if "epsilon" in metadata:
                        self.agent.state.epsilon = float(metadata["epsilon"])
                    if "best_reward" in metadata:
                        best_reward = float(metadata["best_reward"])
                    try:
                        resume_episode_offset = max(0, int(metadata.get("episode", 0)))
                    except (TypeError, ValueError):
                        resume_episode_offset = 0
                    print(
                        f"[DQN] Resume from checkpoint: {candidate} "
                        f"(last_episode={metadata.get('episode', 'unknown')}, epsilon={self.agent.state.epsilon:.4f})"
                    )
                    break
                except Exception as exc:
                    print(
                        f"[DQN] Checkpoint incompatible ({candidate}, {exc.__class__.__name__}); "
                        "trying next available checkpoint."
                    )
            else:
                print("[DQN] No compatible checkpoint found; starting fresh for this run.")
        elif not self.resume and last_checkpoint.exists():
            print(f"[DQN] Fresh training requested. Ignoring existing checkpoint: {last_checkpoint}")

        append_metrics = self.resume and metrics_path.exists()
        with metrics_path.open("a" if append_metrics else "w", newline="", encoding="utf-8") as metrics_file:
            writer = csv.DictWriter(
                metrics_file,
                fieldnames=[
                    "episode",
                    "steps",
                    "score",
                    "episode_reward",
                    "best_reward",
                    "epsilon",
                    "end_reason",
                    "wall_collisions_total",
                    "self_collisions_total",
                    "board_filled_total",
                    "pipes_passed_episode",
                    "pipes_passed_total",
                    "pipe_collisions_total",
                    "ground_collisions_total",
                    "ceiling_collisions_total",
                ],
            )
            if not append_metrics:
                writer.writeheader()
            
            for episode in range(1, self.config.episodes + 1):
                global_episode = resume_episode_offset + episode
                state = self.env.reset()
                episode_reward = 0.0
                done = False
                steps = 0
                end_reason = "max_steps_per_episode"
                episode_pipes_passed = 0
                self._write_web_feed(
                    training=True,
                    episode=global_episode,
                    step=steps,
                    episode_reward=episode_reward,
                    board_filled_count=board_filled_count,
                    wall_collision_count=wall_collision_count,
                    self_collision_count=self_collision_count,
                )

                while not done and steps < self.config.max_steps_per_episode:
                    action = self.agent.select_action(
                        state,
                        len(self.env.action_space()),
                        policy_net=self.policy_net,
                        device=self.device,
                        action_mask=self._action_mask_from_state(state),
                    )
                    outcome = self.env.step(action)
                    episode_pipes_passed += int(outcome.info.get("passed_pipes", 0))
                    self.memory.push(
                        Transition(state=state, action=action, reward=outcome.reward, next_state=outcome.state, done=outcome.done)
                    )
                    if steps % max(1, self.config.learn_every_n_steps) == 0:
                        self._learn_if_possible()
                    state = outcome.state
                    episode_reward += outcome.reward
                    done = outcome.done
                    if outcome.done:
                        end_reason = str(outcome.info.get("reason", "terminal"))
                    steps += 1
                    if steps % max(1, self.config.web_feed_every_n_steps) == 0 or done:
                        self._write_web_feed(
                            training=True,
                            episode=global_episode,
                            step=steps,
                            episode_reward=episode_reward,
                            board_filled_count=board_filled_count,
                            wall_collision_count=wall_collision_count,
                            self_collision_count=self_collision_count,
                        )

                total_reward += episode_reward
                best_reward = max(best_reward, episode_reward)
                if end_reason == "board_filled":
                    board_filled_count += 1
                elif end_reason == "wall_collision":
                    wall_collision_count += 1
                elif end_reason == "self_collision":
                    self_collision_count += 1
                elif end_reason == "pipe_collision":
                    pipe_collision_count += 1
                elif end_reason == "ground_collision":
                    ground_collision_count += 1
                elif end_reason == "ceiling_collision":
                    ceiling_collision_count += 1
                pipes_passed_total += episode_pipes_passed
                self.agent.decay_epsilon()
                writer.writerow(
                    {
                        "episode": global_episode,
                        "steps": steps,
                        "score": int(getattr(getattr(self.env, "_logic_instance", None), "score", 0)),
                        "episode_reward": f"{episode_reward:.4f}",
                        "best_reward": f"{best_reward:.4f}",
                        "epsilon": f"{self.agent.state.epsilon:.6f}",
                        "end_reason": end_reason,
                        "wall_collisions_total": wall_collision_count,
                        "self_collisions_total": self_collision_count,
                        "board_filled_total": board_filled_count,
                        "pipes_passed_episode": episode_pipes_passed,
                        "pipes_passed_total": pipes_passed_total,
                        "pipe_collisions_total": pipe_collision_count,
                        "ground_collisions_total": ground_collision_count,
                        "ceiling_collisions_total": ceiling_collision_count,
                    }
                )
                if episode % max(1, self.config.target_update_every_episodes) == 0:
                    self._sync_target_network()
                if self.config.game == "flappy":
                    print(
                        f"[{episode:3d}/{self.config.episodes} | total {global_episode}] "
                        f"Steps: {steps:6d} | "
                        f"Reason: {end_reason:>18} | "
                        f"Reward: {episode_reward:7.2f} | "
                        f"Epsilon: {self.agent.state.epsilon:.4f} | "
                        f"PipesEp: {episode_pipes_passed} | "
                        f"PipesTot: {pipes_passed_total} | "
                        f"PipeHit: {pipe_collision_count} | "
                        f"Ground: {ground_collision_count} | "
                        f"Ceiling: {ceiling_collision_count}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{episode:3d}/{self.config.episodes} | total {global_episode}] "
                        f"Steps: {steps:6d} | "
                        f"Reason: {end_reason:>18} | "
                        f"Reward: {episode_reward:7.2f} | "
                        f"Epsilon: {self.agent.state.epsilon:.4f} | "
                        f"Walls: {wall_collision_count} | "
                        f"Self: {self_collision_count} | "
                        f"Filled: {board_filled_count}",
                        flush=True,
                    )
                if episode % max(1, self.config.checkpoint_every_episodes) == 0 or episode == self.config.episodes:
                    save_checkpoint(
                        path=last_checkpoint,
                        model=self.policy_net,
                        optimizer=self.optimizer,
                        metadata={
                            "game": self.config.game,
                            "episode": global_episode,
                            "episode_reward": episode_reward,
                            "best_reward": best_reward,
                            "epsilon": self.agent.state.epsilon,
                            "algorithm": "double_dqn" if self.config.double_dqn else "dqn",
                        },
                    )

                if (
                    self.config.eval_enabled
                    and self.config.eval_episodes > 0
                    and self.config.eval_every_episodes > 0
                    and episode % self.config.eval_every_episodes == 0
                ):
                    periodic_eval = self._run_greedy_evaluation()
                    self._append_eval_metrics(periodic_eval, last_checkpoint)
                    previous_best_eval_score = best_eval_score
                    best_eval_score = self._save_best_eval_if_improved(
                        metrics=periodic_eval,
                        episode=global_episode,
                        checkpoint_path=best_eval_checkpoint,
                        best_eval_score=best_eval_score,
                    )
                    if best_eval_score == previous_best_eval_score:
                        self._restore_best_eval_if_regressed(
                            metrics=periodic_eval,
                            best_eval_checkpoint=best_eval_checkpoint,
                            last_checkpoint=last_checkpoint,
                            best_eval_score=previous_best_eval_score,
                        )
                    self._print_eval_summary(periodic_eval, episode=global_episode)

        self._write_web_feed(
            training=False,
            episode=resume_episode_offset + self.config.episodes,
            step=0,
            episode_reward=0.0,
            board_filled_count=board_filled_count,
            wall_collision_count=wall_collision_count,
            self_collision_count=self_collision_count,
        )

        if self.config.game == "flappy":
            print(
                f"[DQN] Summary | pipes_passed={pipes_passed_total} | "
                f"pipe_collision={pipe_collision_count} | "
                f"ground_collision={ground_collision_count} | "
                f"ceiling_collision={ceiling_collision_count}",
                flush=True,
            )
        else:
            print(
                f"[DQN] Summary | wall_collision={wall_collision_count} | "
                f"self_collision={self_collision_count} | board_filled={board_filled_count}",
                flush=True,
            )

        if self.config.eval_enabled and self.config.eval_episodes > 0:
            eval_metrics = self._run_greedy_evaluation()
            self._append_eval_metrics(eval_metrics, last_checkpoint)
            previous_best_eval_score = best_eval_score
            best_eval_score = self._save_best_eval_if_improved(
                metrics=eval_metrics,
                episode=resume_episode_offset + self.config.episodes,
                checkpoint_path=best_eval_checkpoint,
                best_eval_score=best_eval_score,
            )
            if best_eval_score == previous_best_eval_score:
                self._restore_best_eval_if_regressed(
                    metrics=eval_metrics,
                    best_eval_checkpoint=best_eval_checkpoint,
                    last_checkpoint=last_checkpoint,
                    best_eval_score=previous_best_eval_score,
                )
            self._print_eval_summary(eval_metrics)
            print(f"[DQN] Eval metrics logged: {self.logs_dir / 'eval_metrics.csv'}", flush=True)

        return TrainingResult(
            episodes=self.config.episodes,
            best_reward=best_reward,
            final_epsilon=self.agent.state.epsilon,
            checkpoint_path=last_checkpoint,
            best_eval_checkpoint_path=best_eval_checkpoint if best_eval_checkpoint.exists() else None,
        )

    def _learn_if_possible(self) -> None:
        if len(self.memory) < max(self.config.batch_size, self.config.learning_starts):
            return
        batch, indices, sample_weights = self.memory.sample(self.config.batch_size)
        sampled_count = len(batch)

        states_np = np.asarray([transition.state for transition in batch], dtype=np.float32)
        next_states_np = np.asarray([transition.next_state for transition in batch], dtype=np.float32)
        actions_np = np.fromiter((transition.action for transition in batch), dtype=np.int64, count=sampled_count)
        rewards_np = np.fromiter((transition.reward for transition in batch), dtype=np.float32, count=sampled_count)
        dones_np = np.fromiter((float(transition.done) for transition in batch), dtype=np.float32, count=sampled_count)
        weights_np = np.asarray(sample_weights, dtype=np.float32)

        pin_memory = self.device.type == "cuda"
        states = torch.from_numpy(states_np).pin_memory() if pin_memory else torch.from_numpy(states_np)
        actions = torch.from_numpy(actions_np).pin_memory() if pin_memory else torch.from_numpy(actions_np)
        rewards = torch.from_numpy(rewards_np).pin_memory() if pin_memory else torch.from_numpy(rewards_np)
        next_states = torch.from_numpy(next_states_np).pin_memory() if pin_memory else torch.from_numpy(next_states_np)
        dones = torch.from_numpy(dones_np).pin_memory() if pin_memory else torch.from_numpy(dones_np)
        weights = torch.from_numpy(weights_np).pin_memory() if pin_memory else torch.from_numpy(weights_np)

        states = states.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True).unsqueeze(1)
        rewards = rewards.to(self.device, non_blocking=True)
        next_states = next_states.to(self.device, non_blocking=True)
        dones = dones.to(self.device, non_blocking=True)
        weights = weights.to(self.device, non_blocking=True)

        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: choose the next action with the online policy,
                # then evaluate that chosen action with the slower target net.
                next_actions = self._masked_next_policy_q(next_states, next_states_np).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.config.gamma * next_q_values * (1.0 - dones)

        td_errors = targets - current_q_values
        per_sample_loss = F.smooth_l1_loss(current_q_values, targets, reduction="none")
        loss = (weights * per_sample_loss).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors.detach().abs().cpu().tolist())
