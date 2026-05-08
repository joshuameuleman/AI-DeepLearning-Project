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
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.target_net.load_state_dict(self.policy_net.state_dict())

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
        metrics_path = self.logs_dir / "metrics.csv"

        if self.resume and last_checkpoint.exists():
            try:
                metadata = load_checkpoint(last_checkpoint, self.policy_net, self.optimizer)
                self.policy_net.to(self.device)
                self._move_optimizer_state_to_device()
                self.target_net.load_state_dict(self.policy_net.state_dict())
                if "epsilon" in metadata:
                    self.agent.state.epsilon = float(metadata["epsilon"])
                if "best_reward" in metadata:
                    best_reward = float(metadata["best_reward"])
                print(
                    f"[DQN] Resume from checkpoint: {last_checkpoint} "
                    f"(last_episode={metadata.get('episode', 'unknown')}, epsilon={self.agent.state.epsilon:.4f})"
                )
            except Exception as exc:
                print(
                    f"[DQN] Checkpoint incompatible ({exc.__class__.__name__}); "
                    f"starting fresh for this run."
                )
        elif not self.resume and last_checkpoint.exists():
            print(f"[DQN] Fresh training requested. Ignoring existing checkpoint: {last_checkpoint}")

        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
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
            writer.writeheader()
            
            for episode in range(1, self.config.episodes + 1):
                state = self.env.reset()
                episode_reward = 0.0
                done = False
                steps = 0
                end_reason = "max_steps_per_episode"
                episode_pipes_passed = 0
                self._write_web_feed(
                    training=True,
                    episode=episode,
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
                            episode=episode,
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
                        "episode": episode,
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
                        f"[{episode:3d}/{self.config.episodes}] "
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
                        f"[{episode:3d}/{self.config.episodes}] "
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
                            "episode": episode,
                            "episode_reward": episode_reward,
                            "best_reward": best_reward,
                            "epsilon": self.agent.state.epsilon,
                        },
                    )

                if (
                    self.config.game == "flappy"
                    and self.config.eval_enabled
                    and self.config.eval_every_episodes > 0
                    and episode % self.config.eval_every_episodes == 0
                ):
                    periodic_eval = self._run_greedy_evaluation()
                    self._append_eval_metrics(periodic_eval, last_checkpoint)
                    eval_metric = float(periodic_eval["avg_steps"])
                    if self.config.save_best_eval_checkpoint and eval_metric > best_eval_score:
                        best_eval_score = eval_metric
                        save_checkpoint(
                            path=best_eval_checkpoint,
                            model=self.policy_net,
                            optimizer=self.optimizer,
                            metadata={
                                "game": self.config.game,
                                "episode": episode,
                                "epsilon": self.agent.state.epsilon,
                                "eval_avg_score": float(periodic_eval["avg_score"]),
                                "eval_avg_reward": float(periodic_eval["avg_reward"]),
                                "eval_avg_steps": float(periodic_eval["avg_steps"]),
                            },
                        )
                        print(
                            f"[DQN] New best eval checkpoint: {best_eval_checkpoint} "
                            f"(episode={episode}, avg_steps={eval_metric:.2f})",
                            flush=True,
                        )

                    print(
                        f"[DQN] Eval@{episode} (greedy, eps=0) | "
                        f"avg_score={periodic_eval['avg_score']:.2f} | "
                        f"avg_reward={periodic_eval['avg_reward']:.2f} | "
                        f"cap_hit={periodic_eval['max_steps_reached_pct']:.2f}% | "
                        f"pipe={periodic_eval['pipe_collision_pct']:.2f}% | "
                        f"ground={periodic_eval['ground_collision_pct']:.2f}% | "
                        f"ceiling={periodic_eval['ceiling_collision_pct']:.2f}%",
                        flush=True,
                    )

        self._write_web_feed(
            training=False,
            episode=self.config.episodes,
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

            if self.config.game == "flappy":
                eval_metric = float(eval_metrics["avg_steps"])
                if self.config.save_best_eval_checkpoint and eval_metric > best_eval_score:
                    best_eval_score = eval_metric
                    save_checkpoint(
                        path=best_eval_checkpoint,
                        model=self.policy_net,
                        optimizer=self.optimizer,
                        metadata={
                            "game": self.config.game,
                            "episode": self.config.episodes,
                            "epsilon": self.agent.state.epsilon,
                            "eval_avg_score": float(eval_metrics["avg_score"]),
                            "eval_avg_reward": float(eval_metrics["avg_reward"]),
                            "eval_avg_steps": float(eval_metrics["avg_steps"]),
                        },
                    )
                    print(
                        f"[DQN] New best eval checkpoint: {best_eval_checkpoint} "
                        f"(episode={self.config.episodes}, avg_steps={eval_metric:.2f})",
                        flush=True,
                    )

                print(
                    f"[DQN] Eval (greedy, eps=0) | episodes={int(eval_metrics['eval_episodes'])} | "
                    f"avg_score={eval_metrics['avg_score']:.2f} | "
                    f"avg_reward={eval_metrics['avg_reward']:.2f} | "
                    f"avg_steps={eval_metrics['avg_steps']:.2f} | "
                    f"cap_hit={eval_metrics['max_steps_reached_pct']:.2f}% | "
                    f"pipe={eval_metrics['pipe_collision_pct']:.2f}% | "
                    f"ground={eval_metrics['ground_collision_pct']:.2f}% | "
                    f"ceiling={eval_metrics['ceiling_collision_pct']:.2f}%",
                    flush=True,
                )
            else:
                print(
                    f"[DQN] Eval (greedy, eps=0) | episodes={int(eval_metrics['eval_episodes'])} | "
                    f"avg_score={eval_metrics['avg_score']:.2f} | "
                    f"avg_reward={eval_metrics['avg_reward']:.2f} | "
                    f"avg_steps={eval_metrics['avg_steps']:.2f} | "
                    f"board_filled={eval_metrics['board_filled_pct']:.2f}% | "
                    f"self={eval_metrics['self_collision_pct']:.2f}% | "
                    f"wall={eval_metrics['wall_collision_pct']:.2f}%",
                    flush=True,
                )
            print(f"[DQN] Eval metrics logged: {self.logs_dir / 'eval_metrics.csv'}", flush=True)

        return TrainingResult(
            episodes=self.config.episodes,
            best_reward=best_reward,
            final_epsilon=self.agent.state.epsilon,
            checkpoint_path=last_checkpoint,
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

        states = torch.from_numpy(states_np).to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions_np).to(self.device, non_blocking=True).unsqueeze(1)
        rewards = torch.from_numpy(rewards_np).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states_np).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones_np).to(self.device, non_blocking=True)
        weights = torch.from_numpy(weights_np).to(self.device, non_blocking=True)

        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            # Double DQN: choose actions with policy net, evaluate with target net.
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.config.gamma * next_q_values * (1.0 - dones)

        td_errors = targets - current_q_values
        per_sample_loss = F.smooth_l1_loss(current_q_values, targets, reduction="none")
        loss = (weights * per_sample_loss).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors.detach().abs().cpu().tolist())
