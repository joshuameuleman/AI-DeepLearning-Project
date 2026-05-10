from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    game: str = "snake"
    episodes: int = 1000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 128
    memory_size: int = 300_000
    hidden_size: int = 256
    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    epsilon_decay: float = 0.9995
    max_steps_per_episode: int = 500
    learn_every_n_steps: int = 4
    target_update_every_episodes: int = 10
    web_feed_every_n_steps: int = 5
    learning_starts: int = 2_000
    max_grad_norm: float = 10.0
    checkpoint_every_episodes: int = 5
    device: str = "auto"
    prioritized_replay: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 250_000
    per_priority_epsilon: float = 1e-5
    eval_enabled: bool = True
    eval_episodes: int = 25
    eval_max_steps: int = 0
    eval_every_episodes: int = 0
    save_best_eval_checkpoint: bool = False
    mask_unsafe_actions: bool = False
    cpu_threads: int = 0
