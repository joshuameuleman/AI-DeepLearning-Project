from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Welke game wordt getraind: "snake", "flappy" of "2048".
    game: str = "snake"

    # Aantal volledige pogingen/runs. Een episode loopt van reset tot terminal/game over.
    episodes: int = 1000

    # Learning rate = grootte van de optimizer-stap.
    # Te hoog kan instabiel worden; te laag leert langzaam.
    learning_rate: float = 2e-4

    # Gamma = discount factor voor toekomstige rewards.
    # Hoe dichter bij 1.0, hoe belangrijker lange-termijnbeloning wordt.
    gamma: float = 0.99

    # Aantal transitions dat tegelijk uit replay memory wordt geleerd.
    batch_size: int = 128

    # Maximum aantal ervaringen in replay memory.
    memory_size: int = 300_000

    # Aantal neuronen in de verborgen lagen van het Q-network.
    hidden_size: int = 256

    # Epsilon bepaalt de kans op random exploration.
    # Start hoog, eindigt laag, en daalt met epsilon_decay.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9997

    # Veiligheidslimiet: stop een episode na te veel stappen.
    max_steps_per_episode: int = 2000

    # Train niet elke game-step, maar om de N steps.
    learn_every_n_steps: int = 4

    # Hoe vaak het target network hard wordt gesynchroniseerd.
    # Bij Polyak updates gebeurt dit zachter/geleidelijker.
    target_update_every_episodes: int = 2

    # Hoe vaak state naar de live web-feed wordt gepubliceerd.
    web_feed_every_n_steps: int = 5

    # Eerst ervaringen verzamelen voordat het netwerk gaat leren.
    learning_starts: int = 2_000

    # Gradient clipping: voorkomt extreem grote neural-network updates.
    max_grad_norm: float = 1.0

    # Hoe vaak latest.pth wordt opgeslagen.
    checkpoint_every_episodes: int = 5

    # "auto" gebruikt CUDA als PyTorch een GPU ziet, anders CPU.
    device: str = "auto"

    # Prioritized replay sampled belangrijke/leerzame transitions vaker.
    prioritized_replay: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 250_000
    per_priority_epsilon: float = 1e-5

    # Evaluatie meet prestatie zonder exploration/training.
    eval_enabled: bool = True
    eval_episodes: int = 25
    eval_max_steps: int = 0
    eval_every_episodes: int = 20
    save_best_eval_checkpoint: bool = True

    # Herstel naar best_eval.pth als evaluatie te sterk achteruitgaat.
    restore_best_eval_on_regression: bool = True

    # Soft/Polyak target update: target_net beweegt langzaam richting policy_net.
    use_polyak_target: bool = True
    polyak_tau: float = 0.005

    # Drempels om te bepalen wanneer evaluatie "regressie" is.
    eval_regression_ratio: float = 0.75
    eval_regression_min_gap: float = 50.0

    # Voor Snake: maskeer acties die direct gevaarlijk/ongeldig zijn.
    mask_unsafe_actions: bool = True

    # Double DQN vermindert overschatting van Q-values.
    double_dqn: bool = True

    # 0 betekent PyTorch kiest zelf; hoger beperkt CPU-threadgebruik.
    cpu_threads: int = 0
