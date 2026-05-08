from __future__ import annotations

from dataclasses import dataclass
from random import random
from typing import Any, Sequence

import torch

# DQN agent implementation with epsilon-greedy action selection and decay.
@dataclass
class AgentState:
    epsilon: float

# DQN agent implementation with epsilon-greedy action selection and decay.
class DQNAgent:
    # Initialize agent with epsilon parameters for exploration and decay schedule.
    def __init__(self, epsilon_start: float, epsilon_end: float, epsilon_decay: float) -> None:
        self.state = AgentState(epsilon=epsilon_start)
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    # Select an action using epsilon-greedy strategy based on the current state and policy network.
    def select_action(
        self, 
        state: Any, 
        n_actions: int, 
        policy_net: Any | None = None,
        device: torch.device | str = "cpu"
    ) -> int:
        if n_actions <= 0:
            return -1

        if policy_net is None or random() < self.state.epsilon:
            return int(torch.randint(low=0, high=n_actions, size=(1,)).item())

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())
    # Decay epsilon after each episode to reduce exploration over time.
    def decay_epsilon(self) -> None:
        self.state.epsilon = max(self.epsilon_end, self.state.epsilon * self.epsilon_decay)
