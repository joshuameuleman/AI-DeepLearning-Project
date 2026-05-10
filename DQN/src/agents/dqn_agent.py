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
        device: torch.device | str = "cpu",
        action_mask: Sequence[bool] | None = None,
    ) -> int:
        if n_actions <= 0:
            return -1

        valid_actions = self._valid_actions(n_actions, action_mask)
        if policy_net is None or random() < self.state.epsilon:
            random_index = int(torch.randint(low=0, high=len(valid_actions), size=(1,)).item())
            return int(valid_actions[random_index])

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            if action_mask is not None:
                mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
                q_values = q_values.masked_fill(~mask_tensor, -1.0e9)
        return int(torch.argmax(q_values, dim=1).item())

    def _valid_actions(self, n_actions: int, action_mask: Sequence[bool] | None) -> list[int]:
        if action_mask is None:
            return list(range(n_actions))

        valid = [
            action
            for action in range(n_actions)
            if action < len(action_mask) and bool(action_mask[action])
        ]
        return valid or list(range(n_actions))
    # Decay epsilon after each episode to reduce exploration over time.
    def decay_epsilon(self) -> None:
        self.state.epsilon = max(self.epsilon_end, self.state.epsilon * self.epsilon_decay)
