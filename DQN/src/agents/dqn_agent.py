from __future__ import annotations

from dataclasses import dataclass
from random import random
from typing import Any, Sequence

import torch


@dataclass
class AgentState:
    # epsilon = kans dat de agent een willekeurige actie kiest.
    # Een hoge epsilon betekent veel exploratie; een lage epsilon betekent meer vertrouwen op het model.
    epsilon: float


class DQNAgent:
    """Kiest acties voor DQN met epsilon-greedy exploration.

    De agent bevat zelf geen neural network. Hij krijgt optioneel een `policy_net`
    mee in `select_action()` en beslist dan of hij exploreert of het netwerk volgt.
    """

    def __init__(self, epsilon_start: float, epsilon_end: float, epsilon_decay: float) -> None:
        """Bewaar de epsilon-instellingen voor exploration.

        epsilon_start:
            Beginwaarde. Vaak hoog, zodat de agent in het begin veel probeert.
        epsilon_end:
            Minimumwaarde. Zo blijft er tijdens training altijd wat randomness over.
        epsilon_decay:
            Vermenigvuldigingsfactor na elke episode. Lager dan 1.0 laat epsilon dalen.
        """
        self.state = AgentState(epsilon=epsilon_start)
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(
        self, 
        state: Any, 
        n_actions: int, 
        policy_net: Any | None = None,
        device: torch.device | str = "cpu",
        action_mask: Sequence[bool] | None = None,
    ) -> int:
        """Kies een actie voor de huidige state.

        state:
            De numerieke speltoestand die het Q-network als input gebruikt.
        n_actions:
            Aantal mogelijke acties, bijvoorbeeld 3 voor Snake.
        policy_net:
            Neural network dat per actie een Q-value voorspelt.
        device:
            CPU/GPU waarop de tensor en het model draaien.
        action_mask:
            Optionele lijst booleans waarmee onveilige/ongeldige acties worden uitgesloten.
        """
        if n_actions <= 0:
            return -1

        valid_actions = self._valid_actions(n_actions, action_mask)

        # Exploration: bij kans epsilon kiezen we random uit de geldige acties.
        # Dit voorkomt dat de agent te vroeg vastloopt in een slechte strategie.
        if policy_net is None or random() < self.state.epsilon:
            random_index = int(torch.randint(low=0, high=len(valid_actions), size=(1,)).item())
            return int(valid_actions[random_index])

        # Exploitation: laat het Q-network voorspellen welke actie de hoogste waarde heeft.
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            if action_mask is not None:
                # Ongeldige acties krijgen een extreem lage Q-value, zodat argmax ze niet kiest.
                mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
                q_values = q_values.masked_fill(~mask_tensor, -1.0e9)
        return int(torch.argmax(q_values, dim=1).item())

    def _valid_actions(self, n_actions: int, action_mask: Sequence[bool] | None) -> list[int]:
        """Maak een lijst met acties die gekozen mogen worden."""
        if action_mask is None:
            return list(range(n_actions))

        valid = [
            action
            for action in range(n_actions)
            if action < len(action_mask) and bool(action_mask[action])
        ]
        return valid or list(range(n_actions))

    def decay_epsilon(self) -> None:
        """Verlaag epsilon na een episode, maar nooit onder epsilon_end."""
        self.state.epsilon = max(self.epsilon_end, self.state.epsilon * self.epsilon_decay)
