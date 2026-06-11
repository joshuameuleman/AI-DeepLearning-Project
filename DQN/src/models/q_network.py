from __future__ import annotations

import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        
        # Oude net architectuur parameter mapping check:
        # Als old checkpoints geladen moeten worden raden we aan strict=False te gebruiken of
        # the architecture plain te initialiseren, maar standard gebruiken we Dueling DQN
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        
        # Value stream berekent de algemene "waarde" van het bord (veilig vs gevaarlijk)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream berekent het specifieke "voordeel" van een bepaalde actie
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        # Backward compatibility mode for old checkpoints fallback
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.use_legacy_net = False

    def forward(self, x):
        if self.use_legacy_net:
            return self.net(x)
            
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Custom loader om oude (niet-Dueling) modellen auto-detecting te laden."""
        if "net.0.weight" in state_dict and "feature_layer.0.weight" not in state_dict:
            previous_mode = self.use_legacy_net
            try:
                result = super().load_state_dict(state_dict, strict=False, assign=assign)
            except Exception:
                self.use_legacy_net = previous_mode
                raise
            self.use_legacy_net = True
            return result
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
