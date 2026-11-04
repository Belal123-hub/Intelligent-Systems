"""The MLP used throughout Lab 0.

Architecture (Lab 0 §3.2):
  64 inputs -> n_layers hidden layers of `hidden` units with ReLU + dropout
            -> 10 outputs
Loss: cross-entropy. Optimizer: Adam.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_layers: int, hidden: int, dropout: float, n_classes: int = 10, n_inputs: int = 64):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0,1], got {dropout}")

        layers = []
        in_features = n_inputs
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_features = hidden
        layers.append(nn.Linear(in_features, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)