"""
Deep learning models: a Feedforward Neural Network for tabular features,
and an LSTM for sequential per-card transaction behavior.
"""
import torch
import torch.nn as nn


class FraudFNN(nn.Module):
    """Feedforward network for tabular transaction features."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


class FraudLSTM(nn.Module):
    """LSTM over a sequence of a cardholder's recent transactions."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return torch.sigmoid(self.fc(last_hidden)).squeeze(-1)
