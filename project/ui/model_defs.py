"""Model definitions shared by the training notebooks and the Streamlit app.

The class names and layer shapes must match the saved state dictionaries.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


class CNNRegressor(nn.Module):
    def __init__(self, input_channels: int, channels=(32, 64),
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_ch = input_channels
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(in_ch + input_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        spatial = self.pool(self.features(x)).flatten(1)
        current = x[:, :, -1, -1]
        return self.head(torch.cat([spatial, current], dim=1))


class Autoencoder(nn.Module):
    def __init__(self, input_size: int, bottleneck: int = 8, hidden: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class RegressionHead(nn.Module):
    def __init__(self, input_size: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)
