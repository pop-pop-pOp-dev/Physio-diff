from __future__ import annotations

import torch
from torch import nn


class SignalToTextHead(nn.Module):
    def __init__(self, in_channels: int, text_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, text_dim),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(signal).squeeze(-1)
        return nn.functional.normalize(self.proj(feat), dim=-1)


class SignalTextCycle(nn.Module):
    def __init__(self, in_channels: int, text_dim: int):
        super().__init__()
        self.signal_to_text = SignalToTextHead(in_channels=in_channels, text_dim=text_dim)
        self.artifact_to_text = SignalToTextHead(in_channels=in_channels, text_dim=text_dim)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signal_to_text(signal), self.artifact_to_text(signal)
