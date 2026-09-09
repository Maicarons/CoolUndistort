# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Residual Mamba restoration blocks (lightweight: RMB x2 default)."""
from __future__ import annotations

from torch import nn


class ResidualMambaBlock(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        # TODO(P1): insert mamba-ssm block here (Linux/WSL2); fallback: channel attention.

    def forward(self, x):
        return x + self.body(x)


class RestorationModule(nn.Module):
    def __init__(self, channels: int = 32, n_blocks: int = 2) -> None:
        super().__init__()
        self.entry = nn.Conv2d(4, channels, 3, padding=1)  # image+prompt (partial-conv later)
        self.blocks = nn.Sequential(*[ResidualMambaBlock(channels) for _ in range(n_blocks)])
        self.exit = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, image, prompt):
        import torch

        x = torch.cat([image, prompt], dim=1)
        return self.exit(self.blocks(self.entry(x)))
