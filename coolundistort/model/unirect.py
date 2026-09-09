# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""UniRect-lite assembly: DM(RP-TPS) -> RM(Mamba)."""
from __future__ import annotations

from torch import nn

from .deformation import ResidualProgressiveTPS
from .restoration import RestorationModule


class UniRectLite(nn.Module):
    def __init__(self, grid_h: int = 10, grid_w: int = 12, rm_blocks: int = 2) -> None:
        super().__init__()
        self.dm = ResidualProgressiveTPS(grid_h, grid_w)
        self.rm = RestorationModule(n_blocks=rm_blocks)

    def forward(self, image, prompt):
        warped = self.dm(image, prompt)
        return self.rm(warped, prompt)
