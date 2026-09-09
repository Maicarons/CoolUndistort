# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Losses: La (L1) + Lb (boundary) + Lp (line/shape) + Lg (gradient)."""
from __future__ import annotations

import torch
from torch import nn


class DeformationLoss(nn.Module):
    def __init__(self, a1: float = 1e-2, a2: float = 1.0, a3: float = 1e-2) -> None:
        super().__init__()
        self.a1, self.a2, self.a3 = a1, a2, a3
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        la = self.l1(pred, target)
        lg = self.l1(pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
        # TODO(P1): Lb (prompt zero-level-set distance) + Lp (mesh line/shape penalty).
        return la + self.a3 * lg
