# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Losses: La (L1) + Lb (boundary) + Lp (line/shape) + Lg (gradient).

Mirrors UniRect Eq.9: L = sum_j gamma_j * (La + a1*Lb + a2*Lp + a3*Lg).
Boundary masks come from the training prompt PNGs (T2/T3 borders).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DeformationLoss(nn.Module):
    def __init__(self, a1: float = 1e-2, a2: float = 1.0, a3: float = 1e-2) -> None:
        super().__init__()
        self.a1, self.a2, self.a3 = a1, a2, a3
        self.l1 = nn.L1Loss()

    def boundary_loss(self, pred: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """Keep valid-content edges aligned: penalize drift where prompt is 0/1 boundary."""
        gx = prompt[..., :, 1:] - prompt[..., :, :-1]
        gy = prompt[..., 1:, :] - prompt[..., :-1, :]
        edge = (F.pad(gx.abs(), (0, 1, 0, 0)) + F.pad(gy.abs(), (0, 0, 0, 1))).clamp(0, 1)
        px = pred[..., :, 1:] - pred[..., :, :-1]
        return (F.pad(px.abs(), (0, 1, 0, 0)) * edge).mean()

    def shape_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Second-order smoothness: discourage kinks (mesh Lp surrogate)."""
        dxx = pred[..., :, 2:] - 2 * pred[..., :, 1:-1] + pred[..., :, :-2]
        dyy = pred[..., 2:, :] - 2 * pred[..., 1:-1, :] + pred[..., :-2, :]
        return dxx.abs().mean() + dyy.abs().mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                prompt: torch.Tensor | None = None) -> torch.Tensor:
        la = self.l1(pred, target)
        lg = self.l1(pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
        lb = self.boundary_loss(pred, prompt) if prompt is not None else pred.new_zeros(())
        lp = self.shape_loss(pred)
        return la + self.a1 * lb + self.a2 * 0.01 * lp + self.a3 * lg


class RestorationLoss(nn.Module):
    """Appearance (L1) + perceptual stub (P1: plug VGG features)."""

    def __init__(self, perceptual: float = 0.0) -> None:
        super().__init__()
        self.perceptual = perceptual
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target)
