# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""RP-TPS deformation module (two-step residual control points).

P1 implementation target. Reference: UniRect (arXiv:2512.18718) Fig.2b,
TPS solver style follows RecRecNet / CoupledTPS.
"""
from __future__ import annotations

import torch
from torch import nn


class ControlPointPredictor(nn.Module):
    """Predicts residual offsets of control points from (image, prompt)."""

    def __init__(self, n_points: int = 120, hidden: int = 64) -> None:
        super().__init__()
        self.n_points = n_points
        self.encoder = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(hidden, n_points * 2)

    def forward(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, prompt], dim=1)
        feat = self.encoder(x).flatten(1)
        return self.head(feat).view(-1, self.n_points, 2)


class ResidualProgressiveTPS(nn.Module):
    """c1 = c0 + C0(X0, M0); c2 = c1 + C1(X1, M1); sample twice from X0."""

    def __init__(self, grid_h: int = 10, grid_w: int = 12) -> None:
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        n = grid_h * grid_w
        self.c0 = ControlPointPredictor(n)
        self.c1 = ControlPointPredictor(n)

    def base_points(self, device: torch.device) -> torch.Tensor:
        ys = torch.linspace(-1, 1, self.grid_h, device=device)
        xs = torch.linspace(-1, 1, self.grid_w, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    def forward(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        base = self.base_points(image.device)
        c1 = base + self.c0(image, prompt)
        x1 = self.warp(image, base, c1)
        m1 = self.warp(prompt, base, c1)
        c2 = c1 + self.c1(x1, m1)
        return self.warp(image, base, c2)

    def warp(self, image: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        # TODO(P1): replace affine placeholder with full TPS closed-form solve.
        theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=image.device)
        grid = torch.nn.functional.affine_grid(theta.repeat(image.size(0), 1, 1), image.size(), align_corners=True)
        return torch.nn.functional.grid_sample(image, grid, align_corners=True)
