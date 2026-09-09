# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""RP-TPS deformation module (two-step residual control points).

Training-side mirror of crates/infer/src/tps.rs: uniform base points,
closed-form TPS solve in double precision, grid_sample warp. Both
intermediate samples come from the ORIGINAL input (no error buildup).

Reference: UniRect (arXiv:2512.18718) Fig.2b; solver style follows
RecRecNet / CoupledTPS.
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
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(hidden, n_points * 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, prompt], dim=1)
        feat = self.encoder(x).flatten(1)
        return self.head(feat).view(-1, self.n_points, 2)


def tps_parameters(src: torch.Tensor, dst: torch.Tensor):
    """Closed-form TPS solve. src/dst: (..., N, 2) normalized. Returns (w, A)."""
    src_d, dst_d = src.double(), dst.double()
    n = src_d.shape[-1] if src_d.dim() == 2 else src_d.shape[-2]
    s = src_d.reshape(-1, n, 2)
    d = dst_d.reshape(-1, n, 2)
    b = s.shape[0]
    diff = s[:, :, None, :] - s[:, None, :, :]
    r2 = (diff ** 2).sum(-1).clamp_min(1e-12)
    K = torch.where(r2 < 1e-10, torch.zeros_like(r2), r2 * r2.log())
    ones = torch.ones(b, n, 1, dtype=torch.float64, device=s.device)
    P = torch.cat([ones, s], dim=-1)
    zeros3 = torch.zeros(b, 3, 3, dtype=torch.float64, device=s.device)
    top = torch.cat([K, P], dim=-1)
    bottom = torch.cat([P.transpose(1, 2), zeros3], dim=-1)
    L = torch.cat([top, bottom], dim=1)
    Y = torch.cat([d, torch.zeros(b, 3, 2, dtype=torch.float64, device=s.device)], dim=1)
    sol = torch.linalg.solve(L, Y)
    return sol[:, :n, :].float(), sol[:, n:, :].float()


def tps_grid(src: torch.Tensor, dst: torch.Tensor, out_h: int, out_w: int, device: torch.device) -> torch.Tensor:
    """Sampling grid (1, H, W, 2) mapping rectified pixels -> distorted source."""
    w, A = tps_parameters(src, dst)
    ys = torch.linspace(-1, 1, out_h, device=device)
    xs = torch.linspace(-1, 1, out_w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    flat = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).double()
    diff = flat[None, :, None, :] - src.double().reshape(src.shape[0], 1, -1, 2)
    r2 = (diff ** 2).sum(-1).clamp_min(1e-12)
    U = torch.where(r2 < 1e-10, torch.zeros_like(r2), r2 * r2.log())
    mapped = U @ w.double() + torch.cat(
        [torch.ones_like(flat[:, :1]), flat], dim=-1) @ A.double()
    return mapped.float().view(src.shape[0], out_h, out_w, 2)


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

    def warp(self, image: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        base = self.base_points(image.device).unsqueeze(0).expand(b, -1, -1)
        grid = tps_grid(base, dst, image.shape[-2], image.shape[-1], image.device)
        return torch.nn.functional.grid_sample(image, grid, align_corners=True)

    def forward(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        base = self.base_points(image.device).unsqueeze(0).expand(b, -1, -1)
        c1 = (base + self.c0(image, prompt)).clamp(-1.5, 1.5)
        x1 = self.warp(image, c1)
        m1 = self.warp(prompt, c1)
        c2 = (c1 + self.c1(x1, m1)).clamp(-1.5, 1.5)
        return self.warp(image, c2)
