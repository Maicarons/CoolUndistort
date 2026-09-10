# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""AutoLambda: blind division-model distortion estimation from a single image.

Real, self-contained inference task (cf. Li et al. 2019 blind geometric
distortion correction): a tiny CNN regresses the division-model parameter
lambda in [0, LAMBDA_MAX] from a distorted grayscale image. The Rust core
(`crates/infer/src/onnx.rs`) loads the exported ONNX with tract (pure Rust)
and undistorts with the estimated lambda. No calibration target needed.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

LAMBDA_MAX = 0.6
INPUT_SIZE = 128


class AutoLambdaNet(nn.Module):
    """~150k-param CNN: (1, 128, 128) grayscale -> lambda in [0, LAMBDA_MAX]."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1) * LAMBDA_MAX


def _norm_grids(size: int):
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = size / 2.0
    norm = float(np.hypot(cx, cy))
    return (xs - cx) / norm, (ys - cy) / norm, cx, cy, norm


def distort_division(img: np.ndarray, lam: float) -> np.ndarray:
    """Forward-distort a clean image with the division model (backward remap)."""
    h, w = img.shape[:2]
    dx, dy, cx, cy, norm = _norm_grids(max(h, w))
    dx = dx[:h, :w]
    dy = dy[:h, :w]
    rd = np.sqrt(dx * dx + dy * dy)
    r = rd.copy()
    for _ in range(8):
        denom = 1.0 + lam * r * r
        df = (1.0 - lam * r * r) / (denom * denom + 1e-12)
        r = np.where(rd > 1e-6, np.maximum(r - (r / denom - rd) / np.where(np.abs(df) < 1e-6, 1e-6, df), 0.0), 0.0)
    s = np.where(rd > 1e-6, r / np.maximum(rd, 1e-9), 1.0)
    map_x = (cx + dx * s * norm).astype(np.float32)
    map_y = (cy + dy * s * norm).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def random_pattern(rng: np.random.Generator, size: int) -> np.ndarray:
    """Structured pattern (lines/grid/circles/checker) on a smooth background."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    bg = 0.35 + 0.3 * np.sin(xx / size * 6.28 * rng.uniform(1, 3)) * np.cos(yy / size * 6.28 * rng.uniform(1, 3))
    img = np.clip(bg + rng.normal(0, 0.03, (size, size)), 0, 1)
    img = (img * 255).astype(np.uint8)
    kind = rng.integers(0, 4)
    col = int(rng.integers(180, 256))
    if kind == 0:  # random straight lines (strongest distortion cue)
        for _ in range(int(rng.integers(6, 15))):
            p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            p2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            cv2.line(img, p1, p2, col, int(rng.integers(1, 3)))
    elif kind == 1:  # grid
        step = int(rng.integers(10, 24))
        ox, oy = int(rng.integers(0, step)), int(rng.integers(0, step))
        for x in range(ox, size, step):
            cv2.line(img, (x, 0), (x, size - 1), col, 1)
        for y in range(oy, size, step):
            cv2.line(img, (0, y), (size - 1, y), col, 1)
    elif kind == 2:  # circles + spokes
        c = (int(rng.integers(size // 4, 3 * size // 4)), int(rng.integers(size // 4, 3 * size // 4)))
        for r in range(8, size // 2, int(rng.integers(8, 18))):
            cv2.circle(img, c, r, col, 1)
        for _ in range(int(rng.integers(4, 9))):
            a = rng.uniform(0, 6.283)
            cv2.line(img, c, (int(c[0] + size * np.cos(a)), int(c[1] + size * np.sin(a))), col, 1)
    else:  # checkerboard patch
        n = int(rng.integers(4, 9))
        cell = size // (n + 2)
        ox, oy = int(rng.integers(0, cell * 2)), int(rng.integers(0, cell * 2))
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 == 0:
                    cv2.rectangle(img, (ox + i * cell, oy + j * cell),
                                  (ox + (i + 1) * cell, oy + (j + 1) * cell), col, -1)
    return img


def make_sample(rng: np.random.Generator, size: int = INPUT_SIZE):
    """Return (distorted_gray_float01, lambda)."""
    lam = float(rng.uniform(0.0, LAMBDA_MAX))
    clean = random_pattern(rng, size)
    return distort_division(clean, lam).astype(np.float32) / 255.0, lam


def make_sample_multiscale(rng: np.random.Generator, out: int = INPUT_SIZE,
                           native_min: int = 96, native_max: int = 320):
    """Distort at a random native size, then resize to `out`.

    Simulates the serving path (arbitrary-size photo -> 128 thumbnail),
    so the model learns resize-invariant distortion cues.
    """
    lam = float(rng.uniform(0.0, LAMBDA_MAX))
    native = int(rng.integers(native_min, native_max + 1))
    clean = random_pattern(rng, native)
    dist = distort_division(clean, lam)
    small = cv2.resize(dist, (out, out), interpolation=cv2.INTER_AREA if native > out else cv2.INTER_LINEAR)
    return small.astype(np.float32) / 255.0, lam
