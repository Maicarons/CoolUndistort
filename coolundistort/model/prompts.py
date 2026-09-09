# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Visual prompts for T1..T4 (UniRect prompt designs).

Used at TRAINING time to bake prompt PNGs (data/processed/<task>/prompt/).
Mirrors crates/infer/src/prompt.rs, which derives the runtime prompt.
"""
from __future__ import annotations

import numpy as np

TASKS = ("t1", "t2", "t3", "t4")


def normalize(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    return (m - m.min()) / max(float(m.max() - m.min()), 1e-6)


def border_mask(img: np.ndarray, threshold: int = 16) -> np.ndarray:
    """Valid-content mask for stitched / rectified wide images (dark padding -> 0)."""
    f = img.astype(np.float32)
    luma = (f[..., 0] * 299 + f[..., 1] * 587 + f[..., 2] * 114) / 1000
    return (luma > threshold).astype(np.float32)


def make_prompt(task: str, height: int, width: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Return single-channel float32 prompt in [0, 1]."""
    task = task.lower()
    if task == "t4":
        return np.ones((height, width), dtype=np.float32)
    if mask is not None:
        return normalize(mask)
    return np.ones((height, width), dtype=np.float32)
