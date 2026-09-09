# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Visual prompts for T1..T4 (UniRect §Prompt Designs)."""
from __future__ import annotations

import numpy as np

TASKS = ("t1", "t2", "t3", "t4")


def make_prompt(task: str, height: int, width: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Return single-channel float32 prompt in [0, 1]."""
    task = task.lower()
    if task == "t4":
        return np.ones((height, width), dtype=np.float32)
    if mask is not None:
        m = mask.astype(np.float32)
        return (m - m.min()) / max(m.max() - m.min(), 1e-6)
    # T1/T2/T3 without detector output: fall back to ones; boundary masks
    # should come from face detector (T1) or border detection (T2/T3).
    return np.ones((height, width), dtype=np.float32)
