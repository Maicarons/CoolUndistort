# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Unified dataset layout: data/processed/{t1..t4}/{img,prompt,meta.json}.

Each sample: distorted input + prompt + rectified target. When a task has
no pixel GT (T1 portraits), target falls back to the input and training
uses prompt-weighted losses only.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class UndistortDataset(Dataset):
    def __init__(self, root: str | Path, task: str, size: int = 256) -> None:
        self.root = Path(root)
        self.task = task
        self.size = size
        self.images = sorted(p for p in (self.root / "img").glob("*") if p.is_file())
        if not self.images:
            raise FileNotFoundError(f"no images in {self.root / 'img'} (run scripts/prepare_data.py)")

    def __len__(self) -> int:
        return len(self.images)

    def _load(self, p: Path, gray: bool) -> np.ndarray | None:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.resize(img, (self.size, self.size))

    def __getitem__(self, idx: int):
        import torch

        p = self.images[idx]
        img = self._load(p, gray=False)
        if img is None:
            raise RuntimeError(f"cannot read {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        prompt_path = self.root / "prompt" / (p.stem + ".png")
        if prompt_path.exists():
            prompt = self._load(prompt_path, gray=True).astype(np.float32) / 255.0
        else:
            prompt = np.ones((self.size, self.size), dtype=np.float32)
        target_path = self.root / "target" / p.name
        if target_path.exists():
            target = cv2.cvtColor(self._load(target_path, gray=False), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            target = img.copy()
        img_t = torch.from_numpy(img).permute(2, 0, 1)
        return {"image": img_t, "prompt": torch.from_numpy(prompt).unsqueeze(0),
                "target": torch.from_numpy(target).permute(2, 0, 1), "name": p.name}
