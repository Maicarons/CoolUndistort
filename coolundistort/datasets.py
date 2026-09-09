# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Unified dataset layout: data/processed/{t1..t4}/{img,prompt,meta.json}."""
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
        self.images = sorted((self.root / "img").glob("*"))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        import torch

        p = self.images[idx]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size)).astype(np.float32) / 255.0
        prompt_path = self.root / "prompt" / (p.stem + ".png")
        if prompt_path.exists():
            prompt = cv2.imread(str(prompt_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            prompt = cv2.resize(prompt, (self.size, self.size))
        else:
            prompt = np.ones((self.size, self.size), dtype=np.float32)
        img_t = torch.from_numpy(img).permute(2, 0, 1)
        prompt_t = torch.from_numpy(prompt).unsqueeze(0)
        return {"image": img_t, "prompt": prompt_t, "target": img_t.clone(), "name": p.name}
