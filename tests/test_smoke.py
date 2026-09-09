# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Training-only smoke tests. Inference is pure Rust (see crates/infer)."""
import numpy as np
import torch

from coolundistort.model.prompts import make_prompt
from coolundistort.model.unirect import UniRectLite


def test_prompts():
    for t in ("t1", "t2", "t3", "t4"):
        p = make_prompt(t, 32, 32)
        assert p.shape == (32, 32)
        assert p.dtype == np.float32


def test_unirect_forward_cpu():
    model = UniRectLite(grid_h=4, grid_w=5, rm_blocks=1).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 32, 32), torch.ones(1, 1, 32, 32))
    assert out.shape == (1, 3, 32, 32)
