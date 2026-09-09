# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Training-only smoke tests. Inference is pure Rust (see crates/infer)."""
import numpy as np
import torch

from coolundistort.losses import DeformationLoss
from coolundistort.model.deformation import tps_parameters
from coolundistort.model.prompts import border_mask, make_prompt
from coolundistort.model.unirect import UniRectLite


def test_prompts():
    for t in ("t1", "t2", "t3", "t4"):
        p = make_prompt(t, 32, 32)
        assert p.shape == (32, 32)
        assert p.dtype == np.float32


def test_border_mask():
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, 4:] = 200
    m = border_mask(img)
    assert m[0, 0] == 0.0
    assert m[0, 7] == 1.0


def test_tps_identity():
    s = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    w, a = tps_parameters(s, s)
    assert w.shape == (1, 4, 2)
    assert a.shape == (1, 3, 2)


def test_unirect_forward_cpu():
    model = UniRectLite(grid_h=4, grid_w=5, rm_blocks=1).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 32, 32), torch.ones(1, 1, 32, 32))
    assert out.shape == (1, 3, 32, 32)


def test_loss_terms():
    fn = DeformationLoss()
    pred = torch.rand(1, 3, 16, 16)
    target = torch.rand(1, 3, 16, 16)
    prompt = torch.ones(1, 1, 16, 16)
    loss = fn(pred, target, prompt)
    assert torch.isfinite(loss)
