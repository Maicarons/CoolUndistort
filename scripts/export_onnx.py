# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Export UniRect-lite (.pt) to ONNX for the Rust inference core.

Training-only tooling. The Rust side (`crates/infer/src/onnx.rs`) loads
the produced file; Python never runs inference with it.

Usage: python scripts/export_onnx.py --ckpt checkpoints/unirect_lite_t3/epoch_200.pt --out weights/unirect_lite_t3.onnx [--size 256]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from coolundistort.model.unirect import UniRectLite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--grid-h", type=int, default=10)
    ap.add_argument("--grid-w", type=int, default=12)
    ap.add_argument("--rm-blocks", type=int, default=2)
    args = ap.parse_args()

    model = UniRectLite(grid_h=args.grid_h, grid_w=args.grid_w, rm_blocks=args.rm_blocks).eval()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    img = torch.rand(1, 3, args.size, args.size)
    prompt = torch.ones(1, 1, args.size, args.size)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (img, prompt), str(out), input_names=["image", "prompt"],
                      output_names=["corrected"], opset_version=17, dynamo=False)
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
