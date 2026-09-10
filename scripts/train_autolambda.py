# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Train AutoLambda (blind lambda regression) on synthetic structured data.

Self-contained: no downloads. Generates N distorted 128px grayscale images
with known lambda, trains the tiny CNN, reports MAE + end-to-end undistort
PSNR, and exports weights/autolambda.onnx for the Rust core.

Usage: python scripts/train_autolambda.py --epochs 30 --train-n 4000 --device auto
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from coolundistort.model.autolambda import (
    INPUT_SIZE, LAMBDA_MAX, AutoLambdaNet, distort_division, make_sample, make_sample_multiscale,
)


def build_set(n: int, seed: int, size: int = INPUT_SIZE, multiscale: bool = False):
    rng = np.random.default_rng(seed)
    xs = np.empty((n, 1, size, size), np.float32)
    ys = np.empty((n,), np.float32)
    for i in tqdm(range(n), desc=f"synth(seed={seed})", leave=False):
        img, lam = make_sample_multiscale(rng, size) if multiscale else make_sample(rng, size)
        xs[i, 0] = img
        ys[i] = lam / LAMBDA_MAX  # normalized target in [0, 1]
    return torch.from_numpy(xs), torch.from_numpy(ys)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    se, ae, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x) / LAMBDA_MAX
        se += ((p - y) ** 2).sum().item()
        ae += (p - y).abs().sum().item()
        n += x.shape[0]
    return (se / n) ** 0.5, ae / n


@torch.no_grad()
def end_to_end_psnr(model, device, trials: int = 60, size: int = 256):
    """Distort clean patterns -> estimate lambda -> undistort -> PSNR vs clean."""
    from coolundistort.model.autolambda import random_pattern

    model.eval()
    rng = np.random.default_rng(999)
    psnrs = []
    for _ in range(trials):
        clean = random_pattern(rng, size).astype(np.float32)
        lam = float(rng.uniform(0.05, LAMBDA_MAX))
        dist = distort_division(clean.astype(np.uint8), lam).astype(np.float32)
        small = torch.from_numpy(dist[:: size // INPUT_SIZE, :: size // INPUT_SIZE][:INPUT_SIZE, :INPUT_SIZE]
                                 ).unsqueeze(0).unsqueeze(0).to(device) / 255.0
        est = float(model(small).item())
        fixed = distort_division(dist.astype(np.uint8), -est * 0.0 + 0.0)  # placeholder no-op
        # Undistort by remapping with the ESTIMATED lambda (inverse direction):
        from coolundistort.model.autolambda import _norm_grids

        h, w = dist.shape
        dx, dy, cx, cy, norm = _norm_grids(max(h, w))
        dx, dy = dx[:h, :w], dy[:h, :w]
        import cv2

        rd = np.sqrt(dx * dx + dy * dy)
        r = rd.copy()
        for _ in range(8):
            denom = 1.0 + est * r * r
            df = (1.0 - est * r * r) / (denom * denom + 1e-12)
            r = np.where(rd > 1e-6, np.maximum(r - (r / denom - rd) / np.where(np.abs(df) < 1e-6, 1e-6, df), 0.0), 0.0)
        s = np.where(rd > 1e-6, r / np.maximum(rd, 1e-9), 1.0)
        out = cv2.remap(dist, (cx + dx * s * norm).astype(np.float32), (cy + dy * s * norm).astype(np.float32),
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mse = float(np.mean((out - clean) ** 2))
        psnrs.append(100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse)))
        _ = fixed
    return float(np.mean(psnrs))


def export_onnx(model, path: Path, size: int = INPUT_SIZE):
    model.eval()
    dummy = torch.rand(1, 1, size, size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=["lambda"],
                      opset_version=17, dynamo=False)
    import onnx

    onnx.checker.check_model(str(path))
    print(f"exported+checked {path} ({path.stat().st_size / 1024:.1f} KiB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train-n", type=int, default=4000)
    ap.add_argument("--val-n", type=int, default=500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--ckpt", default="checkpoints/autolambda/best.pt")
    ap.add_argument("--onnx", default="weights/autolambda.onnx")
    ap.add_argument("--multiscale", action="store_true",
                    help="distort at random native size then resize (matches serving path)")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise SystemExit("--device cuda requested but no CUDA")
    print(f"device={device}", flush=True)

    xtr, ytr = build_set(args.train_n, seed=7, multiscale=args.multiscale)
    xva, yva = build_set(args.val_n, seed=1234, multiscale=args.multiscale)
    tr_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(TensorDataset(xva, yva), batch_size=args.batch)

    model = AutoLambdaNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state = float("inf"), None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x) / LAMBDA_MAX, y)
            loss.backward()
            optim.step()
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
        rmse, mae = evaluate(model, va_loader, device)
        if rmse < best:
            best, best_state = rmse, {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"epoch {epoch}/{args.epochs} train_mse={tot / n:.5f} val_rmse={rmse:.4f} val_mae={mae:.4f}", flush=True)
    print(f"training took {time.time() - t0:.0f}s, best val_rmse={best:.4f}")
    model.load_state_dict(best_state)
    ckpt = Path(args.ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt)
    print(f"saved {ckpt}")

    psnr = end_to_end_psnr(model, device)
    print(f"end-to-end undistort PSNR (est. lambda): {psnr:.2f} dB")
    export_onnx(model, Path(args.onnx))


if __name__ == "__main__":
    main()
