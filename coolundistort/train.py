# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Training entry: python -m coolundistort.train --config configs/unirect_lite.yaml

Features: AMP, gradient accumulation, train/val split, resume from
checkpoint (--resume ckpt.pt), per-epoch checkpoints + best.pt,
TensorBoard logging. Kaggle-safe: every epoch is resumable (9h cap).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .datasets import UndistortDataset
from .losses import DeformationLoss
from .model.unirect import UniRectLite


def run_epoch(model, loader, loss_fn, device, optim=None, accum=1, scaler=None):
    train = optim is not None
    model.train(train)
    total, n = 0.0, 0
    stepped = False
    ctx = torch.enable_grad() if train else torch.no_grad()
    if train and optim is not None:
        optim.zero_grad()
    with ctx:
        for i, batch in enumerate(tqdm(loader, desc="train" if train else "val", leave=False)):
            img = batch["image"].to(device)
            prompt = batch["prompt"].to(device)
            target = batch["target"].to(device)
            with torch.autocast(device.type, enabled=scaler is not None):
                pred = model(img, prompt)
                loss = loss_fn(pred, target, prompt)
            if train:
                (loss / accum).backward()
                if (i + 1) % accum == 0:
                    if scaler is not None:
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                    optim.zero_grad()
                    stepped = True
            total += loss.item()
            n += 1
    if train and not stepped and n > 0:
        # Loader shorter than one accumulation window: flush remaining grads.
        if scaler is not None:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
        optim.zero_grad()
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/unirect_lite.yaml")
    ap.add_argument("--resume", default="")
    ap.add_argument("--epochs", type=int, default=0, help="override config epochs")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tcfg = cfg["train"]

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    model = UniRectLite(grid_h=mcfg["grid_h"], grid_w=mcfg["grid_w"],
                        rm_blocks=mcfg.get("rm_blocks", 2)).to(device)
    dcfg = cfg["data"]
    full = UndistortDataset(dcfg["root"], dcfg["task"], dcfg["size"])
    n_val = min(max(len(full) // 10, 1), len(full) - 1)
    train_ds, val_ds = random_split(full, [len(full) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], num_workers=2)
    optim = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda e: 0.96 ** e)
    scaler = torch.amp.GradScaler("cuda") if tcfg.get("amp") and device.type == "cuda" else None
    loss_fn = DeformationLoss()

    ckpt = Path(tcfg["ckpt_dir"])
    ckpt.mkdir(parents=True, exist_ok=True)
    start, best = 1, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state.get("model", state))
        if "optim" in state:
            optim.load_state_dict(state["optim"])
        start = int(state.get("epoch", 0)) + 1
        best = float(state.get("best", best))
        print(f"resumed {args.resume} at epoch {start}")

    epochs = args.epochs or tcfg["epochs"]
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(ckpt / "logs"))
    except Exception:
        pass
    for epoch in range(start, epochs + 1):
        tr = run_epoch(model, train_loader, loss_fn, device, optim, tcfg.get("accum", 1), scaler)
        va = run_epoch(model, val_loader, loss_fn, device)
        sched.step()
        torch.save({"epoch": epoch, "model": model.state_dict(), "optim": optim.state_dict(), "best": min(best, va)},
                   ckpt / f"epoch_{epoch:03d}.pt")
        if va < best:
            best = va
            torch.save(model.state_dict(), ckpt / "best.pt")
        if writer is not None:
            writer.add_scalar("loss/train", tr, epoch)
            writer.add_scalar("loss/val", va, epoch)
        print(f"epoch {epoch}/{epochs} train={tr:.4f} val={va:.4f} best={best:.4f}", flush=True)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
