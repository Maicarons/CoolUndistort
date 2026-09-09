# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Training entry: python -m coolundistort.train --config configs/unirect_lite.yaml"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import UndistortDataset
from .losses import DeformationLoss
from .model.unirect import UniRectLite


def train_one_epoch(model, loader, optim, loss_fn, device, accum: int):
    model.train()
    optim.zero_grad()
    total = 0.0
    for i, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        img = batch["image"].to(device)
        prompt = batch["prompt"].to(device)
        target = batch["target"].to(device)
        pred = model(img, prompt)
        loss = loss_fn(pred, target) / accum
        loss.backward()
        if (i + 1) % accum == 0:
            optim.step()
            optim.zero_grad()
        total += loss.item() * accum
    return total / max(len(loader), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/unirect_lite.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniRectLite(grid_h=cfg["model"]["grid_h"], grid_w=cfg["model"]["grid_w"],
                        rm_blocks=cfg["model"]["rm_blocks"]).to(device)
    ds = UndistortDataset(cfg["data"]["root"], cfg["data"]["task"], cfg["data"]["size"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=2)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = DeformationLoss()

    ckpt = Path(cfg["train"]["ckpt_dir"])
    ckpt.mkdir(parents=True, exist_ok=True)
    scaler_on = cfg["train"]["amp"] and device.type == "cuda"
    for epoch in range(cfg["train"]["epochs"]):
        avg = train_one_epoch(model, loader, optim, loss_fn, device, cfg["train"]["accum"])
        print(f"epoch {epoch + 1}/{cfg['train']['epochs']} loss={avg:.4f}", flush=True)
        torch.save(model.state_dict(), ckpt / f"epoch_{epoch + 1:03d}.pt")


if __name__ == "__main__":
    main()
