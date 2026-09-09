# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Prepare unified data layout: data/processed/<task>/{img,prompt,target,meta.json}.

Bakes training prompts per UniRect designs:
  T1: face-region mask (needs --faces dir with matching stems, else ones)
  T2/T3: border_mask from the image itself (dark padding -> 0)
  T4: all-ones

Usage: python scripts/prepare_data.py --task t3 --src <raw_dir> --dst data/processed/t3 [--faces faces/]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from coolundistort.model.prompts import border_mask

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["t1", "t2", "t3", "t4"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--faces", default="", help="optional face-mask dir for T1")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    (dst / "img").mkdir(parents=True, exist_ok=True)
    (dst / "prompt").mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in EXTS]
    for p in files:
        shutil.copy2(p, dst / "img" / p.name)
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        if args.task in ("t2", "t3"):
            mask = (border_mask(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) * 255).astype(np.uint8)
        elif args.task == "t1" and args.faces:
            cand = Path(args.faces) / (p.stem + ".png")
            mask = np.ones((h, w), np.uint8) * 255 if not cand.exists() else cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.ones((h, w), np.uint8) * 255
        cv2.imwrite(str(dst / "prompt" / (p.stem + ".png")), cv2.resize(mask, (w, h)))
    (dst / "meta.json").write_text(json.dumps({"task": args.task, "count": len(files)}, indent=2), encoding="utf-8")
    print(f"task={args.task} baked {len(files)} prompts -> {dst}")


if __name__ == "__main__":
    main()
