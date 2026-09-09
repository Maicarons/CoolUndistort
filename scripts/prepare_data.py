# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Prepare unified data layout: data/processed/{t1..t4}/{img,prompt,meta.json}.

Usage: python scripts/prepare_data.py --task t3 --src <raw_dir> --dst data/processed/t3
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["t1", "t2", "t3", "t4"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    (dst / "img").mkdir(parents=True, exist_ok=True)
    (dst / "prompt").mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in exts]
    for p in files:
        shutil.copy2(p, dst / "img" / p.name)
    (dst / "meta.json").write_text(json.dumps({"task": args.task, "count": len(files)}, indent=2), encoding="utf-8")
    print(f"task={args.task} copied {len(files)} images -> {dst}")


if __name__ == "__main__":
    main()
