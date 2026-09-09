# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Evaluate the Rust CLI on a processed task dir; writes results/metrics.json.

Usage: python scripts/eval.py --root data/processed/t3 --out results/metrics.json [--task t3] [--lambda 0.35]
NOTE: metrics are computed by driving the Rust binary (cargo run -p coolundistort-cli).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/metrics.json")
    ap.add_argument("--task", default="t3")
    ap.add_argument("--mode", default="fisheye")
    ap.add_argument("--lambda_", default="0.35")
    args = ap.parse_args()

    imgs = [p for p in sorted(Path(args.root, "img").glob("*")) if p.is_file()][:50]
    psnrs: list[float] = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in imgs:
            probe = Path(tmp, "probe.json")
            r = subprocess.run(
                ["cargo", "run", "-q", "-p", "coolundistort-cli", "--",
                 "--input", str(p), "--output", str(probe),
                 "--task", args.task, "--mode", args.mode,
                 "--lambda", args.lambda_, "--eval"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print(f"skip {p.name}: {r.stderr.strip().splitlines()[-1] if r.stderr else r.returncode}")
                continue
            psnrs.append(json.loads(probe.read_text(encoding="utf-8"))["psnr_self"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"mode": args.mode, "task": args.task, "n": len(psnrs),
                                          "psnr_mean": sum(psnrs) / len(psnrs) if psnrs else 0.0}, indent=2),
                             encoding="utf-8")
    print(f"n={len(psnrs)} -> {args.out}")


if __name__ == "__main__":
    main()
