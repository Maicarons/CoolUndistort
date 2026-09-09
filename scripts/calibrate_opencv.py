# Copyright (C) 2026 CoolUndistort contributors. See LICENSE (AGPL-3.0-or-later).
"""Checkerboard calibration -> Rust-compatible calibration JSON.

This is TRAINING-adjacent tooling (Python): it produces the `calib.json`
that the Rust inference core (`crates/infer`) loads. No inference here.

Standard lens:  python scripts/calibrate_opencv.py --images calib/*.jpg --pattern 9x6 --out calib.json
Fisheye lens:   ... --fisheye
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def find_corners(path: Path, pattern: tuple[int, int]):
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, pattern)
    if not ok:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    return img.shape[1::-1], corners.reshape(-1, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="glob, e.g. calib/*.jpg")
    ap.add_argument("--pattern", default="9x6", help="inner corners WxH")
    ap.add_argument("--fisheye", action="store_true")
    ap.add_argument("--out", default="calib.json")
    args = ap.parse_args()

    import glob as globmod

    w, h = (int(v) for v in args.pattern.lower().split("x"))
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    obj_pts, img_pts, size = [], [], None
    for f in sorted(globmod.glob(args.images)):
        r = find_corners(Path(f), (w, h))
        if r is None:
            print(f"skip (no corners): {f}")
            continue
        size, corners = r
        obj_pts.append(objp)
        img_pts.append(corners)
    if not obj_pts:
        raise SystemExit("no usable calibration images")

    if args.fisheye:
        K = np.eye(3)
        D = np.zeros((4, 1))
        _, K, D, _, _ = cv2.fisheye.calibrate([o.reshape(1, -1, 3) for o in obj_pts],
                                              [i.reshape(1, -1, 2) for i in img_pts],
                                              size, K, D)
        # Fit division-model lambda from the fisheye D1 as a starting point.
        calib = {"model": "division", "lambda": float(min(max(-float(D[0][0]) * 2.0, 0.0), 1.0))}
    else:
        _, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
        k1, k2, p1, p2, k3 = (list(dist.flatten()) + [0.0] * 5)[:5]
        calib = {"model": "brown_conrady", "k1": float(k1), "k2": float(k2),
                 "k3": float(k3), "p1": float(p1), "p2": float(p2)}
    Path(args.out).write_text(json.dumps(calib, indent=2), encoding="utf-8")
    print(f"calibrated on {len(obj_pts)} views -> {args.out}: {calib}")


if __name__ == "__main__":
    main()
