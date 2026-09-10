# CoolUndistort

统一图像去畸变与矩形整形：**推理全部 Rust** + Python 仅做训练，配套 Tauri 桌面 GUI。

- 研究背景见 [UNDISTORT_RESEARCH_REPORT.md](./UNDISTORT_RESEARCH_REPORT.md)
- 开发流程见 [docs/DEVELOPMENT.md](./docs/DEVELOPMENT.md)
- 训练见 [docs/TRAINING.md](./docs/TRAINING.md)，GUI 见 [docs/GUI.md](./docs/GUI.md)

## 功能（v1.1 真实可用）

- **自动盲去畸变（默认）**：`--mode auto` 单图盲估计除法模型 λ（内置 `weights/autolambda.onnx`，
  tract 纯 Rust 加载），CLI 侧 6 点 sweep MAE **0.0147**，开箱即用，无需标定。
- **Rust 推理核**（`crates/infer`，唯一推理实现）：标定模型（identity/division/Brown-Conrady + JSON）、
  TPS 闭式解 warp、T4 旋转校正、T1–T4 任务行为、边界 mask prompt、ONNX 真实加载。
- **CLI**（`crates/cli`，默认 auto）：单张/批量目录、`--calib/--lambda/--angle/--deltas/--grid/--onnx`、
  `--eval` 输出 PSNR+SSIM+lambda_est。
- **Python 训练**：AutoLambda（`scripts/train_autolambda.py --multiscale`，合成结构光数据，
  多尺度训练消除 serving skew）+ UniRect-lite（RP-TPS + Loss + prompt 烘焙 + 断点续训 + ONNX 导出）。
- **Tauri GUI**：默认 auto（显示估计 λ）、多选批量、前后对比滑块、标定面板、TPS/ONNX、保存结果。
- **标定**：`scripts/calibrate_opencv.py`（棋盘格 → calib.json）。

## 语言边界

- **Rust（推理全部）**：`crates/infer`、`crates/cli`、`gui/src-tauri`。
- **Python（仅训练 + 数据/标定/导出工具）**：`coolundistort/`、`scripts/`。不提供推理实现。

## 仓库结构

```
CoolUndistort/
├── LICENSE / Cargo.toml / pyproject.toml
├── crates/
│   ├── infer/src/   # lib/calibration/fisheye/tps/rotation/prompt/sampler/onnx
│   └── cli/src/     # coolundistort 二进制
├── coolundistort/   # 训练专用：model/ train.py datasets.py losses.py
├── configs/         # base / unirect_lite
├── scripts/         # prepare_data / calibrate_opencv / eval / export_onnx
├── gui/             # Tauri 应用：前端 invoke → Rust 后端推理
├── tests/           # pytest（训练前向冒烟）
└── docs/
```

## 快速开始

```powershell
# 推理（纯 Rust，无需 Python）
cargo run -p coolundistort-cli -- --input in.jpg --output out.png --task t2
cargo run -p coolundistort-cli -- --batch in_dir --out-dir out_dir --task t3 --eval

# 标定自己的镜头
python scripts/calibrate_opencv.py --images calib/*.jpg --pattern 9x6 --out calib.json
cargo run -p coolundistort-cli -- --input in.jpg --output out.png --calib calib.json

# 训练（Python，建议 WSL2；mamba-ssm 仅支持 Linux）
pip install -e ".[train]"
python scripts/prepare_data.py --task t3 --src raw/t3 --dst data/processed/t3
python -m coolundistort.train --config configs/unirect_lite.yaml
# 断点续训：python -m coolundistort.train --config ... --resume checkpoints/.../epoch_010.pt

# GUI（前端直调 Rust，无后端服务）
cd gui; npm install; npm run tauri dev
```

## License

AGPL-3.0-or-later，见 [LICENSE](./LICENSE)。网络服务分发修改版必须提供源码（AGPL §13）。
