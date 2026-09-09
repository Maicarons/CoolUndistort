# CoolUndistort 开发文档

> 主文档。训练细节见 [TRAINING.md](./TRAINING.md)，GUI 见 [GUI.md](./GUI.md)，
> 论文研究见 [/UNDISTORT_RESEARCH_REPORT.md](../UNDISTORT_RESEARCH_REPORT.md)。

## 1. 项目目标与架构

统一去畸变桌面工具：输入畸变图 + 任务类型（T1 人像 / T2 广角整形 / T3 拼接整形 / T4 旋转校正），输出校正图。

**语言边界（硬性）：推理全部 Rust，Python 只做训练 + 数据/标定/导出工具。**

```
┌─────────────┐  Tauri invoke  ┌──────────────────────┐
│ Tauri GUI   │ ─────────────▶ │ Rust 推理核           │
│ Vite+TS     │ ◀───────────── │ crates/infer          │
│ 对比/批量   │  base64 PNG    │ 标定/TPS/旋转/prompt  │
└─────────────┘               └──────────────────────┘
┌─────────────┐  ONNX  ┌───────┴──────────────┐
│ Python 训练 │ ──────▶│ Rust checkpoint 模式  │
└─────────────┘       └──────────────────────┘
```

## 2. 环境

| 场景 | 说明 |
|---|---|
| 推理/开发本机 | Windows + `cargo` + Node 即可，无需 Python |
| 训练 | **WSL2 Ubuntu**（`mamba-ssm` 官方仅 Linux）。8G settings 见 TRAINING §3 |
| 溢出 | Kaggle（选 **T4**，不要 P100；30h/周；单 session ≤9h，靠 `--resume` 接力） |

## 3. 目录与模块

```
crates/infer/src/
├── lib.rs        # 唯一推理入口 undistort / undistort_full(img, task, mode, params)
├── calibration.rs# identity/division/Brown-Conrady + JSON load/save
├── fisheye.rs    # 除法模型（经 calibration）
├── tps.rs        # TPS 闭式解 + tps_warp_with_deltas（与训练 warp 对齐）
├── rotation.rs   # T4 旋转校正
├── prompt.rs     # T1-T4 prompt kinds + border_mask（与训练烘焙对齐）
├── sampler.rs    # 双线性采样 + backward remap
└── onnx.rs       # checkpoint 钩子（P1 接 ort）
crates/cli/src/main.rs  # 单张/批量/--eval(PSNR+SSIM)/标定/TPS/ONNX 参数
coolundistort/          # 训练专用：model(train TPS)/train.py/datasets.py/losses.py
scripts/                # prepare_data / calibrate_opencv / eval(调 Rust CLI) / export_onnx
gui/                    # Tauri：对比滑块/批量队列/标定面板/保存/关于，见 GUI.md
```

推理新增一律进 `crates/infer`（CLI、Tauri、批处理共享同一入口），
禁止在 Python/TS 里另起推理实现。

## 4. 常用命令

```powershell
# 推理（Rust）
cargo test
cargo run -p coolundistort-cli -- --input in.jpg --output out.jpg --task t2
cargo run -p coolundistort-cli -- --batch in_dir --out-dir out_dir --task t3 --eval
python scripts/calibrate_opencv.py --images calib/*.jpg --pattern 9x6 --out calib.json

# 训练（WSL2，Python）
pip install -e ".[train]"
python scripts/prepare_data.py --task t3 --src raw/t3 --dst data/processed/t3
python -m coolundistort.train --config configs/unirect_lite.yaml
python scripts/export_onnx.py --ckpt checkpoints/unirect_lite_t3/best.pt --out weights/unirect_lite_t3.onnx

# GUI / 测试
cd gui; npm install; npm run tauri dev
cargo test; pytest tests/ -q
```

## 5. GitHub 协作规范

- 分支：`main` 稳定，`feat/*`、`docs/*` 短分支，PR 合并。
- 大文件不进仓：`data/`、`checkpoints/`、`runs/`、`results/`、`weights/` 已 ignore；权重放 Release 附件。
- 新增源码文件带 AGPL 头（Rust 用 `//` 注释头）。
- Issue 模板：bug（复现+环境）/ train（配置+loss 曲线）/ gui（截图+控制台日志）。

## 6. AGPL 合规要点

- 全仓 AGPL-3.0-or-later；MOWA 权重是 NTU S-Lab License，**只做本地评测锚点，不分发进包**。
- 若把推理做成网络服务，修改版必须向用户提供源码（§13）。
- GUI 关于页（footer）放 License 文本 + 源码链接。
