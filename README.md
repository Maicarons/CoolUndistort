# CoolUndistort

统一图像去畸变与矩形整形：**推理全部 Rust** + Python 仅做训练，配套 Tauri 桌面 GUI。

- 研究背景见 [UNDISTORT_RESEARCH_REPORT.md](./UNDISTORT_RESEARCH_REPORT.md)
- 开发流程见 [docs/DEVELOPMENT.md](./docs/DEVELOPMENT.md)
- 训练见 [docs/TRAINING.md](./docs/TRAINING.md)，GUI 见 [docs/GUI.md](./docs/GUI.md)

## 语言边界

- **Rust（推理全部）**：`crates/infer`（推理核：除法模型基线 + prompt + RP-TPS 骨架）、
  `crates/cli`（`coolundistort` 命令行）、`gui/src-tauri`（Tauri 后端直调推理核）。
- **Python（仅训练）**：`coolundistort/`（UniRect-lite 模型 + 训练入口 + 数据集 + 损失）。
  不提供推理/服务入口；评测脚本 `scripts/eval.py` 只是驱动 Rust CLI 算分。

## 仓库结构

```
CoolUndistort/
├── LICENSE                  # AGPL-3.0-or-later
├── Cargo.toml                 # Rust workspace (crates/*)
├── crates/
│   ├── infer/                 # 推理核：fisheye / prompt / tps（唯一推理实现）
│   └── cli/                   # coolundistort 二进制：--input/--output/--task/--mode/--eval
├── pyproject.toml             # Python 训练包 (coolundistort-train)
├── coolundistort/             # 训练专用：model/ train.py datasets.py losses.py
├── configs/                   # 训练配置 (base / unirect_lite)
├── scripts/                   # prepare_data.py（训练数据）/ eval.py（调 Rust CLI 评测）
├── gui/                       # Tauri 应用：前端 invoke → Rust 后端推理
├── tests/                     # pytest（训练前向冒烟）
└── docs/                      # 文档
```

## 快速开始

```powershell
# 推理（纯 Rust，无需 Python）
cargo run -p coolundistort-cli -- --input in.jpg --output out.png --task t2 --mode fisheye

# 训练（Python，建议 WSL2；mamba-ssm 仅支持 Linux）
pip install -e ".[train]"
python -m coolundistort.train --config configs/unirect_lite.yaml

# GUI（前端直调 Rust，无后端服务）
cd gui; npm install; npm run tauri dev
```

## License

AGPL-3.0-or-later，见 [LICENSE](./LICENSE)。网络服务分发修改版必须提供源码（AGPL §13）。
