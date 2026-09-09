# CoolUndistort 开发文档

> 主文档。训练细节见 [TRAINING.md](./TRAINING.md)，GUI 见 [GUI.md](./GUI.md)，
> 论文研究见 [/UNDISTORT_RESEARCH_REPORT.md](../UNDISTORT_RESEARCH_REPORT.md)。

## 1. 项目目标与架构

做一个统一的去畸变桌面工具：输入畸变图 + 任务类型（人像 T1 / 广角整形 T2 / 拼接整形 T3 / 旋转校正 T4），输出校正图。

**语言边界（硬性）：推理全部 Rust，Python 只做训练。**

```
┌─────────────┐   Tauri invoke         ┌──────────────────────┐
│ Tauri GUI   │ ─────────────────────▶ │ Rust 推理核           │
│ Vite+TS     │ ◀───────────────────── │ crates/infer          │
│ 前端        │    base64 PNG          │  ├─ fisheye 除法模型   │
└─────────────┘                       │  ├─ prompt (T1-T4)    │
                                      │  └─ tps (P1 RP-TPS)   │
                                      └──────────────────────┘
┌──────────────────────┐
│ Python（仅训练）      │
│ coolundistort/train  │ ──产出权重──▶ Rust P1 加载（ONNX）
└──────────────────────┘
```

三阶段路线（与研究报告一致）：

- **P0 基线**：Rust 除法模型 + 标定参数，GUI/CLI 先调通。
- **P1 自研**：Python 训 UniRect-lite（RP-TPS + 轻量 RM + Visual Prompt），导出 ONNX 由 Rust 加载推理。
- **P2 统一**：Sparse MoE 四合一 + 剪枝 + 单文件分发。

## 2. 环境

| 场景 | 说明 |
|---|---|
| 推理/开发本机 | RTX 4060 Laptop 8G，Windows；`cargo` + Node 即可，无需 Python |
| 训练 | **WSL2 Ubuntu**（`mamba-ssm` 官方仅 Linux）。8G settings 见 TRAINING §3 |
| 溢出 | Kaggle（选 **T4**，不要 P100；30h/周；单 session ≤9h，必须断点续训） |

```powershell
# 推理：零 Python 依赖
cargo test
cargo run -p coolundistort-cli -- --input in.jpg --output out.png

# 训练（WSL2）
pip install -e ".[train]"
python -m coolundistort.train --config configs/unirect_lite.yaml
```

## 3. 目录与模块

```
crates/infer/src/
├── lib.rs        # 唯一推理入口 undistort(img, task, mode, params)
├── fisheye.rs    # 除法模型去畸变（P0 基线，单参数 lambda）
├── prompt.rs     # T1-T4 prompt kinds（P1 接检测器）
└── tps.rs        # RP-TPS 骨架（P1：TPS 闭式解 + grid_sample，权重来自训练）
crates/cli/src/main.rs  # coolundistort：--input/--output/--task/--mode/--lambda/--eval
coolundistort/          # 训练专用
├── model/        # deformation / restoration / prompts / unirect
├── train.py      # 唯一入口：python -m coolundistort.train --config ...
├── datasets.py   # 四任务统一 Dataset
└── losses.py     # La/Lb/Lp/Lg
configs/unirect_lite.yaml  # 8G 减配（img 256, batch 2 + 累积, RM×2）
gui/               # Tauri：前端 invoke("undistort_image") → Rust 后端，见 GUI.md
scripts/prepare_data.py   # 训练数据整理
scripts/eval.py           # 驱动 Rust CLI 在 processed 目录上算分
tests/test_smoke.py       # 训练前向冒烟（Rust 侧用 cargo test）
```

推理新增一律进 `crates/infer`（CLI、Tauri、批处理共享同一 `undistort` 入口），
禁止在 Python/TS 里另起推理实现。

## 4. 常用命令

```powershell
# 推理（Rust）
cargo test
cargo run -p coolundistort-cli -- --input in.jpg --output out.jpg --task t2 --mode fisheye --lambda 0.35

# 训练（WSL2，Python）
python -m coolundistort.train --config configs/unirect_lite.yaml
python scripts/eval.py --root data/processed/t3

# GUI
cd gui; npm install; npm run tauri dev

# 测试
cargo test; pytest tests/ -q
```

## 5. GitHub 协作规范

- 分支：`main` 稳定，`feat/*`、`docs/*` 短分支，PR 合并。
- 大文件不进仓：`data/`、`checkpoints/`、`runs/`、`results/` 已 ignore；权重放 Release 附件。
- 新增源码文件带 AGPL 头（Rust 用 `//` 注释头，见 crates/infer/src/lib.rs）。
- Issue 模板：bug（复现+环境）/ train（配置+loss 曲线）/ gui（截图+控制台日志）。

## 6. AGPL 合规要点

- 全仓 AGPL-3.0-or-later；MOWA 权重是 NTU S-Lab License，**只做本地评测锚点，不分发进包**。
- 若把推理做成网络服务，修改版必须向用户提供源码（§13）。
- GUI 关于页放 License 文本 + 源码链接。
