# 训练文档（Python 专用；推理全部在 Rust）

> 两条训练线：
> - **AutoLambda（已训好，权重进仓）**：单图盲估计除法模型 λ，小 CNN + 合成结构光数据，
>   Rust 用 tract 真实加载推理。见 §0。
> - **UniRect-lite（论文复现线）**：对应论文 arXiv:2512.18718，见 §1–§5。

## 0. AutoLambda（真实可用，默认推理路径）

```powershell
python scripts/train_autolambda.py --epochs 30 --train-n 4000 --device cpu --multiscale
# 产物：checkpoints/autolambda/best.pt + weights/autolambda.onnx（~400KB，进仓）
python scripts/eval_auto.py --trials 30   # 真实 Rust CLI 精度评估
```

- 任务：盲回归 λ∈[0, 0.6]（Li et al. 2019 一脉的发表任务类型），~150k 参数 CNN，输入 128 灰度图。
- 数据：自包含合成（直线/网格/圆/棋盘格 + smooth 背景 + 噪声），`--multiscale` 在随机原生尺寸
  畸变后再 resize 到 128——**必须开**，否则 serving 路径（大图→缩略图）会有 train/serve skew
  （实测 CLI 侧 MAE 0.0838→**0.0147**）。
- 实测：CLI 6 点 sweep MAE **0.0147**；30 trial 端到端 PSNR 13.17dB（合成数据 oracle 上限 13.31dB，已打满）。
- Rust 侧 `crates/infer/src/onnx.rs` 用 tract 加载，`--mode auto`（默认）全分辨率重采样。

## 1. 数据

四个任务公开源头（与论文一致）：

- T1 人像：Tan et al. CVPR21（5 种超广角手机实拍 + 人工 GT，无像素 GT，用 ShapeACC/LineACC）。
- T2 广角整形：RecRecNet（Liao et al. ICCV23）。
- T3 拼接整形：Nie et al. CVPR22。
- T4 旋转校正：Nie et al. TIP23（ImageNet 派生）。

```powershell
python scripts/prepare_data.py --task t3 --src raw/t3 --dst data/processed/t3
# T1 有人脸 mask：加 --faces faces/ ；输出 img/ + prompt/ + target(可选) + meta.json
```

prompt 烘焙规则（与 Rust `prompt.rs` 对齐）：T2/T3 用 `border_mask`（暗边→0），T4 全 1，T1 用人脸 mask。

## 2. 模型

- `model/deformation.py` — RP-TPS 闭式解（double 精度 `torch.linalg.solve`），C0/C1 两步残差，
  **两次采样都从原图采**，head 零初始化，输出钳制 ±1.5。与 Rust `tps.rs` 同一数学。
- `model/restoration.py` — RMB×2 起步（论文×4，8G 先减半），首层拼 prompt。
- Loss（论文 Eq.9）：`L = La + a1*Lb + a2*0.01*Lp + a3*Lg`，
  `a1=1e-2, a2=1.0, a3=1e-2`；Lb 用 prompt 边缘加权，Lp 用二阶平滑 surrogate。
- 先训 DM、冻结后再联合微调（论文是同时训，分阶段更稳）。

## 3. 8G 训练配置（configs/unirect_lite.yaml）

- 分辨率 256，batch 2 × 累积 4（等效 8），AMP 开。
- Adam：lr 1e-4（旋转任务 1e-5），poly 0.96，wd 1e-5，200 epoch。
- 单任务先行（four-by-one），验收：T3 PSNR>23、T4>22.5、T2>19；同时记录跨任务退化矩阵（论文 Fig.4）。

```bash
# WSL2
python -m coolundistort.train --config configs/unirect_lite.yaml
# 断点续训（Kaggle 9h 接力）
python -m coolundistort.train --config configs/unirect_lite.yaml --resume checkpoints/unirect_lite_t3/epoch_010.pt
```

每 epoch 存 `epoch_NNN.pt`（含 optim，可续训），最优存 `best.pt`，TensorBoard 日志在 `ckpt/logs/`。

## 4. Kaggle 溢出指南

- 选 **T4 GPU**（P100 sm60 跑不了 mamba-ssm；先 `python -c "import mamba_ssm"` 验证）。
- 数据集挂 Kaggle Dataset，`checkpoints/` 每 epoch 存，单 session ≤9h，靠 `--resume` 接力。
- 30h/周只花在两种任务：本地跑不动的大实验、Linux 可复现验证。

## 5. 评测与导出

- `python scripts/eval.py --root data/processed/t3` 驱动 Rust CLI（PSNR+SSIM）输出到 `results/`。
  T1 用 ShapeACC/LineACC；T3 色调不一致（SSIM 虚低）与 T2 GT 模糊（感知指标虚低）以相对值为准。
- `python scripts/export_onnx.py --ckpt checkpoints/.../best.pt --out weights/...onnx`，
  再用 `coolundistort --mode checkpoint --onnx weights/...onnx` 在 Rust 内验证。
