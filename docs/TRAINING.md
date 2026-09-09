# 训练文档（Python 专用；推理全部在 Rust）

> 对应论文 arXiv:2512.18718。目标：在 8G -ish 显卡上复现 four-by-one，再考虑 four-in-one。
> 本目录只管训练出权重；**一切推理（含基线/评测/部署）都走 `crates/infer` + `crates/cli`**，
> 训练产物经 ONNX 导出给 Rust 加载（P1）。

## 1. 数据

四个任务公开源头（与论文一致）：

- T1 人像：Tan et al. CVPR21（5 种超广角手机实拍 + 人工 GT，无像素 GT，用 ShapeACC/LineACC）。
- T2 广角整形：RecRecNet（Liao et al. ICCV23）。
- T3 拼接整形：Nie et al. CVPR22。
- T4 旋转校正：Nie et al. TIP23（ImageNet 派生）。

目录约定（已 ignore，大文件勿提交）：

```
data/raw/{t1,t2,t3,t4}/        # 原始下载
data/processed/{t1,t2,t3,t4}/  # 统一结构：img/ + prompt/ + meta.json
```

MOWA 仓库自带六任务清洗脚本，数据结构可直接参考对齐（MOWA 权重仅本地评测，不分发）。

## 2. 模型（P1 实现顺序）

1. `model/deformation.py` — RP-TPS：基础控制点 12×10，C0/C1 两步残差，**两次采样都从原图采**（防插值累积）。TPS 求解参考 RecRecNet / CoupledTPS 开源实现。
2. `model/prompts.py` — T1 人脸 mask / T2-T3 边界 mask / T4 全白图；prompt 参与 `Lb`。
3. `model/restoration.py` — RMB×2 起步（论文×4，>300M 参数，8G 先减半），首层 partial conv。
4. 先训 DM、冻结后再联合微调（论文是同时训，分阶段更稳）。

Loss（论文 Eq.9 起步）：`L = Σγj(La + α1Lb + α2Lp + α3Lg)`，
`γ=0.9, α1=1e-2, α2=1.0, α3=1e-2`；T1/T4 全图 La，T2/T3 边界加权；RM 加 perceptual。

## 3. 8G 训练配置（configs/unirect_lite.yaml）

- 分辨率 256，batch 2 × 累积 4（等效 8），AMP 开。
- Adam：lr 1e-4（旋转任务 1e-5），poly 0.96，wd 1e-5，200 epoch。
- 单任务先行（four-by-one），验收：T3 PSNR>23、T4>22.5、T2>19；同时记录跨任务退化矩阵（论文 Fig.4）。

```bash
# WSL2
python -m coolundistort.train --config configs/unirect_lite.yaml
```

## 4. Kaggle 溢出指南

- 选 **T4 GPU**（P100 sm60 跑不了 mamba-ssm；T4 也可能要降级。
  先 `python -c "import mamba_ssm"` 验证，不行就用 CNN 对照分支）。
- 数据集挂 Kaggle Dataset，`checkpoints/` 每 epoch 存，单 session ≤9h，靠断点续训接力。
- 30h/周只花在两种任务：本地跑不动的大实验、Linux 可复现验证。

## 5. 评测

PSNR/SSIM/FID/LPIPS；T1 用 ShapeACC/LineACC（像素 GT 不存在）。
注意 T3 数据集色调不一致（SSIM 虚低）与 T2 GT 模糊（感知指标虚低），以相对值为准。
`scripts/eval.py` 驱动 Rust CLI（`cargo run -p coolundistort-cli -- --eval`）输出 JSON 到 `results/`；
Python 侧不做推理实现。
