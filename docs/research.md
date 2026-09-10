# Undistort 深度研究报告：从论文 2512.18718 (UniRect) 到 CoolUndistort 最合适工作流

> 论文：Rectification Reimagined: A Unified Mamba Model for Image Correction and Rectangling with Prompts
> arXiv: 2512.18718 (2025-12-21)，AAAI 2026 (aaai.v40i10.37811)
> 作者：Linwei Qiu, Gongzhe Li, Xiaozhe Zhang, Qilin Sun, Fengying Xie*（北航 / 天目山实验室 / 港中深）
> 代码宣称：https://github.com/yyywxk/UniRect（截至 2026-09-09 仍 404，未开源）

## TL;DR（一句话结论）

- 本论文的最大贡献不是单点精度，而是**视角统一**：把人像校正、广角整形、拼接整形、旋转校正四个任务的逆问题统一为一个**通用畸变模型**，再用**同一个网络结构 (UniRect)** 解决（four-by-one），并用 **Sparse MoE** 做成真正的四合一模型（four-in-one）。
- 对 `CoolUndistort` 而言**最合适的工作流不是直接复刻 UniRect**（代码未发布、357.9M 参数太重、7 天 V100 训练成本），而是**三段式**：
  1. **现在**：OpenCV 经典标定基线 + MOWA（已开源、有预训练、六合一）快速验证；
  2. **中期**：按 UniRect 思想自研轻量 `Deformation (RP-TPS) + Restoration (Mamba/CNN)` 双阶段 + Visual Prompt；
  3. **长期**：Sparse MoE 多任务统一 + 轻量化部署（ONNX / 移动端）。
- 本仓库目前是空脚手架（仅 `package.json`、无提交），且 JS 模板与视觉任务错配，**建议以 Python + PyTorch 为主仓，JS 只做 Demo/前端调用层**。

---

## 1. 论文精读：它到底解决了什么？

### 1.1 四个任务（都与手机摄影强相关）

- **T1 人像校正 Portrait correction**：广角端人脸拉伸。数据集：5 种超广角手机实拍 + 人工校正真值，无像素级 GT，只能用 LineACC / ShapeACC 及其 LLM 版评估。
- **T2 整流广角整形 Rectified wide-angle rectangling**：广角先做内容整流导致不规则边界，再整成矩形。数据集来自 RecRecNet (Liao et al. ICCV23)。
- **T3 拼接整形 Stitched rectangling**：全景拼接后的不规则边界整形。数据集来自 Nie et al. CVPR22（多为真实拼接结果 + 人工 GT）。
- **T4 旋转校正 Rotation correction**：任意角度旋转后保持矩形边界的内容校正。数据集来自 Nie et al. TIP23（ImageNet 派生 + 人工处理）。

传统做法是 four-to-four：四个任务四个专用网络。论文做到 four-by-one（同一结构、不同权重均 SOTA 级）→ four-in-one（同一权重同时处理四任务）。

### 1.2 通用畸变模型（General Distortion Model，核心数学贡献）

用光流（RAFT）可视化所有任务的逆问题 `T^{-1}`（即从真值到畸变的过程），发现都可以写成：

```
[xd, yd]^T = (1/r) * Σ(kj·θ^{2j-1} + k'j·r^{2j-1}) · Rα · [x, y]^T + T0
```

- 退化到 T1：令 k'=0, α=0, T0=0 → Kannala-Brandt 鱼眼模型。
- 退化到 T2：令 k=0, α=0, T0=0 → Brown-Conrady 径向模型。
- 退化到 T3：令 k=0, α=0 → Brown-Conrady + 偏心平移 T0（拼接边界导致光流中心偏离 O2→O'2）。
- 退化到 T4：令 k=0, T0=0 → Brown-Conrady + 旋转矩阵 Rα。

补充材料进一步给出蕴含关系图：T2 ⊂ T3（T3 = T2 + 偏心），T2 ⊂ T4（α→0 时）。这解释了跨任务退化实验（Fig.4）：在 T2 上训的模型 w2 迁移到 T3 退化最小；T1/T4 数据 t-SNE 纠缠，w1 迁移到 T4 最好。这是 MoE 必要性的理论依据。

### 1.3 UniRect 架构：DM + RM 双阶段

```
输入 (Xi0, Mi0) → [Deformation Module] → XiD, MiD → [Restoration Module] → XiR
```

**Deformation Module（几何整形，RP-TPS）：**

- 经典 TPS 用均匀分布的基础控制点 P + 预测的畸变点 P' 求闭式变换 Φ。单次 TPS 对复杂任务不够。
- **Residual Progressive TPS**：初值 c0 = 基础点；两个同结构预测器 C0, C1 预测残差偏移：
  `c1 = c0 + C0(X0,M0); X1 = S(G(c1); X0)`，`c2 = c1 + C1(X1,M1); XD = S(G(c2); X0)`。
- 关键细节：**两次采样都从最原始输入 X0 采样**，避免中间插值误差累积。作者试过递归重复 G+C，无提升还增加矩阵求逆开销。
- 控制点数默认 **12×10**，消融显示 14×12 后瓶颈且计算暴涨。
- C0/C1 内部 latent 有 **Mamba 块**扫描几何/边界长程依赖；消融 Mamba vs CNN (508.9M) vs Transformer (1.168G)：Mamba 精度最高且参数最小 (357.9M)。

**Visual Prompt（不是 task-id，是空间提示）：**

- T1：人脸 mask（关注脸部）；T2/T3：不规则边界 mask（参与 boundary loss 计算、可控整形位置）；T4：全白图（关注全局旋转）。
- 消融：去掉 prompt，T1/T3 大幅掉点，T2 轻微，T4 无影响。t-SNE 显示 prompt 把除 T1/T4 外的任务分布解耦。task-id 无法指示位置、无法参与 loss，故不用。

**Restoration Module（外观恢复，RMB）：**

- 4 个 Residual Mamba Block（每块 conv 32ch + 同款 Mamba）+ 首层 partial conv（处理 T2/T3 不规则边界；T1/T4 时退化为普通 conv）。
- 只用 RM 不用 DM 会失败：空白区被当有效信息，T3 语义全错。DM 定像素位置，RM 补采样/估计误差。
- 消融：T4 上 RM 提升巨大（PSNR 22.84→23.16，LPIPS 0.156→0.087）；T2 上 DM 主导，RM 主要改善感知指标。

**Loss：**

- DM：`L = Σγj(La + α1Lb + α2Lp + α3Lg)`，γ=0.9, α1=1e-2, α2=1.0, α3=1e-2。La=L1（仅 T1/T4 用全图 La，T2/T3 有边界变化不全用），Lb=边界损失（最外层控制点贴近 prompt 零水平集），Lp=线/形惩罚（防过整），Lg=梯度损失（对齐纹理）。T3 消融：+Lp/Lg 明显涨点，+RM 后 PSNR 21.86→25.08 但 SSIM 涨幅小（数据集本身色调不一致，SSIM 对结构敏感）。
- RM：appearance + perceptual loss（同超分做法）。

### 1.4 Sparse MoE（从 four-by-one 到 four-in-one）

- 直接混合训练（ML）或顺序训练（SL）都有严重任务竞争：任一单任务模型跨任务评估都大退化；SL 首任务主导（SL(3-2-4-1) 顺序不同结果迥异）。
- SMoE：5 个 UniRect 专家 + ResNet18 gating + Top-k (k=1)，`SMoE = Σ G(X0)j·Ej(X0,M0)`。k=1 最轻且最接近单任务性能，加大 k 训练不稳且无涨点。
- 结果（Tab.2）：SMoE 在 T2/T3/T4 几乎追平单任务（T2 19.90 / T3 25.07 / T4 23.16），T1 97.390 接近最优 97.409；ML 则崩到 T2 13.07 / T3 15.74。路由可视化显示 T1/T4 门控易混淆（与 t-SNE 纠缠一致），是残余损失来源。

### 1.5 定量结果速览（Tab.1 精选）

- T2：PSNR 19.90 (+1.2dB over RecRecNet 18.68), SSIM 0.5721；但 FID 27.02 / LPIPS 0.1245 略逊于 RecRecNet（GT 本身模糊，不适合 restoration 评测）。
- T3：PSNR 25.10 / SSIM 0.7526 / FID 19.59 / LPIPS 0.1120，全优于 Nie22 与 MOWA（MOWA T3 仅 20.42/0.6307）。
- T4：PSNR 23.16 / SSIM 0.7179 / FID 6.55 / LPIPS 0.0873，全优于 CoupledTPS (22.29/0.679)。
- T1：LineACC 66.523 / ShapeACC 97.454（略低于 Zhu22 的 66.825/97.491，作者归因于 TPS 不可逆导致关键点坐标估计误差）；但自提的 Qwen-VL 指标 LineACC-LLM 7.168 / ShapeACC-LLM 8.152 反超（该指标对 prompt 敏感、有幻觉，仅供参考）。
- 复杂度：357.9M / 62.98G FLOPs / 35.8 FPS（V100 语境），比 RecDiffusion（800M, >1T）轻得多，但仍重于专用小模型（Nie22 52M, RecRecNet 62.7M）。参数大头在 RM（>300M），可减块轻量化，甚至可不要 RM 处理四任务。

### 1.6 有趣应用与局限

- 同一输入给不同 prompt 可做可控整形（Fig.5 同图在 T3/T4 prompt 下输出不同任务结果）；实拍泛化（Fig.16：真实广角整流结果、经典拼接库、COCO 旋转）不错。
- 局限（作者自认）：计算量高于专用模型；多畸变共存单图无 GT 数据集，无法验证；T1/T4 门控混淆；LLM 指标不稳定。

---

## 2. 复现现状：UniRect 代码仍未发布

- 作者主页 `github.com/yyywxk` 现仅有 IWKFormer（遥感整形 Transformer），**UniRect 仓 404**。AAAI 2026 刚收录，可预期 code release 会延迟，不要 dry-wait。
- 四个数据集均有公开源头可独立复现：Tan21 人像、Liao23 RecRecNet、Nie22 DeepRectangling、Nie23 DRC。合成多畸变数据需人工标定+筛选，成本高。
- 训练成本：单任务 V100×4 batch4 200epoch 需 1–3 天；四任务联合 batch10 200epoch 约 7 天。个人/小团队直接复训四合一不现实。

## 3. 竞品横向对比（Undistort 相关）

| 路线 | 代表 | 任务数 | 开源/预训练 | 优点 | 缺点 | 适合 CoolUndistort 的角色 |
|---|---|---|---|---|---|---|
| 多合一 warping | **MOWA (TPAMI25, KangLiao929/MOWA)** | 6（含人像/广角/拼接/旋转/rolling shutter 等） | ✅ 开源+预训练+ONNX 社区版 | 唯一可直接跑的多任务基线；统一 motion 表示；跨域/zero-shot 好 | 不含 UniRect 的数学统一模型；精度在 T3/T4 被 UniRect 超 | **P0：立即验证基线** |
| 单任务 TPS 整形 | RecRecNet (ICCV23) | T2 | ✅ | TPS+DoF 课程学习，思想被 UniRect 继承 | 仅单任务 | T2 专用头 / RP-TPS 参考 |
| 单任务拼接整形 | Nie22 DeepRectangling / Zhou24 RecDiffusion | T3 | 部分开源 | Nie22 轻（52M, 20FPS）；RecDiffusion 质量高 | RecDiffusion 800M+双扩散，太重 | 轻量对照组 |
| 单任务旋转 | DRC (TIP23) / CoupledTPS (TPAMI24) | T4 | 部分开源 | CoupledTPS 半监督 TPS，UniRect 直接对比对象 | 单任务 | T4 专用头 |
| 通用 restoration backbone | MambaIR/MambaIRv2, VMamba, VmambaIR | 通用 | ✅ | UniRect RM 的理论上游；线性复杂度长程建模 | 不做几何 deformation，需配合 warping | RM 模块选型依据 |
| 经典几何 | OpenCV fisheye / Brown-Conrady 标定 | 任意已知相机 | ✅ | 零训练、可解释、移动端成熟 | 需标定板/已知参数；对未知网图/拼接边界无能为力 | **P0：已知相机首选** |

结论：UniRect vs MOWA 是理解关键——两者都做多合一，但 UniRect 有**畸变方程统一 + RP-TPS 渐进 + Mamba + SMoE** 四件套，精度更高；MOWA 有**代码+权重+生态**，工程可用。前者是目标，后者是起点。

## 4. Undistort 工作流全景

```
A. 已知相机（标定板/出厂参数可用）
   标定(K, D) → OpenCV fisheye::undistort / initUndistortRectifyMap → 双线性重采样 → 边界裁剪/补全
   成本最低、质量最稳，手机/工业相机首选。

B. 未知单图单任务（网图、拼接结果、旋转图）
   任务判别 → 专用 warping 网络 (RecRecNet/Nie22/DRC) 或 MOWA 单任务推理
   → 网格/TPS 采样 → 感知/保真后处理。无需标定，但需对齐该任务数据分布。

C. 未知多任务统一（UniRect/MOWA 范式，推荐长期）
   输入 + Visual Prompt → DM(RP-TPS 两步残差) → RM(Mamba 恢复) → 输出
   → SMoE 门控多专家 → 四合一/六合一推理。一次部署覆盖全机型，适合端侧。
```

选择逻辑：**有参数用 A，无参数看任务数量**——单任务用 B（轻），多任务/端侧用 C（统一）。

## 5. 给 CoolUndistort 的最合适工作流（分阶段、可执行）

### Phase 0 — 仓库纠偏（半天）

- 本仓 `package.json (coolundistort, commonjs)` 与视觉任务错配。建议：主仓转 Python (`pyproject.toml` + `torch` + `openmim` 生态），保留 JS 仅作 `demo/` 前端上传+调用推理 API。
- 建目录：`configs/ data/ coolundistort/{deform,restore,gating,losses}/ scripts/ demo/ tests/ docs/`，先放 OpenCV 基线 + MOWA 推理脚本。

### Phase 1 — 两条基线并行（1–2 周，P0）

1. **经典基线**：棋盘格标定 → `cv2.fisheye.calibrate + undistortImage`，输出 K/D 与 remap，供后续 warp 初始化与评测下限。
2. **深度基线**：conda 复跑 MOWA 预训练（`test.py / test_portrait.py`），在 T1–T4 公开测试集上记录 PSNR/SSIM/FID/LPIPS 与 FPS，作为所有自研模型的对比锚点。社区已有 `MOWA-onnxrun`，可直接验证 ONNX 端侧延迟。

### Phase 2 — 自研 UniRect-lite（核心，4–8 周）

- **DM**：实现 RP-TPS（12×10 控制点，两步残差、均从原图采样；TPS 求解器参考 RecRecNet/CoupledTPS 开源实现），C0/C1 用轻 backbone + 1 个 Mamba 块（先用 `mamba-ssm` 或 VMamba 实现，不行则降级为 Swin/CNN 对照）。
- **Prompt**：四任务 prompt 生成器（人脸检测 mask / 边界 mask / 全白图），prompt 参与 `Lb` 计算；推理 API 必传 `task: t1|t2|t3|t4 + prompt`。
- **RM**：2–4 个 RMB（先 2 个控参数），首层 partial conv；先训 DM 冻结再联合微调（论文同时训练，但分阶段更稳）。
- **Loss**：照抄 Eq.9 权重起步（γ0.9, 1e-2/1.0/1e-2），T1/T4 全图 La，T2/T3 边界加权。
- **数据**：逐任务单训复现 Tab.1（four-by-one），用 MOWA 数据清洗脚本统一数据结构。
- **验收**：T3 PSNR>23、T4 PSNR>22.5、T2 PSNR>19 即达标；记录跨任务退化矩阵（复刻 Fig.4）证明任务竞争存在。

### Phase 3 — 四合一与轻量化（长期）

- SMoE：5×UniRect-lite 专家 + ResNet18 gating Top-1，联合微调；重点攻 T1/T4 门控混淆（可加 prompt 嵌入辅助分类）。
- 压缩：RM 减块/通道剪枝/蒸馏 → 目标 <100M；导出 ONNX，转手机端（参考 MOWA-onnxrun 路线）。
- 缺失数据：暂不做多畸变共存单图（无 GT），仅保留 prompt 可控整形 Demo。

### 风险清单

- UniRect 开源跳票 → 已用 MOWA+RecRecNet 对冲，不阻塞。
- Mamba 在 Windows/端侧算子支持差 → 保留纯 CNN/Transformer 对照（Tab.5 已证明 Mamba 最优但 CNN 可用）。
- T1 无像素 GT、T3 色调不一致 → 评测以 SSIM/LPIPS/人工 + LLM 指标联合为准，不唯 PSNR。
- 7 天级训练成本 → 先单任务小图（256px）验证 RP-TPS 收敛，再放大分辨率。

---

## 附：关键超参备忘（照论文起步）

- 控制点 12×10；Adam, lr 1e-4（旋转 1e-5）, poly 0.96, wd 1e-5；单任务 batch4/200epoch，多任务 batch10/200epoch(+50epoch 对齐)；SMoE: ResNet18 gating, k=1, 5 专家。
